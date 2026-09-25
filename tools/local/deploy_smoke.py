"""Deployment-Smoke-Test ohne Rocket League (LOCAL_RUNBOOK Schritt 6):

    python tools/local/deploy_smoke.py --run runs/lucy_1v1 --out results/local_check_<datum>/deploy

1. neuesten Checkpoint des Laufs exportieren (tools/export_policy.py-Logik) nach <out>/policy.pt
2. exportierte Policy laden, Obs-Größe und Aktionszahl gegen den Obs-Builder prüfen (Audit M7)
3. eine Beobachtung aus einem synthetischen RLBot-Paket bauen und eine Inferenz ausführen
4. Latenz messen: Paket -> Obs -> Inferenz -> ControllerState, Median/p95 über N Wiederholungen
5. Ergebnis als JSON (+ Ausgabe), Rückgabe 0 = alles bestanden

Die Latenz hängt von der Maschine ab und wird nur lokal gemessen; Referenz: eine Entscheidung
muss deutlich unter 8 Ticks / 120 Hz = 66 ms liegen, damit der Bot keinen Entscheidungs-Frame
verpasst (deploy/rlbot/bot.py entscheidet alle 8 Ticks).
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from deploy.action_table import LOOKUP_TABLE  # noqa: E402
from deploy.policy import export_policy, load_policy  # noqa: E402
from env.obs_python import BOOST_LOCATIONS, obs_size  # noqa: E402

MAX_PLAYERS, ACTION_STACK, TICK_SKIP = 3, 5, 8
BUDGET_MS = TICK_SKIP / 120.0 * 1000.0


def latest_checkpoint(run_dir: Path) -> Path:
    folder = run_dir / "checkpoints" if (run_dir / "checkpoints").exists() else run_dir
    cands = sorted((d for d in folder.glob("*") if d.is_dir() and d.name.isdigit() and (d / "PPO_POLICY.lt").exists()),
                   key=lambda d: int(d.name))
    if not cands:
        raise SystemExit(f"Kein Checkpoint in {run_dir}")
    return cands[-1]


def synthetic_packet():
    """Ein RLBot-Paket wie in tests/test_packet_adapter.py; braucht rlbot_flatbuffers."""
    import rlbot_flatbuffers as flat

    def player(pos, team):
        return flat.PlayerInfo(
            physics=flat.Physics(location=flat.Vector3(*pos), velocity=flat.Vector3(0, 0, 0),
                                 angular_velocity=flat.Vector3(0, 0, 0), rotation=flat.Rotator(0, 0, 0)),
            team=team, boost=33.0, air_state=flat.AirState.OnGround, has_jumped=False,
            has_double_jumped=False, has_dodged=False, dodge_timeout=-1.0, demolished_timeout=-1.0,
            is_supersonic=False)

    pads = [flat.BoostPadState(is_active=True, timer=0.0) for _ in BOOST_LOCATIONS]
    ball = flat.BallInfo(physics=flat.Physics(location=flat.Vector3(0, 0, 93), velocity=flat.Vector3(0, 0, 0),
                                              angular_velocity=flat.Vector3(0, 0, 0), rotation=flat.Rotator(0, 0, 0)))
    return flat.GamePacket(balls=[ball], players=[player((-2048, -2560, 17), 0), player((2048, 2560, 17), 1)],
                           boost_pads=pads, match_info=flat.MatchInfo(frame_num=0, match_phase=flat.MatchPhase.Active))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "lucy_1v1")
    ap.add_argument("--checkpoint", type=Path, default=None, help="statt neuestem Checkpoint des Laufs")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "deploy_smoke")
    ap.add_argument("--iterations", type=int, default=200)
    a = ap.parse_args()

    result: dict = {"ok": False, "machine": platform.platform(), "python": sys.version.split()[0]}
    try:
        ckpt = a.checkpoint or latest_checkpoint(a.run)
        result["checkpoint"] = str(ckpt)
        a.out.mkdir(parents=True, exist_ok=True)

        # 1. Export
        policy_path = a.out / "policy.pt"
        meta = export_policy(ckpt, policy_path)
        result["export"] = {"path": str(policy_path), "obs_size": meta.obs_size,
                            "action_count": meta.action_count, "layer_sizes": meta.layer_sizes,
                            "timesteps": meta.timesteps}
        print(f"Exportiert: {policy_path} (Obs {meta.obs_size}, Aktionen {meta.action_count}, "
              f"Netz {meta.layer_sizes}, {meta.timesteps:,} Steps)")

        # 2. Laden + Prüfung (wie deploy/rlbot/bot.py)
        policy = load_policy(policy_path)
        expected = obs_size(MAX_PLAYERS, ACTION_STACK)
        if policy.meta.obs_size != expected:
            raise ValueError(f"Obs-Größe {policy.meta.obs_size} != {expected}")
        if policy.meta.action_count != len(LOOKUP_TABLE):
            raise ValueError(f"Aktionen {policy.meta.action_count} != {len(LOOKUP_TABLE)}")
        result["checks"] = {"obs_size": expected, "action_count": len(LOOKUP_TABLE)}
        print(f"Obs-Größe {expected} und {len(LOOKUP_TABLE)} Aktionen passen zur Policy.")

        # 3./4. Obs + Inferenz + Latenz über den echten Deployment-Pfad
        from deploy.packet_adapter import build_pad_index_map, view_from_packet
        from deploy.rlbot.bot import RLbotAgent
        from env.obs_python import build_obs

        packet = synthetic_packet()
        pad_map = build_pad_index_map(BOOST_LOCATIONS)
        history: list[np.ndarray] = []

        def decide():
            view = view_from_packet(packet, pad_map, action_history={0: history[-ACTION_STACK:]})
            obs = build_obs(view, view.players[0], MAX_PLAYERS, ACTION_STACK)
            idx = int(policy.act(obs, deterministic=True))
            action = LOOKUP_TABLE[idx]
            history.append(action.astype(np.float64))
            return RLbotAgent._to_controller(action), idx, obs

        ctrl, idx, obs = decide()
        if obs.shape != (expected,) or not np.all(np.isfinite(obs)):
            raise ValueError("Obs hat falsche Form oder ist nicht endlich")
        result["inference"] = {"action_index": idx, "throttle": ctrl.throttle, "steer": ctrl.steer,
                               "boost": bool(ctrl.boost), "jump": bool(ctrl.jump)}
        print(f"Eine Inferenz: Aktion {idx} -> throttle {ctrl.throttle:+.1f} steer {ctrl.steer:+.1f} "
              f"boost {bool(ctrl.boost)} jump {bool(ctrl.jump)}")

        for _ in range(10):
            decide()
        times = []
        for _ in range(a.iterations):
            t0 = time.perf_counter()
            decide()
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        lat = {"iterations": a.iterations, "median_ms": statistics.median(times),
               "p95_ms": times[int(0.95 * (len(times) - 1))], "max_ms": times[-1],
               "budget_ms": BUDGET_MS}
        result["latency"] = lat
        print(f"Latenz Paket->Obs->Inferenz->Controller: Median {lat['median_ms']:.2f} ms, "
              f"p95 {lat['p95_ms']:.2f} ms, max {lat['max_ms']:.2f} ms (Budget {BUDGET_MS:.1f} ms je Entscheidung)")
        if lat["p95_ms"] > BUDGET_MS:
            print("WARNUNG: p95 über dem Entscheidungsbudget - der Bot würde Frames verpassen.")
        result["ok"] = lat["p95_ms"] <= BUDGET_MS
    except Exception as e:  # noqa: BLE001 - alles ins Ergebnis, damit die Zusammenfassung es zeigt
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"FEHLER: {result['error']}")
    finally:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "deploy_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Ergebnis: {a.out / 'deploy_smoke.json'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
