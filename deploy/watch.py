"""Lässt einen trainierten Checkpoint im Simulator gegen sich selbst spielen und zeigt es an.

    python deploy/watch.py                      # neuester Checkpoint, 1v1
    python deploy/watch.py --team-size 2 --episodes 5
    python deploy/watch.py --checkpoint runs/lucy_1v1/checkpoints/125222784

Die Anzeige läuft über RLViser (eigenes Fenster, startet automatisch). Rocket League selbst
wird dafür nicht gebraucht. Beide Teams spielen dieselbe Policy.

Gespielt wird mit derselben Beobachtung, derselben Aktionstabelle und derselben Inferenz wie
im Training (siehe Paritätstests), nur eben in Python.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.action_table import LOOKUP_TABLE  # noqa: E402
from deploy.policy import load_policy  # noqa: E402
from deploy.rlgym_adapter import view_from_state  # noqa: E402
from env.obs_python import build_obs  # noqa: E402

TICK_SKIP = 8
MAX_PLAYERS = 3
ACTION_STACK = 5
ROOT = Path(__file__).resolve().parents[1]


def check_no_other_instance() -> None:
    """RLViser bindet einen festen UDP-Port; eine zweite Instanz stürzt sonst als Rust-Panic ab.

    Der eigene Prozessbaum muss dabei ausgenommen werden: Das python.exe im venv startet den
    eigentlichen Interpreter als Kindprozess, beide haben watch.py in der Kommandozeile.
    """
    import psutil

    me = psutil.Process()
    own = {me.pid}
    try:
        own |= {p.pid for p in me.parents()}
        own |= {c.pid for c in me.children(recursive=True)}
    except psutil.Error:
        pass

    others = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"] or []
            if proc.info["pid"] not in own and any("watch.py" in part for part in cmdline):
                others.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if others:
        raise SystemExit(
            f"Es läuft bereits ein watch.py (PID {', '.join(map(str, others))}).\n"
            f"RLViser kann nur einmal laufen. Erst beenden:\n"
            f"  Get-CimInstance Win32_Process | "
            f"Where-Object {{ $_.CommandLine -like '*watch.py*' }} | "
            f"ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"
        )


DEFAULT_RUN = ROOT / "runs" / "lucy_1v1"


def latest_checkpoint(run_dir: Path = DEFAULT_RUN) -> Path:
    """Neuester Checkpoint EINES Laufs (Audit M5): numerisch nach Step-Zahl sortiert.

    Vorher wurde über alle Läufe gesucht, auch über `runs/archive/*` mit anderer Netzgröße.
    """
    folder = run_dir / "checkpoints" if (run_dir / "checkpoints").exists() else run_dir
    candidates = sorted(folder.glob("*/PPO_POLICY.lt"), key=lambda p: int(p.parent.name))
    if not candidates:
        raise SystemExit(f"Kein Checkpoint in {run_dir} (erwartet <run>/checkpoints/<steps>/PPO_POLICY.lt)")
    return candidates[-1]


def build_env(team_size: int, render: bool):
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (AnyCondition, GoalCondition,
                                                      NoTouchTimeoutCondition, TimeoutCondition)
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (FixedTeamSizeMutator, KickoffMutator,
                                                     MutatorSequence)

    # Unsere Aktionstabelle statt der von RLGym: Der Policy-Index muss dieselbe Eingabe bedeuten.
    action_parser = LookupTableAction()
    action_parser._lookup_table = LOOKUP_TABLE.copy()

    renderer = None
    if render:
        from rlgym.rocket_league.rlviser import RLViserRenderer
        renderer = RLViserRenderer(tick_rate=120 / TICK_SKIP)

    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size),
            KickoffMutator(),
        ),
        # Der DefaultObs wird nicht benutzt (wir bauen die Beobachtung selbst), muss aber
        # gesetzt sein, damit RLGym die Umgebung baut.
        obs_builder=DefaultObs(zero_padding=3),
        action_parser=RepeatAction(action_parser, repeats=TICK_SKIP),
        reward_fn=GoalReward(),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=30.0),
            TimeoutCondition(timeout_seconds=300.0),
        ),
        transition_engine=RocketSimEngine(),
        renderer=renderer,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, help="Standard: neuester Checkpoint aus --run")
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN,
                    help="Lauf, aus dem der neueste Checkpoint genommen wird")
    ap.add_argument("--team-size", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--deterministic", action="store_true",
                    help="Immer die wahrscheinlichste Aktion (sonst wird gesampelt wie im Training)")
    ap.add_argument("--no-render", action="store_true", help="Nur Statistik, kein Fenster")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = Echtzeit, 2.0 = doppelt so schnell")
    a = ap.parse_args()

    if not a.no_render:
        check_no_other_instance()

    checkpoint = a.checkpoint or latest_checkpoint(a.run)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "PPO_POLICY.lt"
    policy = load_policy(checkpoint)
    print(f"Policy: {checkpoint}")
    print(f"  Netz {policy.meta.layer_sizes}, Obs {policy.meta.obs_size}, "
          f"Aktionen {policy.meta.action_count}, Steps {policy.meta.timesteps:,}")

    env = build_env(a.team_size, render=not a.no_render)
    step_time = TICK_SKIP / 120.0 / max(a.speed, 0.01)

    total_touches = 0
    total_steps = 0
    goals = 0

    for episode in range(a.episodes):
        env.reset()
        state = env.state
        history: dict[int, deque[np.ndarray]] = {}
        steps = 0
        touches = 0

        while True:
            view, ids = view_from_state(state)
            view.action_history = {cid: list(h) for cid, h in history.items()}

            actions = {}
            for agent_id, car_id in ids.items():
                obs = build_obs(view, view.players[car_id], MAX_PLAYERS, ACTION_STACK)
                index = int(policy.act(obs, deterministic=a.deterministic))
                actions[agent_id] = np.array([index], dtype=np.int64)
                history.setdefault(car_id, deque(maxlen=ACTION_STACK)).append(
                    LOOKUP_TABLE[index].astype(np.float64))

            _, _, terminated, truncated = env.step(actions)
            state = env.state
            steps += 1
            touches += sum(car.ball_touches for car in state.cars.values())

            if not a.no_render:
                env.render()
                time.sleep(step_time)

            if any(terminated.values()) or any(truncated.values()):
                scored = any(terminated.values())
                goals += scored
                print(f"Episode {episode + 1}: {steps} Schritte "
                      f"({steps * TICK_SKIP / 120:.0f} s Spielzeit), {touches} Ballkontakte, "
                      f"{'TOR' if scored else 'kein Tor (Timeout)'}")
                total_steps += steps
                total_touches += touches
                break

    env.close()
    print(f"\n{a.episodes} Episoden: {goals} Tore, {total_touches} Ballkontakte, "
          f"Kontaktrate {total_touches / max(1, total_steps * 2 * a.team_size):.4f}")


if __name__ == "__main__":
    main()
