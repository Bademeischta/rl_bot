"""Anstoß-Auswertung im Python-RocketSim über den Deployment-Pfad (Spieltest-Auftrag, Punkt 1).

    python eval/kickoff_eval.py --a policy:runs/lucy_1v1/checkpoints/6037692544@argmax --b policy:<pfad> --kickoffs 1000
    python eval/kickoff_eval.py --a script:speedflip+policy:<pfad> --b policy:<pfad> --out result.json
    python eval/kickoff_eval.py --trajectory policy:<pfad>@argmax          # Fahrweg, Flip, Boost je Anstoßart
    python eval/kickoff_eval.py --tune                                       # Speedflip-Parameter suchen

Spieler-Angaben:
  policy:<checkpoint oder .pt>[@sample|argmax|argmax_group]   gelernte Policy (Standard: sample)
  script:<boost|frontflip|speedflip>[+policy:...]            geskripteter Anstoß, nach der ersten
                                                             Berührung übernimmt die Policy
  idle                                                       steht still

Schrittfolge wie RLGymSim Gym::Step und damit wie Training und duel.exe: Die Policy entscheidet alle
8 Ticks auf dem Zustand 1 Tick nach Beginn des Schritts (Snapshot), die Aktion gilt 8 Ticks.
Beobachtung (env/obs_python.py), Aktionstabelle und Inferenz (deploy/policy.py) sind die des Bots.
Ein Skript steuert pro Tick.

Je Anstoß (Definitionen wie env/cpp/PlayStats.h): erste Berührung (Seite, Zeit, Tempo), Ballhälfte
und Ballnähe 3 s danach, Tor innerhalb von 10 s. Die Seiten wechseln von Anstoß zu Anstoß, die fünf
Anstoßpositionen kommen gleich oft vor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import RocketSim as rs  # noqa: E402

from deploy.action_select import select  # noqa: E402
from deploy.action_table import LOOKUP_TABLE  # noqa: E402
from deploy.policy import Policy, load_policy  # noqa: E402
from deploy.scripted_kickoff import KickoffCar, ScriptedKickoff  # noqa: E402
from env.obs_python import BOOST_LOCATIONS, BallView, GameView, PlayerView, build_obs  # noqa: E402

TICK_SKIP = 8
MAX_PLAYERS, ACTION_STACK = 3, 5
POSSESSION_SECS, GOAL_SECS = 3.0, 10.0
GOAL_Y = 5124.25
SPAWN_NAMES = {(-2048, -2560): "diagonal_links", (2048, -2560): "diagonal_rechts",
               (-256, -3840): "versetzt_links", (256, -3840): "versetzt_rechts", (0, -4608): "mitte"}

_rs_ready = False


def init_rocketsim() -> None:
    global _rs_ready
    if not _rs_ready:
        import rlgym.rocket_league.sim.rocketsim_engine as engine
        rs.init(os.path.join(os.path.dirname(engine.__file__), "collision_meshes"))
        _rs_ready = True


def _v(t) -> np.ndarray:
    return np.array(t, dtype=np.float64)


# --- Spieler --------------------------------------------------------------------------------

_policy_cache: dict[str, Policy] = {}


def _load(path: str) -> Policy:
    if path not in _policy_cache:
        _policy_cache[path] = load_policy(path)
    return _policy_cache[path]


@dataclass
class Spec:
    kind: str                     # "policy", "script", "idle"
    path: str = ""
    mode: str = "sample"
    variant: str = ""
    after: "Spec | None" = None   # Policy nach dem Skript

    @staticmethod
    def parse(text: str) -> "Spec":
        if text == "idle":
            return Spec("idle")
        if text.startswith("script:"):
            body = text[len("script:"):]
            variant, _, rest = body.partition("+")
            return Spec("script", variant=variant, after=Spec.parse(rest) if rest else None)
        if text.startswith("policy:"):
            body = text[len("policy:"):]
            path, _, mode = body.rpartition("@") if "@" in body else (body, "", "sample")
            return Spec("policy", path=path, mode=mode or "sample")
        raise ValueError(f"Unbekannte Spieler-Angabe {text!r}")

    def __str__(self) -> str:
        if self.kind == "idle":
            return "idle"
        if self.kind == "script":
            return f"script:{self.variant}" + (f"+{self.after}" if self.after else "")
        return f"policy:{self.path}@{self.mode}"


# --- Simulation -----------------------------------------------------------------------------

class KickoffSim:
    def __init__(self, team_size: int = 1):
        init_rocketsim()
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.cars = []
        for _ in range(team_size):          # Reihenfolge wie RLGymSim Gym: Blau, Orange, Blau, ...
            self.cars.append(self.arena.add_car(rs.Team.BLUE))
            self.cars.append(self.arena.add_car(rs.Team.ORANGE))
        pads = self.arena.get_boost_pads()
        self.pads = pads
        self.pad_map = [int(np.argmin([np.linalg.norm(_v(p.get_pos())[:2] - loc[:2]) for p in pads]))
                        for loc in BOOST_LOCATIONS]

    @staticmethod
    def team(i: int) -> int:
        return i % 2

    def view(self) -> GameView:
        players = []
        for i, c in enumerate(self.cars):
            s = c.get_state()
            has_flip = not s.has_double_jumped and not s.has_flipped and s.air_time_since_jump < 1.25
            players.append(PlayerView(
                car_id=i, team=self.team(i), pos=_v(s.pos), forward=_v(s.rot_mat.forward), up=_v(s.rot_mat.up),
                vel=_v(s.vel), ang_vel=_v(s.ang_vel), boost=s.boost / 100.0, on_ground=bool(s.is_on_ground),
                has_flip=bool(has_flip), has_jump=not s.has_jumped, demoed=bool(s.is_demoed),
                supersonic=bool(s.is_supersonic), flipping=bool(s.is_flipping), jumping=bool(s.is_jumping)))
        b = self.arena.ball.get_state()
        return GameView(ball=BallView(pos=_v(b.pos), vel=_v(b.vel), ang_vel=_v(b.ang_vel)), players=players,
                        pad_timers=np.array([self.pads[j].get_state().cooldown for j in self.pad_map]))

    def car_view(self, i: int) -> KickoffCar:
        s = self.cars[i].get_state()
        return KickoffCar(pos=_v(s.pos), forward=_v(s.rot_mat.forward), right=_v(s.rot_mat.right),
                          up=_v(s.rot_mat.up), vel=_v(s.vel), on_ground=bool(s.is_on_ground))


def _controls(a) -> rs.CarControls:
    c = rs.CarControls()
    c.throttle, c.steer, c.pitch, c.yaw, c.roll = (float(x) for x in a[:5])
    c.jump, c.boost, c.handbrake = bool(a[5] >= 0.5), bool(a[6] >= 0.5), bool(a[7] >= 0.5)
    return c


def _nearest_entry(a) -> np.ndarray:
    return LOOKUP_TABLE[int(np.argmin(np.abs(LOOKUP_TABLE - np.asarray(a, dtype=np.float32)).sum(1)))]


def spawn_seeds(count_per_spawn: int = 1) -> dict[str, list[int]]:
    """Seeds für reset_kickoff je Anstoßposition (aus Blaus Sicht)."""
    init_rocketsim()
    arena = rs.Arena(rs.GameMode.SOCCAR)
    blue = arena.add_car(rs.Team.BLUE)
    arena.add_car(rs.Team.ORANGE)
    out: dict[str, list[int]] = {n: [] for n in SPAWN_NAMES.values()}
    seed = 0
    while min(len(v) for v in out.values()) < count_per_spawn:
        arena.reset_kickoff(seed)
        p = blue.get_state().pos
        name = SPAWN_NAMES[(round(p[0]), round(p[1]))]
        if len(out[name]) < count_per_spawn:
            out[name].append(seed)
        seed += 1
    return out


def play_kickoff(sim: KickoffSim, seed: int, specs: list[Spec], rng: np.random.Generator,
                 seconds: float = GOAL_SECS, trace: bool = False) -> dict:
    """Ein Anstoß; specs[i] steuert Auto i (gerade = Blau). Ergebnis wie env/cpp/PlayStats.h."""
    sim.arena.reset_kickoff(seed)
    t0 = sim.arena.tick_count
    n = len(sim.cars)
    history: list[list[np.ndarray]] = [[] for _ in range(n)]
    scripts = [ScriptedKickoff(variant=s.variant) if s.kind == "script" else None for s in specs]
    held: list = [None] * n
    snap = sim.view()
    spawn = SPAWN_NAMES[(round(snap.players[0].pos[0]), round(snap.players[0].pos[1]))]
    last_speed = [float(np.linalg.norm(p.vel)) for p in snap.players]
    boost_used = [0.0, 0.0]
    last_boost = [p.boost * 100 for p in snap.players]
    res = dict(spawn=spawn, first_team=-1, time_to_touch=None, touch_speed=None, ball_half_team=-1,
               closer_team=-1, goal_team=-1)
    touched = False
    touch_tick = None
    frames = []
    total_ticks = int(seconds * 120)
    for tick in range(total_ticks):
        if tick % TICK_SKIP == 0:
            # Entscheidung auf dem Snapshot (bei tick 0 der Reset-Zustand), gilt 8 Ticks
            snap.action_history = {i: history[i][-ACTION_STACK:] for i in range(n)}
            for i, spec in enumerate(specs):
                active = spec.after if (spec.kind == "script" and touched) else spec
                if active is None or active.kind == "idle":
                    held[i] = np.zeros(8, dtype=np.float32)
                elif active.kind == "policy":
                    probs = _load(active.path).action_probs(build_obs(snap, snap.players[i], MAX_PLAYERS, ACTION_STACK))
                    held[i] = LOOKUP_TABLE[select(probs, active.mode, snap.players[i].on_ground, rng)]
                else:
                    held[i] = None                     # Skript: pro Tick
                if held[i] is not None:
                    history[i].append(held[i].astype(np.float64))
        ball_pos = _v(sim.arena.ball.get_state().pos)
        for i, c in enumerate(sim.cars):
            if held[i] is None:
                a = scripts[i].step(sim.car_view(i), ball_pos)
                c.set_controls(_controls(a))
                if tick % TICK_SKIP == TICK_SKIP - 1:
                    history[i].append(_nearest_entry(a).astype(np.float64))
            else:
                c.set_controls(_controls(held[i]))
        sim.arena.step(1)
        if tick % TICK_SKIP == 0:
            snap = sim.view()
        # Erste Berührung (tickgenau), Boost bis dahin
        states = [c.get_state() for c in sim.cars]
        if not touched:
            for i, s in enumerate(states):
                used = last_boost[i] - s.boost
                if used > 0:
                    boost_used[sim.team(i)] += used / (n // 2)
                last_boost[i] = s.boost
            hits = [(s.ball_hit_info.tick_count_when_hit, i) for i, s in enumerate(states)
                    if s.ball_hit_info.is_valid and s.ball_hit_info.tick_count_when_hit >= t0]
            if hits:
                hit_tick, first = min(hits)
                touched = True
                touch_tick = tick
                res.update(first_team=sim.team(first), time_to_touch=(hit_tick - t0) / 120.0,
                           touch_speed=last_speed[first])
            last_speed = [float(np.linalg.norm(_v(s.vel))) for s in states]
        bp = _v(sim.arena.ball.get_state().pos)
        if trace:
            frames.append(dict(t=(tick + 1) / 120.0, ball=bp.round(0).tolist(),
                               cars=[dict(pos=_v(s.pos).round(0).tolist(), speed=round(float(np.linalg.norm(_v(s.vel)))),
                                          boost=round(s.boost, 1), ground=bool(s.is_on_ground),
                                          jumped=bool(s.has_jumped), flipped=bool(s.has_flipped))
                                     for s in states]))
        if touched and res["ball_half_team"] < 0 and (tick - touch_tick) / 120.0 >= POSSESSION_SECS:
            res["ball_half_team"] = 0 if bp[1] > 0 else 1
            d = [np.linalg.norm(_v(s.pos) - bp) for s in states]
            res["closer_team"] = sim.team(int(np.argmin(d)))
        if abs(bp[1]) > GOAL_Y and abs(bp[0]) < 893 and bp[2] < 643:   # wie IsBallScored
            res["goal_team"] = 0 if bp[1] > 0 else 1
            if touched and res["ball_half_team"] < 0:
                res["ball_half_team"] = res["closer_team"] = res["goal_team"]
            break
    res["boost_used"] = boost_used
    if trace:
        res["frames"] = frames
    return res


def evaluate(a: Spec, b: Spec, kickoffs: int, seed: int = 123, team_size: int = 1) -> dict:
    """A und B wechseln die Seiten; jede Anstoßposition gleich oft. Raten aus Sicht von A."""
    sim = KickoffSim(team_size)
    per_spawn = max(1, math.ceil(kickoffs / 10))
    seeds = spawn_seeds(per_spawn)
    order = [s for group in zip(*seeds.values()) for s in group]
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(kickoffs):
        a_blue = k % 2 == 0
        spawn_seed = order[(k // 2) % len(order)]
        specs = [a if (i % 2 == 0) == a_blue else b for i in range(2 * team_size)]
        r = play_kickoff(sim, spawn_seed, specs, rng)
        side = lambda team: -1 if team < 0 else (0 if (team == 0) == a_blue else 1)   # 0 = A
        rows.append(dict(spawn=r["spawn"], a_blue=a_blue, first=side(r["first_team"]),
                         time_to_touch=r["time_to_touch"], touch_speed=r["touch_speed"],
                         ball_half=side(r["ball_half_team"]), closer=side(r["closer_team"]),
                         goal=side(r["goal_team"]),
                         boost_a=r["boost_used"][0 if a_blue else 1], boost_b=r["boost_used"][1 if a_blue else 0]))
    return dict(a=str(a), b=str(b), kickoffs=kickoffs, seed=seed, summary=summarize(rows), rows=rows)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(rows: list[dict]) -> dict:
    def block(rs_: list[dict]) -> dict:
        touched = [r for r in rs_ if r["first"] >= 0]
        n = len(touched)
        out = dict(kickoffs=len(rs_), untouched=len(rs_) - n)
        for key in ("first", "ball_half", "closer"):
            k = sum(r[key] == 0 for r in touched)
            out[f"a_{key}_rate"] = k / n if n else math.nan
            out[f"a_{key}_ci"] = _wilson(k, n)
        out["goals_a_10s"] = sum(r["goal"] == 0 for r in rs_)
        out["goals_b_10s"] = sum(r["goal"] == 1 for r in rs_)
        for s, label in ((0, "a"), (1, "b")):
            mine = [r for r in touched if r["first"] == s]
            out[f"{label}_time_to_touch"] = float(np.mean([r["time_to_touch"] for r in mine])) if mine else math.nan
            out[f"{label}_touch_speed"] = float(np.mean([r["touch_speed"] for r in mine])) if mine else math.nan
        out["boost_a"] = float(np.mean([r["boost_a"] for r in rs_]))
        out["boost_b"] = float(np.mean([r["boost_b"] for r in rs_]))
        return out
    res = {"all": block(rows)}
    for name in SPAWN_NAMES.values():
        sub = [r for r in rows if r["spawn"] == name]
        if sub:
            res[name] = block(sub)
    return res


def trajectory(spec: Spec, opponent: Spec | None = None, seed: int = 1) -> dict:
    """Ein Anstoß je Position (Blau = spec, Orange = opponent, Standard dieselbe Angabe):
    Fahrweg, Sprung/Flip, Boost, Tempo von Blau."""
    sim = KickoffSim(1)
    rng = np.random.default_rng(seed)
    out = {}
    for name, (s,) in spawn_seeds(1).items():
        r = play_kickoff(sim, s, [spec, opponent or spec], rng, seconds=4.0, trace=True)
        frames = r["frames"]
        first = lambda key: next((f["t"] for f in frames if f["cars"][0][key]), None)
        tt = r["time_to_touch"]
        before = [f for f in frames if tt is None or f["t"] <= tt]
        out[name] = dict(time_to_touch=tt, first_team=r["first_team"], touch_speed=r["touch_speed"],
                         first_jump=first("jumped"),
                         first_flip=first("flipped"), boost_used=r["boost_used"][0],
                         max_speed=max(f["cars"][0]["speed"] for f in before) if before else None,
                         path=[f["cars"][0]["pos"] for f in frames[::12]],
                         speed=[f["cars"][0]["speed"] for f in frames[::12]])
    return out


def solo_time(sim: KickoffSim, variant: str, spawn_seed: int, params: dict | None = None
              ) -> tuple[float | None, float | None]:
    """Nur Auto 0 fährt (Skript), der Rest steht: Zeit und Tempo bis zur ersten Berührung."""
    sim.arena.reset_kickoff(spawn_seed)
    t0 = sim.arena.tick_count
    script = ScriptedKickoff(variant=variant, params=dict(params or {}))
    for c in sim.cars[1:]:
        c.set_controls(rs.CarControls())
    last = 0.0
    for _ in range(600):
        s = sim.cars[0].get_state()
        sim.cars[0].set_controls(_controls(script.step(sim.car_view(0), _v(sim.arena.ball.get_state().pos))))
        sim.arena.step(1)
        h = sim.cars[0].get_state().ball_hit_info
        if h.is_valid and h.tick_count_when_hit >= t0:
            return (h.tick_count_when_hit - t0) / 120.0, last
        last = float(np.linalg.norm(_v(s.vel)))
    return None, None


def tune(samples: int = 1500, seed: int = 0) -> dict:
    """Zufallssuche der Speedflip-Parameter je Anstoßart (schnellste erste Berührung)."""
    rng = np.random.default_rng(seed)
    sim = KickoffSim(1)
    seeds = spawn_seeds(1)
    kinds = {"diagonal": seeds["diagonal_links"][0], "offset": seeds["versetzt_links"][0], "center": seeds["mitte"][0]}
    best = {}
    for kind, s in kinds.items():
        base, _ = solo_time(sim, "boost", s)
        cand = (math.inf, None, None)
        for _ in range(samples):
            p = dict(steer_off=float(rng.uniform(0, 0.6)), jump_tick=int(rng.integers(0, 110)),
                     jump_ticks=int(rng.integers(1, 12)), gap=int(rng.integers(1, 8)),
                     cancel=int(rng.integers(4, 100)), final_dist=float(rng.choice([0, 300, 500, 700, 900])))
            t, v = solo_time(sim, "speedflip", s, p)
            if t is not None and t < cand[0]:
                cand = (t, v, p)
        best[kind] = dict(boost_only=base, speedflip=cand[0], touch_speed=cand[1], params=cand[2])
        print(f"{kind:9} nur Boost {base:.3f} s | Speedflip {cand[0]:.3f} s ({cand[1]:.0f} uu/s) {cand[2]}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--kickoffs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--team-size", type=int, default=1)
    ap.add_argument("--trajectory")
    ap.add_argument("--opponent", help="Gegenseite für --trajectory (Standard: dieselbe Angabe)")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-samples", type=int, default=1500)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.tune:
        result = tune(a.tune_samples)
    elif a.trajectory:
        result = trajectory(Spec.parse(a.trajectory), Spec.parse(a.opponent) if a.opponent else None)
        for name, t in result.items():
            print(f"{name:16} erste Berührung {t['time_to_touch'] or float('nan'):.2f} s "
                  f"({'Blau' if t['first_team'] == 0 else 'Orange' if t['first_team'] == 1 else '-'}), "
                  f"Tempo des Berührers {t['touch_speed'] or 0:5.0f} uu/s, Blau: "
                  f"Spitze {t['max_speed'] or 0:5.0f}, Sprung {t['first_jump']}, Flip {t['first_flip']}, "
                  f"Boost {t['boost_used']:.1f}")
    else:
        if not (a.a and a.b):
            ap.error("--a und --b nötig (oder --trajectory / --tune)")
        result = evaluate(Spec.parse(a.a), Spec.parse(a.b), a.kickoffs, a.seed, a.team_size)
        s = result["summary"]["all"]
        print(f"A zuerst {s['a_first_rate']:.3f} [{s['a_first_ci'][0]:.3f}; {s['a_first_ci'][1]:.3f}]  "
              f"Ballhälfte {s['a_ball_half_rate']:.3f}  näher {s['a_closer_rate']:.3f}  "
              f"Tore in 10 s {s['goals_a_10s']}:{s['goals_b_10s']}  (n={s['kickoffs']})")
    if a.out:
        a.out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
