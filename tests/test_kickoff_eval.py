"""Anstoß-Auswertung (eval/kickoff_eval.py), geskripteter Anstoß (deploy/scripted_kickoff.py) und
Aktionsauswahl (deploy/action_select.py). Echte RocketSim-Arena aus dem rlgym-Paket; die Tests mit
Policy nehmen den neuesten Checkpoint des Hauptlaufs (nur gelesen) und werden sonst übersprungen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("RocketSim")

from deploy.action_select import AIR_GROUPS, GROUND_GROUPS, argmax_group, select  # noqa: E402
from deploy.action_table import LOOKUP_TABLE  # noqa: E402
from env.obs_python import POS_COEF  # noqa: E402
from eval import kickoff_eval as ke  # noqa: E402

CKPTS = ROOT / "runs" / "lucy_1v1" / "checkpoints"


def _newest() -> Path | None:
    c = [p for p in CKPTS.iterdir() if p.name.isdigit() and (p / "PPO_POLICY.lt").exists()] if CKPTS.exists() else []
    return max(c, key=lambda p: int(p.name)) if c else None


POLICY = _newest()
needs_policy = pytest.mark.skipif(POLICY is None, reason="kein Checkpoint in runs/lucy_1v1")


def _index(values) -> int:
    return int(np.flatnonzero((LOOKUP_TABLE == np.array(values, dtype=np.float32)).all(1))[0])


# --- Aktionsauswahl (gleiche Regeln wie env/cpp/ActionSelect.h, tests/cpp/test_action_select.cpp) ---

def test_ground_groups_merge_boost_entries_but_not_plain_throttle():
    boost, plain = _index([1, 0, 0, 0, 0, 0, 1, 0]), _index([1, 0, 0, 0, 0, 0, 0, 0])
    assert (GROUND_GROUPS == GROUND_GROUPS[boost]).sum() == 9
    assert (GROUND_GROUPS == GROUND_GROUPS[plain]).sum() == 1
    jumps = np.flatnonzero(LOOKUP_TABLE[:, 5] >= 0.5)
    assert all((GROUND_GROUPS == GROUND_GROUPS[j]).sum() == 1 for j in jumps)


def test_air_groups_ignore_throttle_and_handbrake():
    coast = _index([0, 0, 0, 0, 0, 0, 0, 0])
    assert (AIR_GROUPS == AIR_GROUPS[coast]).sum() == 6


def test_argmax_group_picks_the_intended_effect():
    boost, plain = _index([1, 0, 0, 0, 0, 0, 1, 0]), _index([1, 0, 0, 0, 0, 0, 0, 0])
    probs = np.full(90, 0.0)
    members = np.flatnonzero(GROUND_GROUPS == GROUND_GROUPS[boost])
    probs[members] = 0.08
    probs[plain] = 0.10
    rest = np.setdiff1d(np.arange(90), np.append(members, plain))
    probs[rest] = (1 - probs.sum()) / len(rest)
    assert select(probs, "argmax", True) == plain
    assert GROUND_GROUPS[select(probs, "argmax_group", True)] == GROUND_GROUPS[boost]
    assert argmax_group(probs, AIR_GROUPS) in range(90)
    with pytest.raises(ValueError):
        select(probs, "greedy", True)


# --- Geskripteter Anstoß ----------------------------------------------------------------------

def test_speedflip_reaches_the_ball_earlier_than_boost_only_from_every_spawn_and_side():
    result = ke.evaluate(ke.Spec.parse("script:speedflip"), ke.Spec("idle"), 20)
    boost = ke.evaluate(ke.Spec.parse("script:boost"), ke.Spec("idle"), 20)
    for name in ke.SPAWN_NAMES.values():
        s, b = result["summary"][name], boost["summary"][name]
        assert s["a_first_rate"] == 1.0
        assert s["a_time_to_touch"] < b["a_time_to_touch"], name
        assert s["a_touch_speed"] > 2000, name
    # beide Seiten gleich (Blau und Orange wechseln sich ab)
    rows = result["rows"]
    for name in ke.SPAWN_NAMES.values():
        t = {r["a_blue"]: r["time_to_touch"] for r in rows if r["spawn"] == name}
        assert t[True] == pytest.approx(t[False], abs=1 / 120 + 1e-9), name


def test_policy_decides_every_8_ticks_on_the_snapshot_one_tick_into_the_step(monkeypatch):
    """Schrittfolge wie RLGymSim Gym::Step: Entscheidung k (k >= 1) sieht den Zustand nach Tick 8(k-1)+1."""
    seen = []

    class Recorder:
        def action_probs(self, obs):
            seen.append(obs[43:46] / POS_COEF)               # eigene Position im Obs (Spieler-Block)
            p = np.zeros(90)
            p[_index([1, 0, 0, 0, 0, 0, 1, 0])] = 1.0
            return p

    monkeypatch.setattr(ke, "_load", lambda path: Recorder())
    sim = ke.KickoffSim(1)
    seed = ke.spawn_seeds(1)["mitte"][0]
    r = ke.play_kickoff(sim, seed, [ke.Spec("policy", path="x", mode="argmax"), ke.Spec("idle")],
                        np.random.default_rng(0), seconds=1.0, trace=True)
    frames = r["frames"]                                     # frames[t] = Zustand nach Tick t+1
    assert len(seen) == 15
    for k in range(1, len(seen)):
        expected = np.array(frames[8 * (k - 1)]["cars"][0]["pos"])
        assert np.allclose(seen[k], expected, atol=1.0), (k, seen[k], expected)


# --- Mit echter Policy ------------------------------------------------------------------------

@needs_policy
def test_speedflip_script_wins_the_first_touch_against_the_learned_kickoff():
    """Kern des Spieltest-Befunds: Der gelernte Anstoß kommt deutlich später an als ein Speedflip."""
    spec = f"policy:{POLICY}"
    r = ke.evaluate(ke.Spec.parse(f"script:speedflip+{spec}"), ke.Spec.parse(f"{spec}@argmax"), 10)
    s = r["summary"]["all"]
    assert s["kickoffs"] == 10 and s["untouched"] == 0
    assert s["a_first_rate"] >= 0.9
    assert s["a_time_to_touch"] < 2.6


@needs_policy
def test_trajectory_reports_path_flip_and_boost_per_spawn():
    t = ke.trajectory(ke.Spec.parse(f"policy:{POLICY}@argmax"), ke.Spec("idle"))
    assert set(t) == set(ke.SPAWN_NAMES.values())
    for name, row in t.items():
        assert row["time_to_touch"] is None or 1.5 < row["time_to_touch"] < 4.0, name
        assert len(row["path"]) == len(row["speed"]) > 10
        assert 0 <= row["boost_used"] <= 100
