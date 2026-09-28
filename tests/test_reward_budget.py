"""Reward-Bilanz (tools/cpp/reward_budget.cpp) über den echten Reward-Code.

Die Komponenten laufen als eigene Instanzen neben dem Reward des Matches; ihre Summe (in
Zero-Sum-Form) muss den Match-Reward treffen, sonst misst die Bilanz etwas anderes als das Training.
Echter Checkpoint des Hauptlaufs (nur gelesen), übersprungen, wenn Binary oder Checkpoint fehlen.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("RLBOT_BUILD_DIR", str(ROOT / "build" / "cpp_cu128")))
EXE = BUILD / ("reward_budget.exe" if os.name == "nt" else "reward_budget")
CKPTS = ROOT / "runs" / "lucy_1v1" / "checkpoints"
CONFIG = ROOT / "train" / "configs" / "experiments" / "zero_sum.json"


def _newest() -> Path | None:
    c = [p for p in CKPTS.iterdir() if p.name.isdigit() and (p / "PPO_POLICY.lt").exists()] if CKPTS.exists() else []
    return (max(c, key=lambda p: int(p.name)) / "PPO_POLICY.lt") if c else None


POLICY = _newest()


@pytest.mark.skipif(not EXE.exists() or POLICY is None, reason="reward_budget.exe oder Checkpoint fehlt")
def test_components_add_up_to_the_match_reward_and_show_the_corner_stream(tmp_path):
    out = tmp_path / "budget.json"
    r = subprocess.run([str(EXE), "--config", str(CONFIG), "--policy", str(POLICY), "--games", "4",
                        "--seconds", "60", "--threads", "2", "--meshes", str(ROOT / "collision_meshes"),
                        "--out", str(out)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(out.read_text(encoding="utf-8"))
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["rewards"]

    assert d["max_abs_diff"] < 1e-4                        # Zerlegung = Reward des Trainings
    assert d["goal_value"] == pytest.approx(cfg["goal"] + cfg["concede"])   # Zero-Sum: Tor zählt doppelt
    expected = ["event"] + [k for k in ("touch_ball_to_goal_accel", "offensive_potential_krc",
                                        "dist_weighted_align_krc", "velocity_player_to_ball", "save_boost", "in_air")
                            if cfg.get(k, 0) != 0]
    assert d["components"] == expected

    static = d["static"]
    assert static["neutral_beide_zum_ball"]["zero_sum"] == pytest.approx(0, abs=1e-6)   # symmetrisch
    corner = static["ecke_verteidiger_im_tor"]
    assert corner["zero_sum_per_s"] > 5
    assert corner["goal_equivalent_s"] == pytest.approx(d["goal_value"] / corner["zero_sum_per_s"])
    assert corner["hold_value_3s"] > corner["hold_value_1s"] > 0
    parts = sum(c["zero_sum"] for c in corner["components"].values())
    assert parts == pytest.approx(corner["zero_sum"], abs=1e-6)

    played = d["played"]
    for lage in ("angriffsdrittel_ecke", "angriffsdrittel_mitte", "mittelfeld_blau_sicht"):
        assert played[lage]["steps"] > 0
    shares = sum(played[k]["time_share"] for k in ("angriffsdrittel_ecke", "angriffsdrittel_mitte",
                                                   "mittelfeld_blau_sicht"))
    assert shares == pytest.approx(1.0)
