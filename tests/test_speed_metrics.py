"""Zeitaufschlüsselung in metrics.csv (Geschwindigkeit G1, AUDIT.md §9).

Echter Trainer, winzige CPU-Config, drei Iterationen: Die neuen Zeitspalten müssen da sein und
zueinander passen. "Policy Infer Time" (Upstream) enthielt schon immer das Anhängen der
Trajektorien; G1 weist beide Teile getrennt aus.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from show_metrics import read_rows  # noqa: E402

BUILD = Path(os.environ.get("RLBOT_BUILD_DIR", str(ROOT / "build" / "cpp_cu128")))
TRAINER = BUILD / ("train_bot.exe" if os.name == "nt" else "train_bot")
MESHES = ROOT / "collision_meshes"

G1_COLUMNS = [
    "Infer Call Time", "Traj Append Time", "Obs Tensor Time", "Obs To Device Time", "Agent Wait Time",
    "Collect Concat Time", "Add Experience Time", "Exp Value Pred Time", "Exp GAE Time", "Exp Submit Time",
    "PPO Shuffle Time", "PPO Minibatch Time", "PPO Optim Time", "PPO Param Copy Time", "Empty Cache Time",
    "Prev Tail Time", "Prev Skill Eval Time", "Prev Iteration Callback Time", "Prev Save Time",
    "Step Callback Time", "Play Stats Time",
]


@pytest.mark.skipif(not TRAINER.exists() or not MESHES.exists(), reason="train_bot.exe oder collision_meshes fehlen")
def test_metrics_csv_breaks_the_iteration_time_down(tmp_path):
    cfg = {
        "env": {"no_touch_timeout_secs": 10.0, "game_timeout_secs": 60.0, "mode_mix": [1.0, 1.0, 0.0]},
        "state_setters": {"kickoff": 1.0, "random": 1.0},
        "learner": {"num_threads": 2, "num_games_per_thread": 2, "device": "cpu", "timestep_limit": 3000,
                    "timesteps_per_iteration": 1000, "timesteps_per_save": 10_000_000,
                    "checkpoint_folder": (tmp_path / "run" / "checkpoints").as_posix(),
                    "policy_layer_sizes": [16], "critic_layer_sizes": [16], "ppo_epochs": 1,
                    "exp_buffer_iterations": 1, "ppo_batch_size": 1000, "ppo_mini_batch_size": 500},
        "metrics": {"send_metrics": False, "skill_tracker": False, "run": "pytest_speed"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    env = dict(os.environ, PYTHONHOME=sys.base_prefix)
    env["PATH"] = sys.base_prefix + os.pathsep + env.get("PATH", "")
    r = subprocess.run([str(TRAINER), str(path), "--collision-meshes", str(MESHES)], cwd=ROOT, env=env,
                       capture_output=True, text=True, errors="replace", timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]

    rows = read_rows(tmp_path / "run" / "metrics.csv")
    assert len(rows) >= 3
    missing = [c for c in G1_COLUMNS if c not in rows[0]]
    assert missing == [], f"fehlende Spalten: {missing}"
    for row in rows:
        v = {k: float(x) for k, x in row.items() if x not in ("", None) and k in G1_COLUMNS + [
            "Policy Infer Time", "Env Step Time", "PPO Learn Time", "Collection Time"]}
        assert all(x >= 0 for x in v.values()), v
        # Upstream-Spalte = Summe der beiden neuen Teile
        assert abs(v["Infer Call Time"] + v["Traj Append Time"] - v["Policy Infer Time"]) < 1e-6
        # Teile liegen innerhalb ihres Ganzen
        assert v["Play Stats Time"] <= v["Step Callback Time"] + 1e-9
        assert v["Step Callback Time"] <= v["Env Step Time"] + 1e-3
        ppo_parts = v["PPO Shuffle Time"] + v["PPO Minibatch Time"] + v["PPO Optim Time"] + v["PPO Param Copy Time"]
        assert ppo_parts <= v["PPO Learn Time"] + 1e-6
        exp_parts = v["Exp Value Pred Time"] + v["Exp GAE Time"] + v["Exp Submit Time"]
        assert exp_parts <= v["Add Experience Time"] + 1e-6
    # Der Step-Callback läuft wirklich (Metriken und Spielanalyse je Schritt)
    assert sum(float(row["Step Callback Time"]) for row in rows) > 0
