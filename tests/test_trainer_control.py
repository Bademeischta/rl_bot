"""Sauberes Beenden des Trainers (Review-Befund R15).

train_bot.exe --stop-file: Existiert die Datei, endet das Training nach der laufenden Iteration und
save_on_exit schreibt den End-Checkpoint. run_experiment.ps1 benutzt das über
Stop-TrainerGracefully (tools/experiments/TrainerControl.ps1) und beendet den Prozess nur als
Notfall nach einem Timeout hart.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("RLBOT_BUILD_DIR", str(ROOT / "build" / "cpp_cu128")))
TRAINER = BUILD / ("train_bot.exe" if os.name == "nt" else "train_bot")
MESHES = ROOT / "collision_meshes"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


def _tiny_config(tmp_path: Path) -> Path:
    cfg = {
        "env": {"no_touch_timeout_secs": 10.0, "game_timeout_secs": 60.0, "mode_mix": [1.0, 0.0, 0.0]},
        "state_setters": {"kickoff": 1.0, "random": 1.0},
        "learner": {"num_threads": 1, "num_games_per_thread": 2, "device": "cpu", "timestep_limit": 0,
                    "timesteps_per_iteration": 1000, "timesteps_per_save": 10_000_000, "checkpoints_to_keep": 3,
                    "checkpoint_folder": (tmp_path / "run" / "checkpoints").as_posix(),
                    "policy_layer_sizes": [16], "critic_layer_sizes": [16], "ppo_epochs": 1,
                    "exp_buffer_iterations": 1, "ppo_batch_size": 1000, "ppo_mini_batch_size": 500,
                    "save_on_exit": True},
        "metrics": {"send_metrics": False, "skill_tracker": False, "run": "pytest_stop"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _trainer_env() -> dict:
    env = dict(os.environ)
    env["PYTHONHOME"] = sys.base_prefix           # Learner startet einen eingebetteten Interpreter
    env["PATH"] = sys.base_prefix + os.pathsep + env.get("PATH", "")
    return env


@pytest.mark.skipif(not TRAINER.exists() or not MESHES.exists(), reason="train_bot.exe oder collision_meshes fehlen")
def test_train_bot_stops_cleanly_after_the_iteration_and_saves(tmp_path):
    cfg = _tiny_config(tmp_path)
    stop = tmp_path / "STOP"
    log = (tmp_path / "train.log").open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([str(TRAINER), str(cfg), "--collision-meshes", str(MESHES), "--stop-file", str(stop)],
                            cwd=ROOT, env=_trainer_env(), stdout=log, stderr=subprocess.STDOUT)
    try:
        metrics = tmp_path / "run" / "metrics.csv"
        deadline = time.time() + 180
        while time.time() < deadline and proc.poll() is None:
            if metrics.exists() and len(metrics.read_text(encoding="utf-8").splitlines()) >= 3:
                break
            time.sleep(0.5)
        assert proc.poll() is None, "Trainer endete vor der Stop-Datei"
        stop.write_text("stop", encoding="ascii")
        assert proc.wait(timeout=180) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        log.close()
    out = (tmp_path / "train.log").read_text(encoding="utf-8", errors="replace")
    assert "Stop-Datei" in out
    ckpts = sorted((p for p in (tmp_path / "run" / "checkpoints").iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    assert ckpts, "kein End-Checkpoint (save_on_exit)"
    end = ckpts[-1]
    for name in ("PPO_POLICY.lt", "PPO_CRITIC.lt", "PPO_POLICY_OPTIM.lt", "PPO_CRITIC_OPTIM.lt", "RUNNING_STATS.json"):
        assert (end / name).stat().st_size > 0, name
    assert json.loads((end / "RUNNING_STATS.json").read_text(encoding="utf-8"))["cumulative_timesteps"] == int(end.name)


@pytest.mark.skipif(not TRAINER.exists(), reason="train_bot.exe fehlt")
def test_train_bot_refuses_a_stale_stop_file(tmp_path):
    cfg = _tiny_config(tmp_path)
    stop = tmp_path / "STOP"
    stop.write_text("alt", encoding="ascii")
    r = subprocess.run([str(TRAINER), str(cfg), "--stop-file", str(stop)], cwd=ROOT, env=_trainer_env(),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "Stop-Datei existiert schon" in r.stderr


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell 5.1 nicht vorhanden")
def test_stop_trainer_gracefully_waits_and_forces_only_after_the_timeout(tmp_path):
    """Echte Funktion aus TrainerControl.ps1 mit zwei Prozessen: einer reagiert auf die Stop-Datei,
    einer nicht (der wird erst nach dem Timeout hart beendet)."""
    polite = ("import os, sys, time\n"
              "stop = sys.argv[1]\n"
              "while not os.path.exists(stop): time.sleep(0.1)\n"
              "sys.exit(0)\n")
    stubborn = "import time\ntime.sleep(120)\n"
    (tmp_path / "polite.py").write_text(polite, encoding="ascii")
    (tmp_path / "stubborn.py").write_text(stubborn, encoding="ascii")
    script = tmp_path / "probe.ps1"
    script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f". '{ROOT / 'tools' / 'experiments' / 'TrainerControl.ps1'}'\n"
        f"$py = '{sys.executable}'\n"
        f"$p1 = Start-Process -FilePath $py -ArgumentList '\"{tmp_path / 'polite.py'}\" \"{tmp_path / 'STOP1'}\"' -PassThru -NoNewWindow\n"
        "$null = $p1.Handle\n"
        f"$r1 = Stop-TrainerGracefully -Process $p1 -StopFile '{tmp_path / 'STOP1'}' -TimeoutSeconds 60\n"
        "Write-Output \"R1=$r1 EXIT1=$($p1.ExitCode)\"\n"
        f"$p2 = Start-Process -FilePath $py -ArgumentList '\"{tmp_path / 'stubborn.py'}\"' -PassThru -NoNewWindow\n"
        "$t0 = Get-Date\n"
        f"$r2 = Stop-TrainerGracefully -Process $p2 -StopFile '{tmp_path / 'STOP2'}' -TimeoutSeconds 3\n"
        "Write-Output \"R2=$r2 ALIVE2=$(-not $p2.HasExited) SECS=$([int]((Get-Date) - $t0).TotalSeconds)\"\n",
        encoding="ascii")
    r = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "R1=sauber EXIT1=0" in r.stdout
    line = next(l for l in r.stdout.splitlines() if l.startswith("R2="))
    assert "R2=erzwungen" in line and "ALIVE2=False" in line
    assert int(line.rsplit("SECS=", 1)[1]) >= 3            # erst nach dem Timeout hart beendet
