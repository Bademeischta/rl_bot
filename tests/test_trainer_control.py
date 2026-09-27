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


# --- R16: nur ein nachweislich vollständiger End-Checkpoint --------------------------------

REAL_CKPTS = ROOT / "runs" / "lucy_1v1" / "checkpoints"


def _newest_real() -> Path | None:
    if not REAL_CKPTS.exists():
        return None
    c = [p for p in REAL_CKPTS.iterdir() if p.name.isdigit() and (p / "PPO_POLICY.lt").exists()]
    return max(c, key=lambda p: int(p.name)) if c else None


def _copy_as(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    stats = json.loads((dst / "RUNNING_STATS.json").read_text(encoding="utf-8"))
    stats["cumulative_timesteps"] = int(dst.name)
    (dst / "RUNNING_STATS.json").write_text(json.dumps(stats), encoding="utf-8")
    return dst


@pytest.mark.skipif(_newest_real() is None, reason="kein echter Checkpoint in runs/lucy_1v1")
def test_pick_checkpoint_skips_incomplete_newer_checkpoints(tmp_path):
    """Echte Checkpoint-Dateien (Kopie, Original nur gelesen); neuere Ordner simulieren einen
    Abbruch mitten im Save. Gewählt wird der neueste vollständige, nicht der höchste Ordner."""
    sys.path.insert(0, str(ROOT / "tools" / "experiments"))
    from pick_checkpoint import pick_latest_complete

    src = _newest_real()
    base = int(src.name)
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    shutil.copytree(src, ck / src.name)
    good_newer = _copy_as(src, ck / str(base + 100))
    truncated = _copy_as(src, ck / str(base + 300))                  # Optimizer halb geschrieben
    data = (truncated / "PPO_POLICY_OPTIM.lt").read_bytes()
    (truncated / "PPO_POLICY_OPTIM.lt").write_bytes(data[: len(data) // 2])
    no_stats = _copy_as(src, ck / str(base + 200))                   # RUNNING_STATS fehlt
    (no_stats / "RUNNING_STATS.json").unlink()
    wrong_steps = _copy_as(src, ck / str(base + 400))                # Ordnername passt nicht
    (wrong_steps / "RUNNING_STATS.json").write_text(json.dumps({"cumulative_timesteps": 1}), encoding="utf-8")
    empty_policy = _copy_as(src, ck / str(base + 500))
    (empty_policy / "PPO_POLICY.lt").write_bytes(b"")

    chosen, rejected = pick_latest_complete(ck)
    assert chosen == good_newer
    reasons = "\n".join(rejected)
    for name in (empty_policy.name, wrong_steps.name, truncated.name, no_stats.name):
        assert name in reasons
    assert "kein intaktes Archiv" in reasons and "fehlt" in reasons and "passt nicht" in reasons

    r = subprocess.run([sys.executable, str(ROOT / "tools" / "experiments" / "pick_checkpoint.py"), str(ck)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0 and r.stdout.strip() == str(good_newer)
    assert "verworfen" in r.stderr


def test_pick_checkpoint_fails_without_any_complete_checkpoint(tmp_path):
    ck = tmp_path / "checkpoints"
    (ck / "100").mkdir(parents=True)
    (ck / "100" / "PPO_POLICY.lt").write_bytes(b"PK")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "experiments" / "pick_checkpoint.py"), str(ck)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 1
    assert "kein vollständiger Checkpoint" in r.stderr
