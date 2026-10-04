"""Hauptlauf-Betrieb (B4): Stand mit Grenzwerten, sauberer Stopp, Start über die Aufgabenplanung.

main_run_status.py liest eine metrics.csv aus dem echten C++-Writer (write_metrics_csv.exe); die
Skripte laufen unter Windows PowerShell 5.1 wie beim Nutzer (Stopp gegen einen echten Prozess, der auf
die Stop-Datei reagiert; Start als DryRun, weil ein echter Start den Hauptlauf starten würde).
"""
from __future__ import annotations

import csv
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
WRITER = BUILD / ("write_metrics_csv.exe" if os.name == "nt" else "write_metrics_csv")
STATUS = ROOT / "tools" / "local" / "main_run_status.py"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")
needs_writer = pytest.mark.skipif(not WRITER.exists(), reason=f"{WRITER} fehlt")
needs_ps51 = pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell 5.1 nicht vorhanden")

HEALTHY = {"SB3 Clip Fraction": "0.042", "ep_end_goal": "0.64", "ep_end_drill": "0.36", "ep_end_time": "0",
           "ep_end_notouch": "0.0002", "Steps Collected During Learn": "63000"}


def _run_dir(tmp_path: Path, iterations: int, overrides: dict[tuple[int, str], str] | None = None) -> Path:
    """runs/<x>/metrics.csv über den echten Writer, jede Iteration mit den Hauptlauf-Kennzahlen."""
    run = tmp_path / "run"
    (run / "checkpoints" / "6037692544").mkdir(parents=True)
    args = []
    for i in range(iterations):
        for key, value in HEALTHY.items():
            args += ["--set", str(i), key, (overrides or {}).get((i, key), value)]
        for (j, key), value in (overrides or {}).items():
            if j == i and key not in HEALTHY:
                args += ["--set", str(i), key, value]
    r = subprocess.run([str(WRITER), str(run / "metrics.csv"), str(iterations), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return run


def _status(run: Path, *extra: str, console: str = "cp1252") -> subprocess.CompletedProcess:
    """Wie in einer Windows-Konsole: stdout in cp1252 (so lief der erste echte Aufruf auf den Fehler)."""
    return subprocess.run([sys.executable, str(STATUS), "--run", str(run), "--window", "10", *extra],
                          capture_output=True, text=True, encoding=console, errors="replace",
                          env={**os.environ, "PYTHONIOENCODING": console})


@needs_writer
def test_status_passes_a_healthy_run_and_treats_low_sps_only_as_a_hint(tmp_path):
    run = _run_dir(tmp_path, 25)
    r = _status(run, "--min-sps", "150000")          # der Writer schreibt 68.000 SPS
    assert r.returncode == 0, r.stdout
    assert "alle Grenzwerte eingehalten" in r.stdout
    assert "[HINWEIS] SPS unter" in r.stdout          # Spiel nebenher: kein Grenzwert
    # Drill-Episoden enden planmäßig: 0,64 Tore bei 0,36 Drill = 100 % der übrigen Episoden
    assert "Tor-Anteil ohne Drill 1.000" in r.stdout


@needs_writer
@pytest.mark.parametrize("key,value,rule", [
    ("Policy Entropy", "2.3", "Entropie > 2,5"),
    ("Mean KL Divergence", "0.02", "KL < 0,01"),
    ("SB3 Clip Fraction", "0.15", "Clip-Fraction < 0,10"),
    ("Value Function Loss", "nan", "keine nan/inf (Value Function Loss)"),
    ("Value Function Loss", "5.0", "Value Loss stabil (<= 3x"),
    ("ep_end_time", "0.05", "Timeouts <= 0,01"),
    ("ep_end_goal", "0.5", "Tor-Anteil ohne Drill >= 0,95"),
])
def test_status_flags_each_violated_limit(tmp_path, key, value, rule):
    # Verletzung nur im letzten Fenster (Iterationen 15-24), davor normal
    run = _run_dir(tmp_path, 25, {(i, key): value for i in range(15, 25)})
    r = _status(run)
    assert r.returncode == 3, r.stdout
    assert f"[VERLETZT] {rule}" in r.stdout
    assert "GRENZWERT VERLETZT" in r.stdout


@needs_writer
def test_status_appends_one_row_per_check_to_the_log(tmp_path):
    run = _run_dir(tmp_path, 25)
    log = tmp_path / "status.csv"
    assert _status(run, "--log", str(log)).returncode == 0
    assert _status(run, "--log", str(log)).returncode == 0
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    assert len(rows) == 2 and rows[0]["steps"] == str(3907335040 + 25 * 100000) and rows[0]["verletzt"] == ""


def _ps(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", *args], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


@needs_ps51
def test_stop_script_creates_the_stop_file_and_waits_for_the_trainer(tmp_path):
    """Echter Prozess, der wie train_bot.exe auf --stop-file reagiert (endet nach der Datei)."""
    stop = tmp_path / "STOP"
    code = ("import sys, time, pathlib\n"
            "p = pathlib.Path(sys.argv[1])\n"
            "while not p.exists(): time.sleep(0.2)\n"
            "time.sleep(1.0)\n")
    proc = subprocess.Popen([sys.executable, "-c", code, str(stop)])
    try:
        r = _ps("-File", str(ROOT / "tools" / "local" / "stop_main_run.ps1"), "-StopFile", str(stop),
                "-ProcessId", str(proc.pid), "-TimeoutSeconds", "60", "-CheckpointFolder", str(tmp_path / "none"))
        assert r.returncode == 0, r.stdout + r.stderr
        assert stop.exists() and "Stop-Datei angelegt" in r.stdout and "Sauber beendet" in r.stdout
        assert proc.poll() is not None                    # hat wirklich auf das Ende gewartet
    finally:
        if proc.poll() is None:
            proc.kill()


@needs_ps51
def test_start_script_refuses_a_second_trainer_and_plans_stop_file_and_log(tmp_path):
    start = str(ROOT / "tools" / "local" / "start_main_run.ps1")
    # läuft schon (hier: der Python-Prozess dieses Tests) -> Abbruch, nichts registriert
    r = _ps("-File", start, "-TrainerName", "python", "-DryRun")
    assert r.returncode == 3 and "uft schon (PID" in r.stdout, r.stdout + r.stderr   # Umlaut: OEM-Codepage
    # Plan: eigenes Fenster über die Aufgabenplanung, Stop-Datei und End-Checkpoint, alte Stop-Datei weg
    stale = tmp_path / "STOP"
    stale.write_text("alt", encoding="ascii")
    r = _ps("-File", start, "-TrainerName", "gibt_es_nicht_xyz", "-StopFile", str(stale), "-DryRun")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "run_main.ps1" in r.stdout and "--stop-file" in r.stdout and "--save-on-exit" in r.stdout
    assert "Alte Stop-Datei" in r.stdout and stale.exists()   # DryRun löscht nichts
    assert "lucy_1v1_zero_sum_drill_fast.json" in r.stdout


def _task_state(name: str) -> str:
    r = _ps("-Command", f"$t = Get-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue; if ($t) {{ $t.State }} else {{ 'fehlt' }}")
    return r.stdout.strip()


@needs_ps51
@pytest.mark.skipif(not (BUILD / "train_bot.exe").exists(), reason="start_main_run.ps1 verlangt build\\cpp_cu128\\train_bot.exe")
def test_start_script_restarts_after_a_clean_stop_left_the_task_window_open(tmp_path):
    """B7: Nach dem sauberen Stopp bleibt das -NoExit-Fenster offen, die Aufgabe gilt als laufend und der
    nächste Start wurde ignoriert ("train_bot ist nach 90 s nicht gestartet", 01.10.2026).

    Echte Aufgabenplanung mit eigenem Aufgabennamen; als "Trainer" läuft eine Kopie von ping.exe unter
    eigenem Namen, gestartet von einem Ersatz für run_main.ps1.
    """
    ping = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "ping.exe"
    if not ping.exists():
        pytest.skip("ping.exe fehlt")
    tag = f"{os.getpid()}_{int(time.time())}"
    trainer = f"rlbot_fake_trainer_{tag}"
    task = f"RLbot Test {tag}"
    exe = tmp_path / f"{trainer}.exe"
    shutil.copyfile(ping, exe)
    runner = tmp_path / "fake_runner.ps1"
    runner.write_text("param([string]$Config, [string]$StopFile)\n"
                      f"& '{exe}' -n 600 127.0.0.1 | Out-Null\n", encoding="ascii")
    start = str(ROOT / "tools" / "local" / "start_main_run.ps1")
    args = ["-File", start, "-TaskName", task, "-TrainerName", trainer, "-Runner", str(runner),
            "-StopFile", str(tmp_path / "STOP")]
    kill = ["taskkill", "/F", "/IM", f"{trainer}.exe"]
    try:
        r = _ps(*args)
        assert r.returncode == 0 and "PID" in r.stdout, r.stdout + r.stderr
        # "sauberer Stopp": der Trainer endet, das Fenster der Aufgabe bleibt offen
        subprocess.run(kill, capture_output=True)
        time.sleep(2)
        assert _task_state(task) == "Running"
        r = _ps(*args)
        assert r.returncode == 0 and "PID" in r.stdout, r.stdout + r.stderr
        assert "alte Instanz wird beendet" in r.stdout
    finally:
        subprocess.run(kill, capture_output=True)
        _ps("-Command", f"Stop-ScheduledTask -TaskName '{task}' -ErrorAction SilentlyContinue; "
                        f"Unregister-ScheduledTask -TaskName '{task}' -Confirm:$false -ErrorAction SilentlyContinue")
    assert _task_state(task) == "fehlt"


@needs_ps51
def test_stop_script_names_the_newest_checkpoint_of_the_folder_from_the_trainer_config(tmp_path):
    """B8: Der Hauptlauf lief mit runs/lucy_1v1_lr1e4, das Skript nannte trotzdem den neuesten Checkpoint aus
    runs/lucy_1v1 (03./04.10.2026). Der Ordner kommt jetzt aus der Config auf der Kommandozeile des Trainers."""
    stop = tmp_path / "STOP"
    folder = tmp_path / "anderer lauf" / "checkpoints"
    for steps in ("100", "12345"):
        (folder / steps).mkdir(parents=True)
    config = tmp_path / "lauf_config.json"
    config.write_text(json.dumps({"learner": {"checkpoint_folder": str(folder)}}), encoding="utf-8")
    code = ("import sys, time, pathlib\n"
            "p = pathlib.Path(sys.argv[1])\n"
            "while not p.exists(): time.sleep(0.2)\n"
            "time.sleep(1.0)\n")
    # wie train_bot.exe: <config> --stop-file <datei>
    proc = subprocess.Popen([sys.executable, "-c", code, str(stop), str(config), "--stop-file", str(stop)])
    try:
        r = _ps("-File", str(ROOT / "tools" / "local" / "stop_main_run.ps1"), "-StopFile", str(stop),
                "-ProcessId", str(proc.pid), "-TimeoutSeconds", "60")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "Neuester Checkpoint: 12345" in r.stdout, r.stdout
    finally:
        if proc.poll() is None:
            proc.kill()
