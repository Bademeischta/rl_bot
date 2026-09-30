"""Hauptlauf-Betrieb (B4): Stand mit Grenzwerten, sauberer Stopp, Start über die Aufgabenplanung.

main_run_status.py liest eine metrics.csv aus dem echten C++-Writer (write_metrics_csv.exe); die
Skripte laufen unter Windows PowerShell 5.1 wie beim Nutzer (Stopp gegen einen echten Prozess, der auf
die Stop-Datei reagiert; Start als DryRun, weil ein echter Start den Hauptlauf starten würde).
"""
from __future__ import annotations

import csv
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
