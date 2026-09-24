"""Rechnet aus metrics.csv den echten Durchsatz eines Laufs aus und zeigt, wo die Zeit hingeht.

    python tools/throughput.py runs/lucy_1v1 --started "2026-09-23 14:31:25"

Ohne --started wird die Laufzeit aus den Zeitstempeln der Dateien geschätzt.
Der Unterschied zwischen "Collected SPS" (nur Sammeln) und dem effektiven Durchsatz
(Steps geteilt durch echte Wall-Clock) ist die Zahl, die für Zeitpläne zählt.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

TIMERS = [
    ("Collection Time", "Sammeln"),
    ("Policy Infer Time", "  davon Policy-Inferenz"),
    ("Env Step Time", "  davon Simulation"),
    ("Consumption Time", "Lernen"),
    ("PPO Learn Time", "  davon PPO"),
    ("Total Iteration Time", "Iteration gesamt (laut Report)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="Run-Ordner oder metrics.csv")
    ap.add_argument("--started", help="Startzeit 'YYYY-MM-DD HH:MM:SS' (sonst geschätzt)")
    ap.add_argument("--window", type=int, default=500, help="Iterationen für die Mittelwerte")
    a = ap.parse_args()

    path = a.run if a.run.suffix == ".csv" else a.run / "metrics.csv"
    # Wiederholte Kopfzeilen überspringen: Jeder fortgesetzte Lauf hängt eine eigene an.
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if r.get("Cumulative Timesteps") != "Cumulative Timesteps"]
    if not rows:
        raise SystemExit("keine Daten")
    tail = rows[-a.window:]

    def mean(key: str) -> float:
        vals = [float(r[key]) for r in tail if r.get(key)]
        return sum(vals) / len(vals) if vals else float("nan")

    steps = float(rows[-1]["Cumulative Timesteps"])
    if a.started:
        start = dt.datetime.strptime(a.started, "%Y-%m-%d %H:%M:%S")
    else:
        cfg = path.parent / "config_used.json"
        start = dt.datetime.fromtimestamp((cfg if cfg.exists() else path).stat().st_mtime)
    elapsed = (dt.datetime.fromtimestamp(path.stat().st_mtime) - start).total_seconds()
    effective = steps / elapsed if elapsed > 0 else float("nan")

    print(f"Steps gesamt          : {steps:>12,.0f}")
    print(f"Laufzeit              : {elapsed / 3600:>12.2f} h")
    print(f"Iterationen           : {len(rows):>12,}")
    print()
    print(f"Collected SPS         : {mean('Collected Steps/Second'):>12,.0f}   (nur Sammelphase)")
    print(f"Overall SPS im Report : {mean('Overall Steps/Second'):>12,.0f}   (Sammeln + Lernen)")
    print(f"Effektiv Wall-Clock   : {effective:>12,.0f}   <- diese Zahl zaehlt fuer Zeitplaene")
    print()
    for key, label in TIMERS:
        print(f"{label:<32}{mean(key):>8.2f} s")
    per_iter = elapsed / len(rows)
    gap = per_iter - mean("Total Iteration Time")
    print(f"{'Gemessen pro Iteration':<32}{per_iter:>8.2f} s")
    print(f"{'  nicht im Report erfasst':<32}{gap:>8.2f} s  (Skill-Eval, Checkpoints, Start)")
    print()
    for target in (2e9, 10e9, 30e9):
        print(f"{target / 1e9:>5.0f} Mrd. Steps: {target / effective / 86400:>6.1f} Tage "
              f"bei diesem Tempo")


if __name__ == "__main__":
    main()
