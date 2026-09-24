"""Zeigt, wie sich die Iterationszeiten über einen Lauf entwickeln.

    python tools/timing_trend.py runs/lucy_1v1 --buckets 12

Nützlich, um zu unterscheiden, ob ein Lauf langsam startet (Aufwärmen, kurze Episoden mit
vielen Resets) oder ob eine Konfiguration dauerhaft teuer ist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from show_metrics import read_rows

COLUMNS = [
    ("Total Iteration Time", "iter_s"),
    ("Collection Time", "sammeln"),
    ("Policy Infer Time", "inferenz"),
    ("Env Step Time", "sim"),
    ("Consumption Time", "lernen"),
    ("ball_touch_ratio", "touch"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--buckets", type=int, default=10)
    a = ap.parse_args()

    path = a.run if a.run.suffix == ".csv" else a.run / "metrics.csv"
    rows = read_rows(path)
    size = max(1, len(rows) // a.buckets)

    header = f"{'Iterationen':>16}  {'Steps':>14}"
    for _, label in COLUMNS:
        header += f"  {label:>9}"
    header += f"  {'SPS':>9}"
    print(header)

    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        if len(chunk) < size // 2:
            continue

        def mean(key: str) -> float:
            vals = [float(r[key]) for r in chunk if r.get(key)]
            return sum(vals) / len(vals) if vals else float("nan")

        line = f"{start:>7}-{start + len(chunk):<8}  {float(chunk[-1]['Cumulative Timesteps']):>14,.0f}"
        for key, _ in COLUMNS:
            line += f"  {mean(key):>9.3f}"
        steps_per_iter = mean("Timesteps Collected")
        line += f"  {steps_per_iter / mean('Total Iteration Time'):>9,.0f}"
        print(line)


if __name__ == "__main__":
    main()
