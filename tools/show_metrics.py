"""Zeigt den Verlauf eines Trainingslaufs aus runs/<name>/metrics.csv.

    python tools/show_metrics.py runs/sanity/metrics.csv [--rows 10]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

COLUMNS = [
    ("Cumulative Timesteps", "steps", "{:>12,.0f}"),
    ("ball_touch_ratio", "touch", "{:>8.5f}"),
    ("Average Episode Reward", "ep_rew", "{:>8.2f}"),
    ("Policy Entropy", "entropy", "{:>8.3f}"),
    ("Value Function Loss", "val_loss", "{:>9.3f}"),
    ("in_air_ratio", "in_air", "{:>7.3f}"),
    ("boost_held", "boost", "{:>6.3f}"),
    ("ball_speed", "ball_v", "{:>8.1f}"),
    ("Skill Rating 1v1", "skill", "{:>7.0f}"),
    ("Collected Steps/Second", "sps", "{:>9,.0f}"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Liest metrics.csv und überspringt wiederholte Kopfzeilen.

    Jeder Trainingsprozess schreibt beim Start seine Kopfzeile, ein fortgesetzter Lauf hängt
    also mitten in der Datei eine weitere an.
    """
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return [r for r in rows if r.get("Cumulative Timesteps") != "Cumulative Timesteps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--rows", type=int, default=10)
    a = ap.parse_args()

    rows = read_rows(a.path)
    if not rows:
        print("keine Daten")
        return

    present = [c for c in COLUMNS if c[0] in rows[0]]
    print("  ".join(f"{label:>12}" for _, label, _ in present))

    step = max(1, len(rows) // a.rows)
    for row in rows[::step] + ([rows[-1]] if len(rows) > 1 else []):
        cells = []
        for key, _, fmt in present:
            try:
                cells.append(f"{fmt.format(float(row[key])):>12}")
            except (ValueError, KeyError):
                cells.append(f"{'-':>12}")
        print("  ".join(cells))

    print(f"\n{len(rows)} Iterationen, letzter Stand: "
          f"{float(rows[-1]['Cumulative Timesteps']):,.0f} Steps")

    # Trend der Ballkontaktrate: erstes vs. letztes Fünftel
    if "ball_touch_ratio" in rows[0] and len(rows) >= 10:
        n = max(1, len(rows) // 5)
        first = sum(float(r["ball_touch_ratio"]) for r in rows[:n]) / n
        last = sum(float(r["ball_touch_ratio"]) for r in rows[-n:]) / n
        print(f"Ballkontaktrate: erstes Fünftel {first:.5f} -> letztes Fünftel {last:.5f} "
              f"({'steigend' if last > first else 'nicht steigend'}, Faktor {last / first:.1f}x)"
              if first > 0 else
              f"Ballkontaktrate: erstes Fünftel {first:.5f} -> letztes Fünftel {last:.5f}")


if __name__ == "__main__":
    main()
