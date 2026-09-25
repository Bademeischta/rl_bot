"""Prüft im Smoke-Lauf, dass der K1b-Fix greift (Review-Befund R4, AUDIT.md §7.2b).

    python tools/local/check_k1b.py runs/local_check_<datum>/sanity/metrics.csv

Der Learner meldet je Iteration:
  Timeout Truncations          Timeouts (pro Spieler), bei denen die Umgebung truncated hat
  Trunc Bootstrap Reset Share  Anteil davon, deren Bootstrap-Zustand die Reset-Beobachtung der
                               nächsten Episode ist (vor R4: 1, mit Fix: 0)
  Trunc Bootstrap V Diff       mittleres V(letzte Obs) - V(Reset-Obs)

Exit 0 = es gab Timeouts und der Reset-Anteil ist überall 0 (Fix greift), 1 = Reset-Anteil > 0
oder Spalten fehlen, 2 = im Lauf gab es keinen einzigen Timeout (nichts zu prüfen).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from metrics_util import read_rows, to_float  # noqa: E402

TIMEOUTS = "Timeout Truncations"
SHARE = "Trunc Bootstrap Reset Share"
DIFF = "Trunc Bootstrap V Diff"


def evaluate(rows: list[dict[str, str]]) -> tuple[int, list[str]]:
    if not rows or TIMEOUTS not in rows[0]:
        return 1, [f"Spalte '{TIMEOUTS}' fehlt: Trainer ohne K1b-Diagnose gebaut (Upstream-Patch alt?)"]
    timeouts = [to_float(r.get(TIMEOUTS)) for r in rows]
    total = sum(t for t in timeouts if math.isfinite(t))
    shares = [to_float(r.get(SHARE)) for r in rows if r.get(SHARE) not in (None, "")]
    diffs = [to_float(r.get(DIFF)) for r in rows if r.get(DIFF) not in (None, "")]
    measured = [s for s in shares if math.isfinite(s)]
    lines = [f"Iterationen: {len(rows)}, davon mit Timeouts: {sum(1 for t in timeouts if t > 0)}",
             f"Timeout Truncations gesamt (pro Spieler): {total:.0f}",
             f"Trunc Bootstrap Reset Share: max {max(measured) if measured else float('nan'):.3f} "
             f"über {len(measured)} Iterationen (muss 0 sein)",
             f"Trunc Bootstrap V Diff (V(letzte Obs) - V(Reset-Obs)): Mittel "
             f"{(sum(diffs) / len(diffs)) if diffs else float('nan'):.4f}"]
    if total <= 0:
        return 2, lines + ["Keine Timeouts im Lauf: der Fix ist hier nicht prüfbar (längeren Lauf oder kürzere Timeouts)"]
    if not measured:
        return 1, lines + [f"Timeouts vorhanden, aber '{SHARE}' nie gemessen"]
    if max(measured) > 0:
        return 1, lines + ["FEHLER: Timeouts bootstrappen (teilweise) von der Reset-Beobachtung"]
    return 0, lines + ["OK: Timeouts bootstrappen vom letzten Zustand vor dem Reset"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path)
    a = ap.parse_args()
    code, lines = evaluate(read_rows(a.metrics))
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
