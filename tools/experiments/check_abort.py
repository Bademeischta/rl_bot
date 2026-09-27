"""Abbruchkriterien für einen laufenden Experiment-Lauf (wird von run_experiment.ps1 gepollt).

    python tools/experiments/check_abort.py runs/exp_x/metrics.csv [--warmup 100] [--baseline-sps N]

Rückgabe 0 = weiterlaufen, 3 = ABBRECHEN (Grund auf stdout), 4 = WARNUNG (weiterlaufen).
Die Schwellen sind bewusst weit gewählt: Ein Experiment kostet lokal rund 25-40 Minuten, ein
Fehlalarm kostet mehr als ein schlechter Lauf. Begründung je Kriterium in tools/experiments/README.md.

Kriterien (alle auf metrics.csv, das pro Iteration = ~100k Steps eine Zeile bekommt):
  1. nan, inf oder leeres Feld in Entropie, Value Loss, KL, Advantage oder Val Target -> sofort
     abbrechen. Einmal vergiftet, ist der Rest des Laufs wertlos. (Review-Befund R5: Der Trainer
     schreibt nan/inf jetzt wörtlich; vorher als leeres Feld, und leere Felder wurden hier
     übersprungen, ein NaN-Lauf lief also bis zum Ende.) "Average Episode Reward" ist bewusst
     nicht dabei: Dort heißt nan nur "in dieser Iteration endete keine Episode" (im Hauptlauf
     2x, 31 bzw. 77 Iterationen nach einem Neustart); summarize.py zählt diese Iterationen.
  2. Explodierender Value Loss: Median der letzten 20 Iterationen > VALUE_LOSS_FACTOR (10) mal
     Median des Referenzfensters (Iterationen warmup .. warmup+100 desselben Laufs) UND absolut
     über VALUE_LOSS_ABS_MIN (100). Erst nach der Aufwärmphase (Default 100 Iterationen), weil
     K3 den Critic absichtlich neu einschwingen lässt (AUDIT.md 7.3).
  3. SPS-Einbruch: Mittel "Overall Steps/Second" der letzten 20 Iterationen < SPS_FRACTION (0,4)
     mal Referenz (Mittel der Iterationen 20..120, oder --baseline-sps). Normale Streuung liegt
     bei bis zu 7 %, thermisches Drosseln bei rund 30 % (docs/phase0_results.md); unter 40 % läuft
     etwas anderes schief (CPU-Fallback, anderer Prozess, Swap) und der Lauf misst Unsinn.
  4. Entropie-Kollaps (nur Warnung): Policy Entropy < ENTROPY_WARN (2,5) - die Schwelle aus
     AUDIT.md H2, ab der ent_coef zurückgedreht werden soll. Kein Abbruch, das Ergebnis ist
     trotzdem auswertbar.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_util import column, has_non_finite, mean, median, read_rows  # noqa: E402

VALUE_LOSS_FACTOR = 10.0
VALUE_LOSS_ABS_MIN = 100.0
SPS_FRACTION = 0.4
ENTROPY_WARN = 2.5
WINDOW = 20

NAN_KEYS = ["Policy Entropy", "Value Function Loss", "Mean KL Divergence",
            "Avg Advantage", "Avg Val Target"]

ABORT, WARN, OK = 3, 4, 0


def evaluate(rows: list[dict[str, str]], warmup: int = 100, baseline_sps: float | None = None
             ) -> tuple[int, list[str]]:
    """Liefert (Exit-Code, Meldungen)."""
    messages: list[str] = []
    if not rows:
        return OK, ["noch keine Iterationen"]

    bad = has_non_finite(rows, NAN_KEYS)
    if bad:
        return ABORT, [f"ABBRUCH: nan/inf/leer in {', '.join(bad)} (Iteration {len(rows)})"]

    code = OK
    n = len(rows)

    # 2. Value Loss
    if n >= warmup + 100 + WINDOW:
        ref = median(column(rows[warmup:warmup + 100], "Value Function Loss"))
        recent = median(column(rows[-WINDOW:], "Value Function Loss"))
        if recent > VALUE_LOSS_FACTOR * ref and recent > VALUE_LOSS_ABS_MIN:
            return ABORT, [f"ABBRUCH: Value Loss explodiert: Median zuletzt {recent:.2f} gegen "
                           f"Referenz {ref:.2f} (Faktor {recent / ref:.1f} > {VALUE_LOSS_FACTOR})"]

    # 3. SPS
    if n >= 120 + WINDOW or (baseline_sps and n >= WINDOW):
        ref_sps = baseline_sps if baseline_sps else mean(column(rows[20:120], "Overall Steps/Second"))
        recent_sps = mean(column(rows[-WINDOW:], "Overall Steps/Second"))
        if ref_sps and recent_sps < SPS_FRACTION * ref_sps:
            return ABORT, [f"ABBRUCH: SPS eingebrochen: zuletzt {recent_sps:,.0f} gegen Referenz "
                           f"{ref_sps:,.0f} ({recent_sps / ref_sps:.0%} < {SPS_FRACTION:.0%})"]

    # 4. Entropie (Warnung)
    if n >= WINDOW:
        ent = mean(column(rows[-WINDOW:], "Policy Entropy"))
        if ent < ENTROPY_WARN:
            code = WARN
            messages.append(f"WARNUNG: Entropie {ent:.3f} < {ENTROPY_WARN} (AUDIT.md H2: ent_coef zurückdrehen)")

    messages.append(f"ok: {n} Iterationen, letzte Steps {rows[-1].get('Cumulative Timesteps')}")
    return code, messages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--baseline-sps", type=float, default=None,
                    help="SPS-Referenz von außen (z. B. aus dem Baseline-Lauf)")
    a = ap.parse_args()
    if not a.metrics.exists():
        print("noch keine metrics.csv")
        return OK
    code, messages = evaluate(read_rows(a.metrics), a.warmup, a.baseline_sps)
    for m in messages:
        print(m)
    return code


if __name__ == "__main__":
    sys.exit(main())
