"""Gemeinsame Helfer für die Experiment-Skripte: metrics.csv lesen, Fenster mitteln.

Wird von check_abort.py, summarize.py und compare.py benutzt; ohne Abhängigkeiten außer
der Standardbibliothek, damit es auch in einem nackten Python läuft.
"""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path

# Spalten, die in der Zusammenfassung und im Vergleich auftauchen (Name in metrics.csv, Kurzname)
KEY_COLUMNS = [
    ("Cumulative Timesteps", "steps"),
    ("Overall Steps/Second", "sps"),
    ("Collected Steps/Second", "sps_collect"),
    ("Average Episode Reward", "ep_reward"),
    ("Average Step Reward", "step_reward"),
    ("ep_end_goal", "ep_end_goal"),
    ("ep_end_timeout", "ep_end_timeout"),
    ("ep_end_notouch", "ep_end_notouch"),
    ("ep_end_time", "ep_end_time"),
    ("ep_length_steps", "ep_length_steps"),
    ("ball_touch_ratio", "ball_touch"),
    ("in_air_ratio", "in_air"),
    ("boost_held", "boost_held"),
    ("Policy Entropy", "entropy"),
    ("SB3 Clip Fraction", "clip_fraction"),
    ("Mean KL Divergence", "kl"),
    ("Value Function Loss", "value_loss"),
    ("Avg Val Target", "val_target"),
    ("Avg Advantage", "advantage"),
    ("Truncated Steps", "truncated_steps"),
    # K1b-Diagnose (Review R4, AUDIT.md §7.2b): Reset-Share muss 0 sein
    ("Timeout Truncations", "timeout_truncations"),
    ("Trunc Bootstrap Reset Share", "trunc_reset_share"),
    ("Trunc Bootstrap V Diff", "trunc_v_diff"),
    ("Skill Rating 1v1", "skill_rating"),
    ("Cumulative Model Updates", "model_updates"),
    ("Total Iteration Time", "iter_s"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Liest metrics.csv. Wiederholte Kopfzeilen (Läufe vor Audit M1) werden übersprungen.

    Eine letzte Zeile ohne Zeilenende wird ignoriert: check_abort.py liest die Datei, während der
    Trainer sie schreibt, und eine halb geschriebene Zeile hätte sonst leere Felder (Review R5).
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    return [r for r in rows if r.get("Cumulative Timesteps") not in (None, "", "Cumulative Timesteps")]


def to_float(value) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def column(rows: list[dict[str, str]], key: str) -> list[float]:
    return [to_float(r.get(key)) for r in rows]


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def mean(values: list[float]) -> float:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else math.nan


def median(values: list[float]) -> float:
    vals = sorted(finite(values))
    if not vals:
        return math.nan
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def window_mean(rows: list[dict[str, str]], key: str, fraction: float = 0.2, minimum: int = 20) -> float:
    """Mittel über die letzten `fraction` der Iterationen (mindestens `minimum` Zeilen)."""
    n = max(minimum, int(len(rows) * fraction))
    return mean(column(rows[-n:], key))


def is_bad_value(raw) -> bool:
    """nan, inf und leere Felder (Review-Befund R5).

    Der Trainer schreibt nicht-endliche Werte wörtlich als nan/inf/-inf (Metrics.cpp); ein leeres
    Feld heißt, dass der Schlüssel in dieser Iteration fehlte, und zählt ebenfalls. None (Spalte gab
    es beim Schreiben der Zeile noch nicht, ältere Zeilen sind kürzer) zählt nicht.
    """
    if raw is None:
        return False
    if raw.strip() == "":
        return True
    return not math.isfinite(to_float(raw))


def has_non_finite(rows: list[dict[str, str]], keys: list[str]) -> list[str]:
    """Spalten, in denen mindestens ein Wert nan, inf oder leer ist (siehe is_bad_value)."""
    return [key for key in keys if any(is_bad_value(r.get(key)) for r in rows)]


def count_bad(rows: list[dict[str, str]], key: str) -> int:
    """Wie viele Iterationen in einer Spalte nan, inf oder leer sind."""
    return sum(1 for r in rows if is_bad_value(r.get(key)))
