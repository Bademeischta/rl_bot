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
    # Reward vor einem Zero-Sum-Wrapper (Review R10): im 1v1 mit Zero-Sum ist step_reward sonst 0
    ("raw_step_reward", "raw_step_reward"),
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


def win_rate_ci(wins: int, losses: int, draws: int, z: float = 1.96) -> tuple[float, float, float]:
    """Gewinnrate (Remis = halber Sieg) mit Wilson-Konfidenzintervall über die Spiele (Review R12).

    z = 1,96 ergibt 95 %. Remis als halbe Siege machen das Intervall etwas zu breit (konservativ).
    Liefert (Rate, untere Grenze, obere Grenze); ohne Spiele nan.
    """
    n = wins + losses + draws
    if n <= 0:
        return math.nan, math.nan, math.nan
    p = (wins + 0.5 * draws) / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def t_quantile_975(df: int) -> float:
    """97,5-%-Quantil der t-Verteilung (zweiseitiges 95-%-Intervall), Reihenentwicklung um die
    Normalverteilung (Genauigkeit besser als 0,5 % ab df = 5, ohne scipy)."""
    z = 1.959963984540054
    if df <= 0:
        return math.nan
    return (z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def mean_ci95(values: list[float]) -> tuple[float, float, float, float]:
    """(Mittel, KI unten, KI oben, Standardabweichung) mit t-Intervall; nan bei weniger als 2 Werten."""
    n = len(values)
    if n < 2:
        return (values[0] if values else math.nan), math.nan, math.nan, math.nan
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    half = t_quantile_975(n - 1) * sd / math.sqrt(n)
    return m, m - half, m + half, sd


def duel_stats(d: dict) -> dict:
    """Kennzahlen eines duel.exe-Ergebnisses aus Sicht von A (Hauptkriterium, Review Schritt 1).

    goal_diff:        mittlere Tordifferenz A - B pro Spiel mit 95-%-t-Intervall über die Spiele
    win_rate:         Gewinnrate (Remis = halber Sieg) mit 95-%-Wilson-Intervall
    goals_per_minute: Tore pro Minute Spielzeit für A, B und zusammen
    Ohne Einzelspiele (JSON von vor dem Umbau) nur Mittelwerte, Intervall nan.
    """
    games = d.get("per_game") or []
    out: dict = {}
    if games:
        diffs = [g["goals_a"] - g["goals_b"] for g in games]
        m, low, high, sd = mean_ci95(diffs)
        minutes = sum(g.get("game_seconds", 0.0) for g in games) / 60.0
        goals_a = sum(g["goals_a"] for g in games)
        goals_b = sum(g["goals_b"] for g in games)
        n = len(games)
    else:
        n = d.get("games", 0)
        goals_a, goals_b = d.get("goals_a", 0), d.get("goals_b", 0)
        m = (goals_a - goals_b) / n if n else math.nan
        low = high = sd = math.nan
        minutes = n * d.get("max_seconds", 120) / 60.0
    out["games"] = n
    out["goal_diff"], out["goal_diff_ci_low"], out["goal_diff_ci_high"], out["goal_diff_sd"] = m, low, high, sd
    rate, wlow, whigh = win_rate_ci(d.get("wins_a", 0), d.get("wins_b", 0), d.get("draws", 0))
    out["win_rate"], out["win_rate_ci_low"], out["win_rate_ci_high"] = rate, wlow, whigh
    out["goals_per_minute_a"] = goals_a / minutes if minutes else math.nan
    out["goals_per_minute_b"] = goals_b / minutes if minutes else math.nan
    out["goals_per_minute"] = (goals_a + goals_b) / minutes if minutes else math.nan
    out["distinct_games"] = d.get("distinct_games")
    return out


def games_for_half_width(sd: float, half_width: float) -> int:
    """Spielanzahl, bei der das 95-%-Intervall der mittleren Tordifferenz +-half_width breit ist."""
    return math.ceil((1.96 * sd / half_width) ** 2)


def count_bad(rows: list[dict[str, str]], key: str) -> int:
    """Wie viele Iterationen in einer Spalte nan, inf oder leer sind."""
    return sum(1 for r in rows if is_bad_value(r.get(key)))
