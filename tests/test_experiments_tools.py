"""Tests für die Experiment-Werkzeuge (Schritt 2): Abbruchkriterien, Zusammenfassung, Vergleich."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiments"))

import check_abort  # noqa: E402
import compare  # noqa: E402
import summarize  # noqa: E402
from metrics_util import has_non_finite, read_rows, window_mean  # noqa: E402

COLUMNS = ["Cumulative Timesteps", "Timesteps Collected", "Overall Steps/Second", "Policy Entropy",
           "Value Function Loss", "Mean KL Divergence", "Average Episode Reward", "Avg Advantage",
           "SB3 Clip Fraction", "ep_end_goal", "ep_end_timeout", "Truncated Steps", "Skill Rating 1v1"]


def write_metrics(path: Path, n: int, **overrides):
    """Schreibt n Iterationen mit konstanten Standardwerten; overrides: key -> f(i)."""
    rows = []
    for i in range(n):
        row = {"Cumulative Timesteps": 2_700_000_000 + (i + 1) * 100_000, "Timesteps Collected": 100_000,
               "Overall Steps/Second": 68_000, "Policy Entropy": 3.58, "Value Function Loss": 10.0,
               "Mean KL Divergence": 0.0027, "Average Episode Reward": 2049.0, "Avg Advantage": 0.4,
               "SB3 Clip Fraction": 0.0245, "ep_end_goal": 0.3, "ep_end_timeout": 0.7,
               "Truncated Steps": 1050, "Skill Rating 1v1": 1930}
        for k, f in overrides.items():
            row[k] = f(i)
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


# --- metrics_util --------------------------------------------------------------

def test_read_rows_skips_repeated_headers(tmp_path):
    p = tmp_path / "metrics.csv"
    write_metrics(p, 3)
    text = p.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    p.write_text(text + header + "\n" + text.splitlines()[1] + "\n", encoding="utf-8")   # alter Stil
    rows = read_rows(p)
    assert len(rows) == 4
    assert all(r["Cumulative Timesteps"] != "Cumulative Timesteps" for r in rows)


def test_window_mean_uses_last_fraction(tmp_path):
    p = tmp_path / "metrics.csv"
    write_metrics(p, 100, **{"Policy Entropy": lambda i: 1.0 if i < 80 else 2.0})
    assert window_mean(read_rows(p), "Policy Entropy") == pytest.approx(2.0)


def test_has_non_finite_ignores_empty_fields(tmp_path):
    p = tmp_path / "metrics.csv"
    write_metrics(p, 3, **{"Value Function Loss": lambda i: "" if i == 1 else 1.0})
    assert has_non_finite(read_rows(p), ["Value Function Loss"]) == []
    write_metrics(p, 3, **{"Value Function Loss": lambda i: "nan" if i == 1 else 1.0})
    assert has_non_finite(read_rows(p), ["Value Function Loss"]) == ["Value Function Loss"]


# --- check_abort ---------------------------------------------------------------

def test_healthy_run_continues(tmp_path):
    p = tmp_path / "metrics.csv"
    write_metrics(p, 400)
    code, msgs = check_abort.evaluate(read_rows(p))
    assert code == check_abort.OK, msgs


def test_nan_aborts_immediately(tmp_path):
    p = tmp_path / "metrics.csv"
    write_metrics(p, 5, **{"Policy Entropy": lambda i: "nan" if i == 4 else 3.5})
    code, msgs = check_abort.evaluate(read_rows(p))
    assert code == check_abort.ABORT
    assert "nan/inf" in msgs[0] and "Policy Entropy" in msgs[0]


def test_value_loss_explosion_aborts_only_after_warmup(tmp_path):
    p = tmp_path / "metrics.csv"
    # Aufwärmphase (K3): Value Loss hoch in den ersten 100 Iterationen darf nicht abbrechen
    write_metrics(p, 150, **{"Value Function Loss": lambda i: 500.0 if i < 100 else 10.0})
    assert check_abort.evaluate(read_rows(p))[0] == check_abort.OK
    # Explosion nach der Referenzphase: Faktor 20 und absolut > 100 -> Abbruch
    write_metrics(p, 260, **{"Value Function Loss": lambda i: 10.0 if i < 240 else 200.0})
    code, msgs = check_abort.evaluate(read_rows(p))
    assert code == check_abort.ABORT and "Value Loss" in msgs[0]
    # Faktor 20, aber absolut klein (0,5 -> 10): kein Abbruch
    write_metrics(p, 260, **{"Value Function Loss": lambda i: 0.5 if i < 240 else 10.0})
    assert check_abort.evaluate(read_rows(p))[0] == check_abort.OK


def test_sps_collapse_aborts(tmp_path):
    p = tmp_path / "metrics.csv"
    write_metrics(p, 200, **{"Overall Steps/Second": lambda i: 68_000 if i < 180 else 20_000})
    code, msgs = check_abort.evaluate(read_rows(p))
    assert code == check_abort.ABORT and "SPS" in msgs[0]
    # 30 % Einbruch (thermisch) ist kein Abbruch
    write_metrics(p, 200, **{"Overall Steps/Second": lambda i: 68_000 if i < 180 else 48_000})
    assert check_abort.evaluate(read_rows(p))[0] == check_abort.OK
    # Referenz von außen wirkt schon nach 20 Iterationen
    write_metrics(p, 25, **{"Overall Steps/Second": lambda i: 20_000})
    assert check_abort.evaluate(read_rows(p), baseline_sps=68_000)[0] == check_abort.ABORT


def test_entropy_collapse_only_warns(tmp_path):
    p = tmp_path / "metrics.csv"
    write_metrics(p, 50, **{"Policy Entropy": lambda i: 2.0})
    code, msgs = check_abort.evaluate(read_rows(p))
    assert code == check_abort.WARN
    assert any("Entropie" in m for m in msgs)


# --- summarize -----------------------------------------------------------------

def test_summarize_metrics_and_running_stats(tmp_path):
    p = tmp_path / "metrics.csv"
    write_metrics(p, 200, **{"ep_end_goal": lambda i: 0.2 if i < 160 else 0.5})
    s = summarize.summarize_metrics(read_rows(p))
    assert s["iterations"] == 200
    assert s["steps_run"] == 200 * 100_000
    assert s["last_20pct"]["ep_end_goal"] == pytest.approx(0.5)
    assert s["first_20pct"]["ep_end_goal"] == pytest.approx(0.2)

    ckpt = tmp_path / "2704829056"
    ckpt.mkdir()
    (ckpt / "RUNNING_STATS.json").write_text(json.dumps({
        "cumulative_timesteps": 2704829056, "cumulative_model_updates": 157494, "epoch": 1,
        "reward_running_stats": {"mean": [0.1], "var": [15.12 ** 2 * (4_000_000 - 1)], "shape": 1,
                                 "count": 4_000_000}}), encoding="utf-8")
    rs = summarize.running_stats(ckpt)
    assert rs["return_std"] == pytest.approx(15.12, rel=1e-6)
    assert rs["cumulative_model_updates"] == 157494


def test_load_duel_computes_share_and_se(tmp_path):
    d = tmp_path / "duel.json"
    d.write_text(json.dumps({"games": 100, "goals_a": 60, "goals_b": 40, "wins_a": 55, "wins_b": 40,
                             "draws": 5}), encoding="utf-8")
    out = summarize.load_duel(d)
    assert out["goal_share_a"] == pytest.approx(0.6)
    assert out["goal_share_se"] == pytest.approx(math.sqrt(0.6 * 0.4 / 100))
    assert summarize.load_duel(None) is None


# --- compare -------------------------------------------------------------------

def make_result(folder: Path, name: str, goal: float, entropy: float, duel=None, ratings=None):
    folder.mkdir(parents=True)
    summary = {"name": name, "metrics": {"iterations": 1000,
               "last_20pct": {"ep_end_goal": goal, "ep_end_timeout": 1 - goal, "ep_length_steps": 2700,
                              "ball_touch": 0.03, "entropy": entropy, "clip_fraction": 0.03, "kl": 0.004,
                              "value_loss": 9.0, "val_target": 9.9, "truncated_steps": 1000, "sps": 68000}},
               "duel_baseline": duel, "ratings": ratings, "abort_reason": None}
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_compare_table_marks_baseline_and_deltas(tmp_path):
    base = tmp_path / "exp_baseline_2026-10-01"
    exp = tmp_path / "exp_h2_2026-10-01"
    make_result(base, "baseline", 0.30, 3.58,
                ratings={"exp_baseline/2704829056": {"mu": 25, "sigma": 8.3, "conservative": 0.1},
                         "exp_baseline/2804829056": {"mu": 30, "sigma": 2.0, "conservative": 24.0}})
    make_result(exp, "h2_ent_coef_0004", 0.36, 3.30,
                duel={"games": 100, "goals_a": 70, "goals_b": 40, "wins_a": 60, "wins_b": 30, "draws": 10,
                      "goal_share_a": 70 / 110, "goal_share_se": math.sqrt((70 / 110) * (40 / 110) / 110)})
    experiments = [compare.load_experiment(base), compare.load_experiment(exp)]
    table = compare.build_table(experiments, experiments[0])
    lines = table.splitlines()
    assert lines[0].startswith("| Experiment |")
    assert "baseline (Baseline)" in lines[2]
    assert "24.00 ± 2.00" in lines[2]              # TrueSkill des End-Checkpoints (höchste Steps)
    assert "0.360 (+0.060 (+20.0%))" in lines[3]  # Delta gegen Baseline
    assert "63.6% ±" in lines[3]                  # Duell-Toranteil
    hints = compare.verdict_hints(experiments, experiments[0])
    assert "signifikant besser" in hints


def test_compare_hint_reports_insignificant_duel(tmp_path):
    base = tmp_path / "exp_baseline_x"
    exp = tmp_path / "exp_h3_x"
    make_result(base, "baseline", 0.30, 3.58)
    make_result(exp, "h3_no_shuffle", 0.31, 3.58,
                duel={"games": 20, "goals_a": 11, "goals_b": 9, "wins_a": 10, "wins_b": 9, "draws": 1,
                      "goal_share_a": 0.55, "goal_share_se": math.sqrt(0.55 * 0.45 / 20)})
    experiments = [compare.load_experiment(base), compare.load_experiment(exp)]
    hints = compare.verdict_hints(experiments, experiments[0])
    assert "kein signifikanter Unterschied" in hints


def test_compare_requires_summary(tmp_path):
    (tmp_path / "leer").mkdir()
    with pytest.raises(SystemExit):
        compare.load_experiment(tmp_path / "leer")
