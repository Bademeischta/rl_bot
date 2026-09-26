"""Tests für die Experiment-Werkzeuge (Schritt 2): Abbruchkriterien, Zusammenfassung, Vergleich."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("RLBOT_BUILD_DIR", str(ROOT / "build" / "cpp_cu128")))
sys.path.insert(0, str(ROOT / "tools" / "experiments"))

import check_abort  # noqa: E402
import compare  # noqa: E402
import summarize  # noqa: E402
from metrics_util import has_non_finite, read_rows, window_mean  # noqa: E402

COLUMNS = ["Cumulative Timesteps", "Timesteps Collected", "Overall Steps/Second", "Policy Entropy",
           "Value Function Loss", "Mean KL Divergence", "Average Episode Reward", "Avg Advantage", "Avg Val Target",
           "SB3 Clip Fraction", "ep_end_goal", "ep_end_timeout", "Truncated Steps", "Skill Rating 1v1"]


def write_metrics(path: Path, n: int, **overrides):
    """Schreibt n Iterationen mit konstanten Standardwerten; overrides: key -> f(i)."""
    rows = []
    for i in range(n):
        row = {"Cumulative Timesteps": 2_700_000_000 + (i + 1) * 100_000, "Timesteps Collected": 100_000,
               "Overall Steps/Second": 68_000, "Policy Entropy": 3.58, "Value Function Loss": 10.0,
               "Mean KL Divergence": 0.0027, "Average Episode Reward": 2049.0, "Avg Advantage": 0.4,
               "Avg Val Target": 10.0,
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


def test_has_non_finite_flags_nan_inf_and_empty_fields(tmp_path):
    """Review-Befund R5: leere Felder zählen wie nan/inf (der Trainer schrieb nan früher leer)."""
    p = tmp_path / "metrics.csv"
    for bad in ("", "nan", "inf", "-inf", "-nan(ind)"):
        write_metrics(p, 3, **{"Value Function Loss": lambda i, bad=bad: bad if i == 1 else 1.0})
        assert has_non_finite(read_rows(p), ["Value Function Loss"]) == ["Value Function Loss"], bad
    write_metrics(p, 3)
    assert has_non_finite(read_rows(p), ["Value Function Loss"]) == []


def test_columns_added_later_are_not_flagged_in_older_rows(tmp_path):
    """Ältere Zeilen sind kürzer, wenn eine Spalte später dazukam (M1): das ist kein Fehler."""
    p = tmp_path / "metrics.csv"
    p.write_text('"Cumulative Timesteps","A","B"\n100,1\n200,2,3\n', encoding="utf-8")
    rows = read_rows(p)
    assert rows[0]["B"] is None
    assert has_non_finite(rows, ["B"]) == []


def test_half_written_last_line_is_ignored(tmp_path):
    """check_abort liest, während der Trainer schreibt: eine Zeile ohne Zeilenende zählt nicht."""
    p = tmp_path / "metrics.csv"
    p.write_text('"Cumulative Timesteps","Policy Entropy","Value Function Loss"\n100,3.5,0.2\n200,3.4,',
                 encoding="utf-8")
    rows = read_rows(p)
    assert len(rows) == 1
    assert has_non_finite(rows, ["Value Function Loss"]) == []


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
    assert "nan/inf/leer" in msgs[0] and "Policy Entropy" in msgs[0]


@pytest.mark.parametrize("key", check_abort.NAN_KEYS)
def test_empty_field_in_a_learning_metric_aborts(tmp_path, key):
    p = tmp_path / "metrics.csv"
    write_metrics(p, 5, **{key: lambda i: "" if i == 2 else 0.5})
    code, msgs = check_abort.evaluate(read_rows(p))
    assert code == check_abort.ABORT, msgs
    assert key in msgs[0]


def test_nan_episode_reward_does_not_abort_but_is_counted(tmp_path):
    """Nutzerentscheidung zu R5: nan im Episoden-Reward = keine Episode beendet, kein Abbruch."""
    p = tmp_path / "metrics.csv"
    write_metrics(p, 40, **{"Average Episode Reward": lambda i: "nan" if i in (5, 31) else 2000.0})
    code, msgs = check_abort.evaluate(read_rows(p))
    assert code == check_abort.OK, msgs
    assert summarize.summarize_metrics(read_rows(p))["ep_reward_nan_iterations"] == 2


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


def _duel_json(results: list[tuple[int, int]], seconds: float = 300.0) -> dict:
    """Duell-Ergebnis im Format von duel.exe (per_game mit Toren je Spiel)."""
    per_game = [{"a_blue": i % 2 == 0, "goals_a": a, "goals_b": b, "game_seconds": seconds} for i, (a, b) in enumerate(results)]
    return {"games": len(results), "max_seconds": seconds,
            "goals_a": sum(a for a, _ in results), "goals_b": sum(b for _, b in results),
            "wins_a": sum(a > b for a, b in results), "wins_b": sum(b > a for a, b in results),
            "draws": sum(a == b for a, b in results), "per_game": per_game}


def test_compare_table_marks_baseline_and_deltas(tmp_path):
    base = tmp_path / "exp_baseline_2026-10-01"
    exp = tmp_path / "exp_h2_2026-10-01"
    # Einzel-Ladder des Lauf-Ordners: darf in compare NICHT auftauchen (Review R12)
    make_result(base, "baseline", 0.30, 3.58,
                ratings={"exp_baseline/2704829056": {"mu": 25, "sigma": 8.3, "conservative": 0.1},
                         "exp_baseline/2804829056": {"mu": 30, "sigma": 2.0, "conservative": 24.0}})
    # 100 Spiele: 40x 1:0, 20x 0:1, 40x 0:0 -> Tordifferenz +0,2 pro Spiel
    make_result(exp, "h2_ent_coef_0004", 0.36, 3.30, duel=_duel_json([(1, 0)] * 40 + [(0, 1)] * 20 + [(0, 0)] * 40))
    experiments = [compare.load_experiment(base), compare.load_experiment(exp)]
    table = compare.build_table(experiments, experiments[0], None)
    lines = table.splitlines()
    assert lines[0].startswith("| Experiment |")
    assert "Tordifferenz/Spiel [95-%-KI]" in lines[0] and "Gewinnrate" in lines[0] and "Tore/min" in lines[0]
    assert "baseline (Baseline)" in lines[2]
    assert "24.00" not in table                   # keine Einzel-Ladder
    assert "0.360 (+0.060 (+20.0%))" in lines[3]  # Delta gegen Baseline
    assert "**+0.200** [" in lines[3]             # Tordifferenz pro Spiel
    assert "60.0% [" in lines[3]                  # Gewinnrate (40 + 40/2) / 100
    assert "0.080 : 0.040" in lines[3]            # Tore/min: 40 bzw. 20 Tore in 500 min
    hints = compare.verdict_hints(experiments, experiments[0])
    assert "Hauptkriterium Duell: **besser**" in hints


def test_compare_hint_reports_duel_in_the_noise(tmp_path):
    base = tmp_path / "exp_baseline_x"
    exp = tmp_path / "exp_h3_x"
    make_result(base, "baseline", 0.30, 3.58)
    make_result(exp, "h3_no_shuffle", 0.31, 3.58, duel=_duel_json([(1, 0)] * 11 + [(0, 1)] * 9 + [(0, 0)] * 20))
    experiments = [compare.load_experiment(base), compare.load_experiment(exp)]
    hints = compare.verdict_hints(experiments, experiments[0])
    assert "Hauptkriterium Duell: **im Rauschen**" in hints and "enthält 0" in hints


def test_duel_stats_goal_difference_with_t_interval():
    """Tordifferenz je Spiel mit t-Intervall; Referenz: t(0,975; df) aus der Tabelle."""
    from metrics_util import duel_stats, t_quantile_975
    for df, table in ((5, 2.5706), (10, 2.2281), (30, 2.0423), (100, 1.9840)):
        assert t_quantile_975(df) == pytest.approx(table, rel=5e-3), df
    d = _duel_json([(2, 0), (1, 1), (0, 1), (3, 1), (0, 0), (1, 0), (0, 2), (1, 1), (2, 1), (0, 0)])
    s = duel_stats(d)
    diffs = [2, 0, -1, 2, 0, 1, -2, 0, 1, 0]
    m = sum(diffs) / 10
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / 9)
    assert s["goal_diff"] == pytest.approx(0.3)
    assert s["goal_diff_sd"] == pytest.approx(sd)
    assert s["goal_diff_ci_high"] - s["goal_diff"] == pytest.approx(2.2622 * sd / math.sqrt(10), rel=2e-3)
    assert s["goals_per_minute"] == pytest.approx(17 / 50)   # 10 + 7 Tore in 10 x 5 min
    assert s["win_rate"] == pytest.approx((4 + 0.5 * 4) / 10)


def test_win_rate_ci_matches_the_wilson_interval():
    """Referenzwerte des 95-%-Wilson-Intervalls: 50/100 -> [0,4038; 0,5962], 0/10 -> [0; 0,2775]."""
    from metrics_util import win_rate_ci
    rate, low, high = win_rate_ci(50, 50, 0)
    assert rate == pytest.approx(0.5) and low == pytest.approx(0.4038, abs=1e-4) and high == pytest.approx(0.5962, abs=1e-4)
    rate, low, high = win_rate_ci(0, 10, 0)
    assert rate == 0 and low == 0 and high == pytest.approx(0.2775, abs=1e-4)
    assert win_rate_ci(60, 30, 10)[0] == pytest.approx(0.65)   # Remis = halber Sieg
    assert math.isnan(win_rate_ci(0, 0, 0)[0])


def test_summarize_duel_has_win_rate_and_ci(tmp_path):
    d = tmp_path / "duel.json"
    d.write_text(json.dumps({"games": 100, "goals_a": 60, "goals_b": 40, "wins_a": 55, "wins_b": 40,
                             "draws": 5}), encoding="utf-8")
    out = summarize.load_duel(d)
    assert out["win_rate"] == pytest.approx(0.575)
    assert out["win_rate_ci_low"] < 0.575 < out["win_rate_ci_high"]


DUEL_EXE = BUILD_DIR / ("duel.exe" if sys.platform == "win32" else "duel")
MAIN_CKPTS = ROOT / "runs" / "lucy_1v1" / "checkpoints"


def _real_checkpoints(n: int) -> list[Path]:
    if not MAIN_CKPTS.exists():
        return []
    c = sorted((p for p in MAIN_CKPTS.iterdir() if (p / "PPO_POLICY.lt").exists()), key=lambda p: int(p.name))
    return c[-n:] if len(c) >= n else []


@pytest.mark.skipif(not DUEL_EXE.exists() or not _real_checkpoints(3), reason="duel.exe oder echte Checkpoints fehlen")
def test_compare_plays_one_joint_ladder_with_baseline_start_and_end(tmp_path, capsys):
    """Echter Pfad: compare.py spielt die gemeinsame Ladder mit duel.exe auf echten Checkpoints
    (nur gelesen). Teilnehmer: Baseline-Start, Baseline-Ende, Experiment-Ende; Einzel-Ladders der
    Lauf-Ordner werden ignoriert."""
    start, base_end, exp_end = _real_checkpoints(3)
    base = tmp_path / "exp_baseline_x"
    exp = tmp_path / "exp_h2_x"
    make_result(base, "baseline", 0.30, 3.58,
                ratings={"exp_baseline_x/999": {"mu": 99, "sigma": 0.5, "conservative": 97.5}})
    make_result(exp, "h2_ent_coef_0004", 0.33, 3.40)
    for folder, s_ckpt, e_ckpt in ((base, start, base_end), (exp, start, exp_end)):
        s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        s["start_checkpoint"], s["end_checkpoint"] = str(s_ckpt), str(e_ckpt)
        (folder / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    out = tmp_path / "compare.md"
    sys.argv = ["compare.py", str(base), str(exp), "--ladder-games", "2", "--exe", str(DUEL_EXE), "--out", str(out)]
    assert compare.main() == 0
    ladder = json.loads((tmp_path / "joint_ladder.json").read_text(encoding="utf-8"))
    assert set(ladder["ratings"]) == {"Baseline-Start", "Baseline-Ende", "h2_ent_coef_0004-Ende"}
    assert len(ladder["duels"]) == 3 and all(d["games"] == 2 for d in ladder["duels"])
    md = out.read_text(encoding="utf-8")
    assert "## Gemeinsame TrueSkill-Ladder" in md and "Baseline-Start" in md
    assert "97.50" not in md                       # Einzel-Ladder des Lauf-Ordners nicht benutzt

def test_compare_requires_summary(tmp_path):
    (tmp_path / "leer").mkdir()
    with pytest.raises(SystemExit):
        compare.load_experiment(tmp_path / "leer")


# --- R5 End-to-End: echte CSV aus dem C++-Writer -> check_abort.py ----------------------------

import os  # noqa: E402
import subprocess  # noqa: E402

BUILD = Path(os.environ.get("RLBOT_BUILD_DIR", str(ROOT / "build" / "cpp_cu128")))
WRITER = BUILD / ("write_metrics_csv.exe" if os.name == "nt" else "write_metrics_csv")
needs_writer = pytest.mark.skipif(not WRITER.exists(), reason=f"{WRITER} fehlt (bench/cpp/build.ps1)")


def _cpp_csv(path: Path, iterations: int, *extra: str) -> None:
    r = subprocess.run([str(WRITER), str(path), str(iterations), *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _check_abort_cli(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "tools" / "experiments" / "check_abort.py"), str(path)],
                          capture_output=True, text=True)


@needs_writer
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cpp_writer_nan_triggers_check_abort(tmp_path, value):
    p = tmp_path / "metrics.csv"
    _cpp_csv(p, 30)
    assert _check_abort_cli(p).returncode == check_abort.OK
    _cpp_csv(p2 := tmp_path / "bad.csv", 30, "--set", "17", "Policy Entropy", value)
    assert value in p2.read_text(encoding="utf-8").splitlines()[18]   # wörtlich geschrieben
    r = _check_abort_cli(p2)
    assert r.returncode == check_abort.ABORT, r.stdout + r.stderr
    assert "Policy Entropy" in r.stdout


@needs_writer
def test_cpp_writer_missing_key_is_empty_and_triggers_check_abort(tmp_path):
    p = tmp_path / "metrics.csv"
    _cpp_csv(p, 30, "--drop", "12", "Value Function Loss")
    r = _check_abort_cli(p)
    assert r.returncode == check_abort.ABORT, r.stdout + r.stderr
    assert "Value Function Loss" in r.stdout


@needs_writer
def test_cpp_writer_nan_episode_reward_does_not_abort(tmp_path):
    p = tmp_path / "metrics.csv"
    _cpp_csv(p, 30, "--set", "9", "Average Episode Reward", "nan")
    r = _check_abort_cli(p)
    assert r.returncode == check_abort.OK, r.stdout + r.stderr


# --- R11: Bündel (mehr als eine Änderung gegenüber der Baseline) -----------------------------

EXP_CONFIGS = ROOT / "train" / "configs" / "experiments"


def make_result_with_config(folder: Path, name: str, config: str):
    """Ergebnisordner wie von run_experiment.ps1: summary.json plus config.json (hier die echte
    Experiment-Config, mit BOM wie von PowerShell geschrieben)."""
    make_result(folder, name, 0.3, 3.5)
    raw = (EXP_CONFIGS / f"{config}.json").read_text(encoding="utf-8")
    (folder / "config.json").write_bytes(b"\xef\xbb\xbf" + raw.encode("utf-8"))


def test_compare_marks_k3_as_a_bundle_of_seven_values(tmp_path, capsys):
    base = tmp_path / "exp_baseline_x"
    make_result_with_config(base, "baseline", "baseline")
    make_result_with_config(tmp_path / "exp_k3_x", "k3_rewards", "k3_rewards")
    make_result_with_config(tmp_path / "exp_h2_x", "h2_ent_coef_0004", "h2_ent_coef_0004")
    make_result_with_config(tmp_path / "exp_zs_x", "zero_sum", "zero_sum")
    sys.argv = ["compare.py", str(base), str(tmp_path / "exp_k3_x"), str(tmp_path / "exp_h2_x"),
                str(tmp_path / "exp_zs_x")]
    assert compare.main() == 0
    md = capsys.readouterr().out
    k3_row = next(line for line in md.splitlines() if line.startswith("| k3_rewards"))
    assert "Bündel: 7 Werte" in k3_row
    h2_row = next(line for line in md.splitlines() if line.startswith("| h2_ent_coef_0004"))
    assert "Bündel" not in h2_row
    zs_row = next(line for line in md.splitlines() if line.startswith("| zero_sum"))
    assert "Bündel: 3 Werte" in zs_row
    assert "**Bündel aus 7 Änderungen**" in md and "rewards.in_air" in md
    assert "`learner.ent_coef` 0.01 → 0.004" in md


# --- R17: compare.py löst Glob-Muster selbst auf ------------------------------------------

def test_compare_expands_the_glob_pattern_itself_like_under_powershell(tmp_path):
    """PowerShell übergibt results/exp_* wörtlich an python.exe; subprocess mit Argumentliste
    macht genau das (keine Shell-Expansion). Zip-Dateien mit gleichem Präfix fallen heraus."""
    import subprocess
    make_result(tmp_path / "exp_baseline_x", "baseline", 0.30, 3.58)
    make_result(tmp_path / "exp_h2_x", "h2_ent_coef_0004", 0.33, 3.40)
    (tmp_path / "exp_baseline_x.zip").write_bytes(b"PK")
    (tmp_path / "exp_halb").mkdir()                      # abgebrochener Ordner ohne summary.json
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "experiments" / "compare.py"),
                        str(tmp_path / "exp_*"), "--ladder-games", "0"],
                       capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, r.stderr
    assert "| baseline (Baseline)" in r.stdout and "| h2_ent_coef_0004" in r.stdout
    assert "exp_baseline_x.zip" in r.stderr and "exp_halb" in r.stderr   # übersprungen, gemeldet

    r = subprocess.run([sys.executable, str(ROOT / "tools" / "experiments" / "compare.py"),
                        str(tmp_path / "gibtsnicht_*")], capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode != 0 and "trifft keinen" in r.stderr


# --- R4 Nachtrag: K1b-Diagnose im Smoke-Lauf (echte CSV aus dem C++-Writer) -------------------

def _check_k1b(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "tools" / "local" / "check_k1b.py"), str(path)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


@needs_writer
def test_check_k1b_passes_only_when_timeouts_bootstrap_from_the_final_obs(tmp_path):
    ok = tmp_path / "ok.csv"
    _cpp_csv(ok, 10, "--set", "0", "Timeout Truncations", "0",
             *[x for i in range(1, 10) for x in ("--set", str(i), "Timeout Truncations", "64",
                                                 "--set", str(i), "Trunc Bootstrap Reset Share", "0",
                                                 "--set", str(i), "Trunc Bootstrap V Diff", "0.3")])
    r = _check_k1b(ok)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "576" in r.stdout and "OK" in r.stdout

    bad = tmp_path / "bad.csv"
    _cpp_csv(bad, 5, *[x for i in range(5) for x in ("--set", str(i), "Timeout Truncations", "64",
                                                     "--set", str(i), "Trunc Bootstrap Reset Share", "1")])
    assert _check_k1b(bad).returncode == 1

    none = tmp_path / "none.csv"
    _cpp_csv(none, 5, *[x for i in range(5) for x in ("--set", str(i), "Timeout Truncations", "0")])
    assert _check_k1b(none).returncode == 2

    old = tmp_path / "old.csv"                            # Trainer ohne Diagnose-Spalten
    _cpp_csv(old, 5)
    assert _check_k1b(old).returncode == 1
