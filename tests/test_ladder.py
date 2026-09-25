"""Tests für die TrueSkill-Ladder. Läuft ohne Spiele, mit erfundenen Duell-Ergebnissen."""
from __future__ import annotations

import math

import pytest

from eval.ladder import (DuelResult, ENV, conservative, games_needed, interleave_goals,
                         load_ratings, save_ratings, standard_error, update_ratings)


def make_result(goals_a: int, goals_b: int, games: int = 50) -> DuelResult:
    return DuelResult(name_a="A", name_b="B", games=games, goals_a=goals_a, goals_b=goals_b,
                      wins_a=goals_a, wins_b=goals_b, draws=0)


def test_standard_error_matches_formula():
    assert standard_error(0.5, 100) == pytest.approx(0.05)
    assert standard_error(0.5, 625) == pytest.approx(0.02)
    assert standard_error(0.0, 10) == 0.0
    assert standard_error(0.5, 0) == float("inf")


def test_games_needed_is_the_inverse():
    assert games_needed(0.05) == 100
    assert games_needed(0.02) == 625
    n = games_needed(0.03)
    assert standard_error(0.5, n) <= 0.03


def test_clear_winner_gets_higher_rating():
    ratings = update_ratings({}, make_result(goals_a=40, goals_b=5))
    assert conservative(ratings["A"]) > conservative(ratings["B"])


def test_sigma_shrinks_with_more_games():
    few = update_ratings({}, make_result(10, 8))
    many = update_ratings({}, make_result(100, 80))
    assert many["A"].sigma < few["A"].sigma


def test_equal_result_keeps_ratings_close():
    ratings = update_ratings({}, make_result(25, 25))
    assert abs(ratings["A"].mu - ratings["B"].mu) < 1.0


def test_goalless_duel_leaves_ratings_at_default():
    ratings = update_ratings({}, make_result(0, 0))
    default = ENV.create_rating()
    for name in ("A", "B"):
        assert ratings[name].mu == pytest.approx(default.mu)
        assert ratings[name].sigma == pytest.approx(default.sigma)


def test_ratings_survive_save_and_load(tmp_path):
    ratings = update_ratings({}, make_result(30, 10))
    path = tmp_path / "ratings.json"
    save_ratings(ratings, path)
    loaded = load_ratings(path)

    assert set(loaded) == set(ratings)
    for name in ratings:
        assert loaded[name].mu == pytest.approx(ratings[name].mu, abs=1e-9)
        assert loaded[name].sigma == pytest.approx(ratings[name].sigma, abs=1e-9)


def test_rating_difference_grows_monotonically():
    """Je klarer der Toranteil, desto größer der Abstand der Ratings."""
    gaps = []
    for goals_a in (25, 35, 45):
        ratings = update_ratings({}, make_result(goals_a, 50 - goals_a))
        gaps.append(ratings["A"].mu - ratings["B"].mu)
    assert gaps[0] < gaps[1] < gaps[2]


def test_goals_are_interleaved_evenly():
    assert interleave_goals(3, 3) == [True, False, True, False, True, False]
    assert interleave_goals(3, 3, a_first_on_tie=False) == [False, True, False, True, False, True]
    assert interleave_goals(4, 0) == [True] * 4
    assert interleave_goals(0, 3) == [False] * 3
    assert interleave_goals(0, 0) == []

    # In jedem Präfix darf der Anteil höchstens um ein Tor vom Gesamtanteil abweichen
    seq = interleave_goals(7, 13)
    assert sum(seq) == 7 and len(seq) == 20
    for i in range(1, len(seq) + 1):
        expected = 7 * i / 20
        assert abs(sum(seq[:i]) - expected) <= 1.0


def test_rating_update_is_symmetric():
    """Wer A und wer B heißt, darf das Ergebnis nicht verändern."""
    forward = update_ratings({}, DuelResult("A", "B", 50, 35, 15, 35, 15, 0))
    backward = update_ratings({}, DuelResult("B", "A", 50, 15, 35, 15, 35, 0))
    assert forward["A"].mu == pytest.approx(backward["A"].mu, abs=1e-9)
    assert forward["B"].mu == pytest.approx(backward["B"].mu, abs=1e-9)


def test_trueskill_difference_of_8_3_is_about_80_percent():
    """Seer-Arbeit: rund 8,3 TrueSkill-Differenz entsprechen etwa 80 % Siegquote."""
    strong = ENV.create_rating(mu=33.3, sigma=1.0)
    weak = ENV.create_rating(mu=25.0, sigma=1.0)
    delta_mu = strong.mu - weak.mu
    denom = math.sqrt(2 * ENV.beta ** 2 + strong.sigma ** 2 + weak.sigma ** 2)
    win_prob = 0.5 * (1 + math.erf(delta_mu / (denom * math.sqrt(2))))
    assert 0.75 < win_prob < 0.95


# --- Rating-Schlüssel (Audit M4) -------------------------------------------

def test_rating_key_contains_run_name():
    from pathlib import Path
    from eval.ladder import rating_key
    assert rating_key(Path("runs/lucy_1v1/checkpoints/2704829056/PPO_POLICY.lt")) == "lucy_1v1/2704829056"
    assert rating_key(Path("runs/lucy_1v1/checkpoints/2704829056")) == "lucy_1v1/2704829056"
    # gleiche Step-Zahl in zwei Läufen -> zwei Einträge
    assert (rating_key(Path("runs/sanity/checkpoints/5053568"))
            != rating_key(Path("runs/archive/lucy_1v1_net1024/checkpoints/5053568")))
    assert rating_key(Path("runs/archive/lucy_1v1_net1024/checkpoints/5053568")) == "lucy_1v1_net1024/5053568"


def test_rating_key_falls_back_to_folder_name():
    from pathlib import Path
    from eval.ladder import rating_key
    assert rating_key(Path("some/where/ckpt_a/PPO_POLICY.lt")) == "ckpt_a"
    assert rating_key(Path("ckpt_b")) == "ckpt_b"


def test_default_ratings_path_is_per_run(tmp_path):
    from eval.ladder import RATINGS_PATH, default_ratings_path
    assert default_ratings_path(tmp_path / "runs" / "x") == tmp_path / "runs" / "x" / "ratings.json"
    assert default_ratings_path(None) == RATINGS_PATH


def test_latest_checkpoint_is_numeric_and_per_run(tmp_path, monkeypatch):
    from deploy.watch import latest_checkpoint
    run = tmp_path / "runs" / "a"
    for steps in (999, 1000, 20):
        d = run / "checkpoints" / str(steps)
        d.mkdir(parents=True)
        (d / "PPO_POLICY.lt").write_bytes(b"")
    other = tmp_path / "runs" / "b" / "checkpoints" / "99999"
    other.mkdir(parents=True)
    (other / "PPO_POLICY.lt").write_bytes(b"")

    chosen = latest_checkpoint(run)
    assert chosen.parent.name == "1000"      # numerisch: 1000 > 999, nicht lexikografisch
    assert "runs/a" in chosen.as_posix()     # nicht der 99999 aus dem anderen Lauf


# --- Alte Rating-Schlüssel (Review-Befund R9) ---------------------------------------------

def _legacy_file(path):
    """ratings.json im Format von vor Audit M4 (Schlüssel = Step-Zahl), mit dem echten Writer."""
    from eval.ladder import save_ratings
    save_ratings({"5053568": ENV.create_rating(25.0, 8.3), "10093184": ENV.create_rating(29.2, 7.2)}, path)
    return path.read_bytes()


def test_legacy_keys_stay_readable_and_get_the_run_name(tmp_path):
    from eval.ladder import load_ratings
    path = tmp_path / "runs" / "sanity" / "ratings.json"
    before = _legacy_file(path)
    ratings = load_ratings(path, "sanity")
    assert set(ratings) == {"sanity/5053568", "sanity/10093184"}
    assert ratings["sanity/10093184"].mu == pytest.approx(29.2)
    assert set(load_ratings(path)) == {"5053568", "10093184"}      # ohne Laufnamen unverändert
    assert path.read_bytes() == before                              # Datei nie verändert


def test_migration_only_writes_a_copy(tmp_path):
    import json
    from eval.ladder import migrate_file
    src = tmp_path / "ratings.json"
    before = _legacy_file(src)
    dst = tmp_path / "ratings_v2.json"
    assert migrate_file(src, dst, "lucy_1v1") == 2
    assert src.read_bytes() == before
    assert set(json.loads(dst.read_text(encoding="utf-8"))) == {"lucy_1v1/5053568", "lucy_1v1/10093184"}
    with pytest.raises(FileExistsError):
        migrate_file(src, dst, "lucy_1v1")                          # Kopie wird nie überschrieben
    with pytest.raises(ValueError):
        migrate_file(src, src, "lucy_1v1")                          # nie in place


def test_ladder_cli_refuses_to_rewrite_a_legacy_ratings_file(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    run = tmp_path / "runs" / "lucy_1v1"
    for steps in (100, 200):
        (run / "checkpoints" / str(steps)).mkdir(parents=True)
        (run / "checkpoints" / str(steps) / "PPO_POLICY.lt").write_bytes(b"")
    before = _legacy_file(run / "ratings.json")
    ladder = Path(__file__).resolve().parents[1] / "eval" / "ladder.py"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}   # Umlaute in der Meldung
    r = subprocess.run([sys.executable, str(ladder), "--run", str(run), "--games", "1"],
                       capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode != 0
    assert "alte Schlüssel" in r.stderr and "--migrate" in r.stderr
    assert (run / "ratings.json").read_bytes() == before

    out = run / "ratings_v2.json"
    r = subprocess.run([sys.executable, str(ladder), "--migrate", str(run / "ratings.json"), "--out", str(out)],
                       capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, r.stderr
    assert "lucy_1v1/10093184" in r.stdout
    assert (run / "ratings.json").read_bytes() == before


def test_real_legacy_ratings_file_of_the_sanity_run_is_readable():
    """Die echte Datei aus runs/sanity (vor M4 geschrieben), nur gelesen."""
    import hashlib
    from pathlib import Path
    from eval.ladder import has_legacy_keys, load_ratings
    path = Path(__file__).resolve().parents[1] / "runs" / "sanity" / "ratings.json"
    if not path.exists():
        pytest.skip("runs/sanity/ratings.json fehlt (nur auf dem Trainings-PC)")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ratings = load_ratings(path, "sanity")
    assert ratings and all(k.startswith("sanity/") for k in ratings)
    if has_legacy_keys(path):
        assert set(load_ratings(path)) == {k.split("/", 1)[1] for k in ratings}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
