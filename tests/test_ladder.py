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
