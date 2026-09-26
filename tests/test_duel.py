"""duel.exe als Messinstrument (Review-Auftrag Schritt 1): Spiele à 300 s, Ergebnis je Spiel,
unabhängige und reproduzierbare Spiele.

Echter Pfad: duel.exe mit dem neuesten echten Checkpoint des Hauptlaufs (nur gelesen) gegen sich
selbst. Übersprungen, wenn duel.exe oder der Checkpoint fehlen.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("RLBOT_BUILD_DIR", str(ROOT / "build" / "cpp_cu128")))
DUEL = BUILD / ("duel.exe" if os.name == "nt" else "duel")
CKPTS = ROOT / "runs" / "lucy_1v1" / "checkpoints"


def _newest_policy() -> Path | None:
    if not CKPTS.exists():
        return None
    c = [p for p in CKPTS.iterdir() if p.name.isdigit() and (p / "PPO_POLICY.lt").exists()]
    return (max(c, key=lambda p: int(p.name)) / "PPO_POLICY.lt") if c else None


POLICY = _newest_policy()
needs_duel = pytest.mark.skipif(not DUEL.exists() or POLICY is None, reason="duel.exe oder Checkpoint fehlt")


def _duel(tmp_path: Path, name: str, *extra: str) -> tuple[subprocess.CompletedProcess, dict | None]:
    out = tmp_path / f"{name}.json"
    r = subprocess.run([str(DUEL), "--a", str(POLICY), "--b", str(POLICY), "--meshes", str(ROOT / "collision_meshes"),
                        "--out", str(out), *extra], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)


@needs_duel
def test_duel_plays_full_300_second_matches_with_results_per_game(tmp_path):
    """Standard: ein Spiel läuft 300 s Spielzeit, auch über Tore hinweg (neuer Anstoß). Vorher
    endete es beim ersten Tor oder nach 120 s, und es gab keine Ergebnisse je Spiel."""
    r, d = _duel(tmp_path, "std", "--games", "4")
    assert r.returncode == 0, r.stdout + r.stderr
    assert d["max_seconds"] == 300
    games = d["per_game"]
    assert len(games) == 4
    assert all(g["game_seconds"] == pytest.approx(300.0) for g in games)
    assert all(g["kickoffs"] == 1 + g["goals_a"] + g["goals_b"] for g in games)   # nach jedem Tor ein Anstoß
    assert sum(g["goals_a"] for g in games) == d["goals_a"] and sum(g["goals_b"] for g in games) == d["goals_b"]
    assert [g["a_blue"] for g in games] == [True, False, True, False]            # Seitentausch
    # Spielpaare (2k, 2k+1) teilen die Anstoß-Seeds, verschiedene Paare nicht
    assert games[0]["first_kickoff_seed"] == games[1]["first_kickoff_seed"]
    assert games[2]["first_kickoff_seed"] == games[3]["first_kickoff_seed"]
    assert games[0]["first_kickoff_seed"] != games[2]["first_kickoff_seed"]


@needs_duel
def test_duel_kickoffs_are_seeded_per_pair_and_games_are_distinct(tmp_path):
    """Anstöße waren zeitgeseedet. Jetzt bestimmt --seed die Anstoß-Folge jedes Spielpaars; ein
    anderer Seed gibt andere Anstöße. Bitgenau gleiche Spiele verspricht duel.exe nicht (RocketSims
    Physik ist adressabhängig, siehe duel.cpp), deshalb wird nur die Seed-Struktur geprüft."""
    _, d1 = _duel(tmp_path, "s1", "--games", "6", "--seed", "7", "--threads", "3")
    _, d2 = _duel(tmp_path, "s2", "--games", "6", "--seed", "7", "--threads", "1")
    _, d3 = _duel(tmp_path, "s3", "--games", "6", "--seed", "8")
    seeds = lambda d: [g["first_kickoff_seed"] for g in d["per_game"]]
    assert seeds(d1) == seeds(d2)                                        # unabhängig von der Threadzahl
    assert seeds(d1) != seeds(d3)
    assert len(set(seeds(d1))) == 3                                      # drei Paare, drei Anstoß-Folgen
    assert d1["threads"] == 3 and d2["threads"] == 1
    assert d1["distinct_games"] == 6                                     # stochastisch: keine Wiederholungen


@needs_duel
def test_duel_refuses_more_deterministic_games_than_distinct_starts(tmp_path):
    """Deterministische Policies + Anstoß: nur 5 Anstoßpositionen x 2 Seiten verschiedene Starts."""
    r, d = _duel(tmp_path, "det", "--games", "12", "--deterministic")
    assert r.returncode == 2 and d is None
    assert "Wiederholungen" in r.stderr
    r, d = _duel(tmp_path, "det_ok", "--games", "4", "--deterministic", "--max-seconds", "30")
    assert r.returncode == 0 and d["deterministic"] is True and "distinct_games" in d


@needs_duel
def test_ladder_duels_use_300_second_games(tmp_path):
    sys.path.insert(0, str(ROOT))
    from eval.ladder import run_duel
    result = run_duel(POLICY, POLICY, 2, exe=DUEL)
    assert result.raw["max_seconds"] == 300
    assert len(result.raw["per_game"]) == 2
