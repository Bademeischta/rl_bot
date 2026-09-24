"""Golden-Paritätstest: Python-Obs muss bitgleich zum C++-Obs des Trainings sein.

Die Fixtures erzeugt `dump_obs.exe` (tools/cpp/dump_obs.cpp). Fehlen sie, werden die Tests
übersprungen statt zu scheitern, damit ein reines Python-Checkout testbar bleibt.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from env.obs_python import BallView, GameView, PlayerView, build_obs, obs_size

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "obs_golden.json"
DUMPER = ROOT / "build" / "cpp_cu128" / "dump_obs.exe"


def _ensure_fixture() -> dict:
    if not FIXTURE.exists():
        if not DUMPER.exists():
            pytest.skip(f"Fixture {FIXTURE.name} fehlt und dump_obs.exe ist nicht gebaut")
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(DUMPER), str(FIXTURE), "30"], check=True, cwd=ROOT)
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden() -> dict:
    return _ensure_fixture()


def _to_view(case: dict) -> GameView:
    players = [
        PlayerView(
            car_id=p["car_id"], team=p["team"],
            pos=np.array(p["pos"]), forward=np.array(p["forward"]), up=np.array(p["up"]),
            vel=np.array(p["vel"]), ang_vel=np.array(p["ang_vel"]),
            boost=p["boost"], on_ground=p["on_ground"], has_flip=p["has_flip"],
            has_jump=p["has_jump"], demoed=p["demoed"], supersonic=p["supersonic"],
            flipping=p["flipping"], jumping=p["jumping"],
        )
        for p in case["players"]
    ]
    return GameView(
        ball=BallView(pos=np.array(case["ball"]["pos"]), vel=np.array(case["ball"]["vel"]),
                      ang_vel=np.array(case["ball"]["ang_vel"])),
        players=players,
        pad_timers=np.array(case["pad_timers"]),
        action_history={p["car_id"]: [np.array(a) for a in p["action_history"]]
                        for p in case["players"]},
    )


def test_obs_size_matches_cpp(golden):
    assert obs_size(golden["max_players"], golden["action_stack_size"]) == golden["obs_size"]


def test_every_case_matches_cpp_bit_for_bit(golden):
    max_players = golden["max_players"]
    stack = golden["action_stack_size"]
    worst = 0.0
    checked = 0

    for case_idx, case in enumerate(golden["cases"]):
        view = _to_view(case)
        for p_idx, p_json in enumerate(case["players"]):
            expected = np.asarray(p_json["obs"], dtype=np.float32)
            actual = build_obs(view, view.players[p_idx], max_players, stack)
            assert actual.shape == expected.shape
            diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            worst = max(worst, float(diff.max()))
            bad = np.flatnonzero(diff > 1e-5)
            assert not len(bad), (
                f"Fall {case_idx}, Spieler {p_json['car_id']}: "
                f"{len(bad)} Abweichungen, erste bei Index {bad[:5].tolist()}, "
                f"C++ {expected[bad[:5]]} vs Python {actual[bad[:5]]}"
            )
            checked += 1

    assert checked >= 30, f"zu wenige Fälle geprüft: {checked}"
    print(f"\n{checked} Obs-Vektoren geprüft, größte Abweichung: {worst:.3e}")


def test_pad_timer_inversion_matches_cpp(golden):
    """Die invertierte Pad-Reihenfolge ist im Python-Code als Umkehrung implementiert."""
    for case in golden["cases"]:
        timers = np.array(case["pad_timers"])
        assert np.array_equal(timers[::-1], np.array(case["pad_timers_inv"]))


def test_all_team_sizes_present(golden):
    sizes = {case["team_size"] for case in golden["cases"]}
    assert sizes == {1, 2, 3}, f"Fixtures decken nicht alle Modi ab: {sizes}"
