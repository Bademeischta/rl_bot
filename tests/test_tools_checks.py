"""Tests für die Prüfskripte aus dem Audit: Golden-Vergleich (M3) und Versionsabgleich (M2)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "local"))

from check_golden import compare_golden  # noqa: E402
from check_python_versions import compare, exit_code, parse_pins  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "obs_golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --- M3: Golden-Vergleich -----------------------------------------------------

def test_identical_dump_is_compatible(golden):
    assert compare_golden(golden, copy.deepcopy(golden)) == []


def test_compiler_noise_below_tolerance_is_compatible(golden):
    noisy = copy.deepcopy(golden)
    for case in noisy["cases"]:
        for p in case["players"]:
            p["obs"] = [v + 9e-7 for v in p["obs"]]     # Größenordnung MSVC gegen GCC
    assert compare_golden(golden, noisy) == []


def test_changed_value_is_reported(golden):
    changed = copy.deepcopy(golden)
    changed["cases"][3]["players"][0]["obs"][50] += 0.01
    problems = compare_golden(golden, changed)
    assert len(problems) == 1
    assert "Fall 3" in problems[0] and "Index [50]" in problems[0]


def test_changed_layout_size_is_reported(golden):
    changed = copy.deepcopy(golden)
    changed["obs_size"] = 258
    for case in changed["cases"]:
        for p in case["players"]:
            p["obs"] = p["obs"] + [0.0]
    problems = compare_golden(golden, changed)
    assert any("obs_size" in p for p in problems)
    assert any("Länge" in p for p in problems)


def test_changed_action_table_is_reported(golden):
    changed = copy.deepcopy(golden)
    changed["action_table"][0], changed["action_table"][1] = (
        changed["action_table"][1], changed["action_table"][0])
    problems = compare_golden(golden, changed)
    assert any("action_table" in p for p in problems)


# --- M2: Versionsabgleich ------------------------------------------------------

REQ = """
# Kommentar
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.11.0+cu128
numpy==1.26.4
rlgym[rl-sim]==2.0.1
rlbot_flatbuffers==0.19.0   # trailing
pytest>=9
"""


def test_parse_pins_handles_extras_options_and_comments():
    pins = parse_pins(REQ)
    assert pins == {"torch": "2.11.0+cu128", "numpy": "1.26.4", "rlgym": "2.0.1",
                    "rlbot-flatbuffers": "0.19.0"}


def test_compare_reports_ok_mismatch_and_missing():
    installed = {"torch": "2.11.0+cu128", "numpy": "2.0.1", "rlgym": None}
    rows = compare({"torch": "2.11.0+cu128", "numpy": "1.26.4", "rlgym": "2.0.1"},
                   lookup=lambda n: installed.get(n))
    status = {r.package: r.status for r in rows}
    assert status == {"torch": "OK", "numpy": "ABWEICHUNG", "rlgym": "FEHLT"}
    assert exit_code(rows) == 2
    assert exit_code([r for r in rows if r.package != "rlgym"]) == 1
    assert exit_code([r for r in rows if r.package == "torch"]) == 0


def test_repo_requirements_are_fully_pinned():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    pins = parse_pins(text)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        assert "==" in line, f"nicht gepinnt: {line}"
    assert "rlbot" in pins and "rlbot-flatbuffers" in pins, "rlbot fehlt in requirements.txt (Audit M2)"
    assert pins["numpy"] == "1.26.4"
