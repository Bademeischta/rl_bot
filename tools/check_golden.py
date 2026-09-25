"""Vergleicht einen frischen Obs-Dump mit den Golden-Fixtures, ohne sie zu überschreiben (Audit M3).

    python tools/check_golden.py <neuer_dump.json> [--reference tests/fixtures/obs_golden.json]

Rückgabe 0: Layout unverändert (alle Obs-Vektoren innerhalb der Toleranz, Aktionstabelle
identisch). Rückgabe 1: Abweichung -> das Obs-Layout hat sich geändert, bestehende Checkpoints
sind dann INKOMPATIBEL. Bewusst aktualisieren mit tools/update_golden.ps1.

Der Vergleich ist numerisch (Toleranz 1e-5), nicht per Datei-Hash: Derselbe Code liefert auf
MSVC und GCC Werte, die sich in der 7. Nachkommastelle unterscheiden (in der Cloud-VM gemessen:
größte Abweichung 9,5e-7); ein Hash würde das als Layout-Änderung melden.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "fixtures" / "obs_golden.json"
TOLERANCE = 1e-5


def compare_golden(reference: dict, candidate: dict, tol: float = TOLERANCE) -> list[str]:
    """Liefert eine Liste von Abweichungen; leer = kompatibel."""
    problems: list[str] = []
    for key in ("obs_size", "max_players", "action_stack_size", "player_features", "ball_features"):
        if reference.get(key) != candidate.get(key):
            problems.append(f"{key}: Referenz {reference.get(key)}, neu {candidate.get(key)}")

    ref_table = np.asarray(reference.get("action_table", []), dtype=np.float64)
    new_table = np.asarray(candidate.get("action_table", []), dtype=np.float64)
    if ref_table.shape != new_table.shape:
        problems.append(f"action_table: Form {ref_table.shape} gegen {new_table.shape}")
    elif ref_table.size and not np.array_equal(ref_table, new_table):
        rows = np.flatnonzero(np.abs(ref_table - new_table).max(axis=1) > 0)
        problems.append(f"action_table: {len(rows)} Zeilen weichen ab, erste {rows[:5].tolist()}")

    ref_cases = reference.get("cases", [])
    new_cases = candidate.get("cases", [])
    if len(ref_cases) != len(new_cases):
        problems.append(f"cases: Referenz {len(ref_cases)} Fälle, neu {len(new_cases)}")
    worst = 0.0
    for idx, (rc, nc) in enumerate(zip(ref_cases, new_cases)):
        if len(rc["players"]) != len(nc["players"]):
            problems.append(f"Fall {idx}: Spielerzahl {len(rc['players'])} gegen {len(nc['players'])}")
            continue
        for rp, np_ in zip(rc["players"], nc["players"]):
            a = np.asarray(rp["obs"], dtype=np.float64)
            b = np.asarray(np_["obs"], dtype=np.float64)
            if a.shape != b.shape:
                problems.append(f"Fall {idx}, Spieler {rp['car_id']}: Länge {a.shape} gegen {b.shape}")
                continue
            diff = np.abs(a - b)
            worst = max(worst, float(diff.max()) if diff.size else 0.0)
            bad = np.flatnonzero(diff > tol)
            if len(bad):
                problems.append(
                    f"Fall {idx}, Spieler {rp['car_id']}: {len(bad)} Werte über Toleranz {tol:g}, "
                    f"erste bei Index {bad[:5].tolist()}, Referenz {a[bad[:3]]} neu {b[bad[:3]]}")

    ref_rot = reference.get("rotation_cases", [])
    new_rot = candidate.get("rotation_cases", [])
    if len(ref_rot) != len(new_rot):
        problems.append(f"rotation_cases: {len(ref_rot)} gegen {len(new_rot)}")
    for idx, (rr, nr) in enumerate(zip(ref_rot, new_rot)):
        for key in ("forward", "right", "up"):
            d = float(np.abs(np.asarray(rr[key]) - np.asarray(nr[key])).max())
            if d > tol:
                problems.append(f"rotation_cases[{idx}].{key}: Abweichung {d:.2e}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path, help="frisch erzeugter Dump (dump_obs.exe <datei> 30)")
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--tol", type=float, default=TOLERANCE)
    a = ap.parse_args()

    if not a.reference.exists():
        print(f"Referenz fehlt: {a.reference}")
        return 2
    if not a.candidate.exists():
        print(f"Dump fehlt: {a.candidate}")
        return 2

    reference = json.loads(a.reference.read_text(encoding="utf-8"))
    candidate = json.loads(a.candidate.read_text(encoding="utf-8"))
    problems = compare_golden(reference, candidate, a.tol)
    if problems:
        print("OBS-LAYOUT HAT SICH GEAENDERT - bestehende Checkpoints sind INKOMPATIBEL.")
        for p in problems[:20]:
            print("  -", p)
        if len(problems) > 20:
            print(f"  ... und {len(problems) - 20} weitere")
        print("Wenn das beabsichtigt ist: powershell -File tools\\update_golden.ps1")
        return 1
    n = sum(len(c["players"]) for c in candidate.get("cases", []))
    print(f"Golden-Fixtures unveraendert: {n} Obs-Vektoren, Aktionstabelle und "
          f"{len(candidate.get('rotation_cases', []))} Rotationen innerhalb {a.tol:g}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
