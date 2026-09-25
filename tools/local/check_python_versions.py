"""Vergleicht die installierten Python-Pakete mit den Pins in requirements.txt (Audit M2).

    python tools/local/check_python_versions.py [--requirements requirements.txt] [--json out.json]

Rückgabe 0: alles stimmt. 1: mindestens eine Abweichung. 2: mindestens ein Paket fehlt.
Die Pins wurden ohne Zugriff auf den Trainings-PC aus dem Audit übernommen; eine Abweichung
ist deshalb erst einmal eine Information, kein Fehler - sie gehört ins Ergebnis-Zip.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Pakete, die nicht per Pin in requirements.txt stehen können (git-Installation), aber
# trotzdem eine erwartete Version haben.
EXTRA_EXPECTED = {"rlgym-ppo": "1.3.13"}

_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(text: str) -> dict[str, str]:
    """Liest `name==version`-Zeilen (Extras erlaubt); Kommentare und Optionszeilen werden ignoriert."""
    pins: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = _PIN.match(line)
        if m:
            pins[normalize(m.group(1))] = m.group(2)
    return pins


def installed_version(name: str) -> str | None:
    for candidate in {name, name.replace("-", "_"), name.replace("-", ".")}:
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


@dataclass
class Row:
    package: str
    expected: str
    installed: str | None
    status: str   # OK | ABWEICHUNG | FEHLT


def compare(expected: dict[str, str], lookup=installed_version) -> list[Row]:
    rows = []
    for name in sorted(expected):
        have = lookup(name)
        if have is None:
            status = "FEHLT"
        elif have == expected[name]:
            status = "OK"
        else:
            status = "ABWEICHUNG"
        rows.append(Row(name, expected[name], have, status))
    return rows


def exit_code(rows: list[Row]) -> int:
    if any(r.status == "FEHLT" for r in rows):
        return 2
    if any(r.status == "ABWEICHUNG" for r in rows):
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", type=Path, default=ROOT / "requirements.txt")
    ap.add_argument("--json", type=Path, help="Ergebnis zusätzlich als JSON schreiben")
    a = ap.parse_args()

    expected = parse_pins(a.requirements.read_text(encoding="utf-8"))
    expected.update({normalize(k): v for k, v in EXTRA_EXPECTED.items()})
    rows = compare(expected)

    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    print(f"{'Paket':<22}{'Pin':<18}{'installiert':<18}Status")
    for r in rows:
        print(f"{r.package:<22}{r.expected:<18}{(r.installed or '-'):<18}{r.status}")
    code = exit_code(rows)
    n_bad = sum(r.status != "OK" for r in rows)
    print(f"\n{len(rows) - n_bad} von {len(rows)} Paketen stimmen mit den Pins ueberein.")
    if code:
        print("Abweichungen sind Information, kein Abbruchgrund - bitte mit ins Ergebnis-Zip.")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"python": sys.version, "rows": [asdict(r) for r in rows]},
                                     indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
