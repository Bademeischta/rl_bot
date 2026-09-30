"""Schreibt die eigenen Änderungen am Upstream-Klon als gestapelten Patch (Geschwindigkeit, G1).

    python tools/export_upstream_patch.py [--patch rlgympppo_cpp_speed.patch]

Die Patches in third_party/patches/ bauen aufeinander auf (Reihenfolge wie in
tools/apply_patches.ps1). Der letzte Patch ist der Unterschied zwischen "gepinnter Commit plus alle
Patches davor" und dem aktuellen Stand von third_party/RLGymPPO_CPP. So bleiben die älteren
Patches (Truncation, R4) unverändert, auch wenn der neue Patch dieselben Dateien anfasst.

Ablauf: temporärer Klon (--shared) am gepinnten Commit, vorherige Patches anwenden und committen,
die im echten Klon geänderten bzw. neuen Dateien darüberkopieren, `git diff` schreiben (LF).
Der echte Klon wird nur gelesen.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "RLGymPPO_CPP"
PATCH_DIR = ROOT / "third_party" / "patches"
PINNED = "ee4cc56fc8e43758cc898fdba92a9173a94e52f2"


def git(*args: str, cwd: Path, check: bool = True) -> str:
    # Bytes statt text=True: sonst würden CRLF stillschweigend zu LF (universal newlines)
    r = subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, capture_output=True)
    out = r.stdout.decode("utf-8", errors="replace")
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} fehlgeschlagen:\n{out}{r.stderr.decode(errors='replace')}")
    return out


def patch_order() -> list[str]:
    """Patch-Reihenfolge aus tools/apply_patches.ps1 (eine Quelle der Wahrheit)."""
    text = (ROOT / "tools" / "apply_patches.ps1").read_text(encoding="utf-8-sig")
    block = re.search(r"\$Patches = @\((.*?)\)", text, re.S)
    if not block:
        raise SystemExit("Patch-Liste in tools/apply_patches.ps1 nicht gefunden")
    return re.findall(r"patches\\([\w.]+\.patch)", block.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="rlgympppo_cpp_speed.patch",
                    help="Name des Patches, der geschrieben wird (muss in apply_patches.ps1 stehen)")
    a = ap.parse_args()

    order = patch_order()
    if a.patch not in order:
        raise SystemExit(f"{a.patch} steht nicht in der Patch-Liste von apply_patches.ps1: {order}")
    base = order[:order.index(a.patch)]
    later = order[order.index(a.patch) + 1:]
    if later:
        raise SystemExit(f"Nur der letzte Patch im Stapel kann exportiert werden (danach: {later})")

    changed = git("diff", "--name-only", PINNED, cwd=UPSTREAM).split()
    new = git("ls-files", "--others", "--exclude-standard", cwd=UPSTREAM).split()
    files = sorted(set(changed) | set(new))

    tmp = Path(tempfile.mkdtemp(prefix="rlbot_patch_"))
    try:
        clone = tmp / "up"
        git("clone", "-q", "--shared", "--no-checkout", str(UPSTREAM), str(clone), cwd=ROOT)
        git("checkout", "-q", PINNED, cwd=clone)
        for name in base:
            git("apply", str(PATCH_DIR / name), cwd=clone)
        git("-c", "user.name=export", "-c", "user.email=export@localhost",
            "commit", "-q", "--allow-empty", "-am", "base", cwd=clone)
        for f in files:
            src = UPSTREAM / f
            if src.exists():
                (clone / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, clone / f)
        git("add", "-A", cwd=clone)
        diff = git("diff", "--cached", cwd=clone)
        # Nur echte Änderungen: unterschiedliche Zeilenenden würden ganze Dateien umschreiben
        if git("diff", "--cached", "--stat", cwd=clone) != git("diff", "--cached", "--stat", "--ignore-cr-at-eol", cwd=clone):
            raise SystemExit("Zeilenenden im Klon weichen vom Upstream ab; Patch nicht geschrieben")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if "\r" in diff:
        raise SystemExit("Diff enthält CR: Patches müssen reines LF sein (test_all_patches_are_lf_in_the_working_tree)")
    out = PATCH_DIR / a.patch
    out.write_bytes(diff.encode("utf-8"))
    n = diff.count("\ndiff --git") + diff.startswith("diff --git")
    print(f"{out.relative_to(ROOT)}: {n} Dateien, {len(diff.splitlines())} Zeilen (Basis: {', '.join(base)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
