"""Tests für die Build-Voraussetzungen auf dem Windows-PC (Review-Befunde B1-B3).

B1: Die Upstream-Patches müssen im Arbeitsverzeichnis mit LF liegen, sonst scheitert
    `git apply` (core.autocrlf=true macht sonst CRLF daraus).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "third_party" / "patches"
UPSTREAM = ROOT / "third_party" / "RLGymPPO_CPP"
PINNED = "ee4cc56fc8e43758cc898fdba92a9173a94e52f2"
# Reihenfolge wie in tools/apply_patches.ps1
UPSTREAM_PATCHES = ["rlgympppo_cpp_gcc_compat.patch", "rlgympppo_cpp_truncation.patch"]

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git nicht gefunden")


def _git(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


# --- B1: Zeilenenden der Patches ---------------------------------------------------------

def test_all_patches_are_lf_in_the_working_tree():
    patches = sorted(PATCH_DIR.glob("*.patch"))
    assert patches, "keine Patches gefunden"
    crlf = [p.name for p in patches if b"\r\n" in p.read_bytes()]
    assert crlf == [], f"Patches mit CRLF (git apply scheitert daran): {crlf}"


@needs_git
def test_gitattributes_pins_line_endings():
    for path, expected in (("third_party/patches/x.patch", "lf"), ("tools/x.ps1", "crlf")):
        out = _git("check-attr", "eol", "--", path).stdout.strip()
        assert out.endswith(f"eol: {expected}"), out


@needs_git
@pytest.mark.skipif(not (UPSTREAM / ".git").exists(), reason="third_party/RLGymPPO_CPP fehlt")
def test_patches_apply_to_a_fresh_checkout_of_the_pinned_commit(tmp_path):
    """Wie auf dem PC: frischer Checkout mit der systemweiten core.autocrlf-Einstellung."""
    clone = tmp_path / "upstream"
    r = _git("clone", "-q", "--shared", "--no-checkout", str(UPSTREAM), str(clone))
    assert r.returncode == 0, r.stderr
    r = _git("checkout", "-q", PINNED, cwd=clone)
    assert r.returncode == 0, r.stderr
    for name in UPSTREAM_PATCHES:
        patch = str(PATCH_DIR / name)
        r = _git("apply", "--check", patch, cwd=clone)
        assert r.returncode == 0, f"{name}: {r.stderr}"
        r = _git("apply", patch, cwd=clone)
        assert r.returncode == 0, f"{name}: {r.stderr}"
    # Idempotenz-Erkennung von apply_patches.ps1: angewendete Patches lassen sich umkehren
    for name in UPSTREAM_PATCHES:
        r = _git("apply", "--check", "--reverse", str(PATCH_DIR / name), cwd=clone)
        assert r.returncode == 0, f"{name}: {r.stderr}"
