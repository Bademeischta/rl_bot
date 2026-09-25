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


# --- B2 / R2: native Programme unter Windows PowerShell 5.1 ----------------------------------

POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")
needs_ps51 = pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell 5.1 nicht vorhanden")
PS_SCRIPTS = sorted(p for p in ROOT.rglob("*.ps1")
                    if not any(part in (".venv", "third_party", "build", "runs", "results") for part in p.parts))


def _ps(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", *args],
                          cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")


@needs_ps51
def test_invoke_native_ignores_stderr_and_decides_by_exit_code(tmp_path):
    """git/cmake/python schreiben Warnungen nach stderr; das darf unter 'Stop' kein Abbruch sein."""
    import sys
    script = tmp_path / "probe.ps1"
    script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f". '{ROOT / 'tools' / 'NativeCommand.ps1'}'\n"
        f"$py = '{sys.executable}'\n"
        # keine doppelten Anführungszeichen im Argument: 5.1 reicht sie nicht escaped weiter
        "$code = 'import sys; print(''stdout-zeile''); print(''warnung'', file=sys.stderr); sys.exit(int(sys.argv[1]))'\n"
        "$out = Invoke-Native $py @('-c', $code, '0')\n"
        "Write-Output \"RUECKGABE=$out\"\n"
        "$merged = Invoke-Native $py @('-c', $code, '0') -MergeStdErr\n"
        "Write-Output \"MERGED=$($merged -join '|')\"\n"
        "Invoke-Native $py @('-c', $code, '3') -NoThrow -Quiet | Out-Null\n"
        "Write-Output \"NOTHROW=$LASTEXITCODE\"\n"
        "try { Invoke-Native $py @('-c', $code, '3') -Quiet | Out-Null; Write-Output 'KEIN_ABBRUCH' }\n"
        "catch { Write-Output \"ABBRUCH=$($_.Exception.Message -match 'Exit-Code 3')\" }\n",
        encoding="ascii")
    r = _ps("-File", str(script))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RUECKGABE=stdout-zeile" in r.stdout          # stderr nicht im Rückgabewert
    merged = next(line for line in r.stdout.splitlines() if line.startswith("MERGED="))
    assert sorted(merged[len("MERGED="):].split("|")) == ["stdout-zeile", "warnung"]  # -MergeStdErr: als Text dabei
    assert "NOTHROW=3" in r.stdout
    assert "ABBRUCH=True" in r.stdout


@needs_ps51
@needs_git
@pytest.mark.skipif(not (UPSTREAM / ".git").exists(), reason="third_party/RLGymPPO_CPP fehlt")
def test_apply_patches_ps1_runs_under_ps51_on_a_fresh_checkout(tmp_path):
    """Der echte Pfad: apply_patches.ps1 auf einem ungepatchten Klon. Vorher brach 5.1 beim ersten
    'git apply --check --reverse' ab (stderr "patch failed" -> NativeCommandError)."""
    clone = tmp_path / "upstream"
    assert _git("clone", "-q", "--shared", "--no-checkout", str(UPSTREAM), str(clone)).returncode == 0
    assert _git("checkout", "-q", PINNED, cwd=clone).returncode == 0
    script = str(ROOT / "tools" / "apply_patches.ps1")

    r = _ps("-File", script, "-Check", "-Repo", str(clone))
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("[anwendbar, nicht angewendet]") == 2, r.stdout

    r = _ps("-File", script, "-Repo", str(clone))
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("[angewendet]") == 2, r.stdout

    r = _ps("-File", script, "-Repo", str(clone))
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("[bereits angewendet]") == 2, r.stdout


@needs_ps51
def test_no_script_calls_native_programs_directly():
    """Jeder Aufruf von git/cmake/python/*.exe/... läuft über Invoke-Native (AST-Lint unter 5.1)."""
    scripts = [str(p) for p in PS_SCRIPTS if p.name not in ("NativeCommand.ps1", "ps_lint_native_calls.ps1")]
    assert scripts
    r = _ps("-File", str(ROOT / "tests" / "ps_lint_native_calls.ps1"), *scripts)
    assert r.returncode == 0, "direkte native Aufrufe:\n" + r.stdout + r.stderr


# --- B3 / R3: Kodierung der Skripte ---------------------------------------------------------

UTF8_BOM = b"\xef\xbb\xbf"


def test_every_ps1_is_ascii_or_utf8_with_bom():
    """Windows PowerShell 5.1 liest Skripte ohne BOM als ANSI (Windows-1252): Umlaute werden
    verstümmelt, typografische Zeichen wie – oder „ können dort sogar als Anführungszeichen
    gelesen werden und die Syntax brechen."""
    assert PS_SCRIPTS
    bad = []
    for p in PS_SCRIPTS:
        raw = p.read_bytes()
        if raw.startswith(UTF8_BOM):
            raw[3:].decode("utf-8")      # muss gültiges UTF-8 sein
            continue
        try:
            raw.decode("ascii")
        except UnicodeDecodeError:
            bad.append(str(p.relative_to(ROOT)))
    assert bad == [], f"weder ASCII noch UTF-8 mit BOM: {bad}"


@needs_ps51
def test_every_ps1_parses_and_reads_identically_under_ps51(tmp_path):
    """Jedes Skript so lesen wie powershell.exe -File (BOM -> UTF-8, sonst ANSI), parsen und
    prüfen, dass der gelesene Text dem UTF-8-Quelltext entspricht (keine verstümmelten Zeichen)."""
    probe = (
        "$ErrorActionPreference = 'Stop'\n"
        "foreach ($f in $args) {\n"
        "  $text = [System.IO.File]::ReadAllText($f, [System.Text.Encoding]::Default)\n"
        "  $utf8 = [System.IO.File]::ReadAllText($f, (New-Object System.Text.UTF8Encoding $false))\n"
        "  $t = $null; $e = $null\n"
        "  [void][System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$t, [ref]$e)\n"
        "  Write-Output (\"{0}|{1}|{2}\" -f $f, $e.Count, [int]($text -ceq $utf8.TrimStart([char]0xFEFF)))\n"
        "}\n")
    script = tmp_path / "probe.ps1"
    script.write_text(probe, encoding="ascii")
    r = _ps("-File", str(script), *[str(p) for p in PS_SCRIPTS])
    assert r.returncode == 0, r.stderr
    rows = [line.rsplit("|", 2) for line in r.stdout.splitlines() if "|" in line]
    assert len(rows) == len(PS_SCRIPTS), r.stdout + r.stderr
    parse_errors = [f for f, n, _ in rows if n != "0"]
    garbled = [f for f, _, same in rows if same != "1"]
    assert parse_errors == [], f"Parserfehler unter 5.1: {parse_errors}"
    assert garbled == [], f"5.1 liest anderen Text als UTF-8 (fehlendes BOM?): {garbled}"
