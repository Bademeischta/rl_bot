"""Tests für die Build-Voraussetzungen auf dem Windows-PC (Review-Befunde B1-B3).

B1: Die Upstream-Patches müssen im Arbeitsverzeichnis mit LF liegen, sonst scheitert
    `git apply` (core.autocrlf=true macht sonst CRLF daraus).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
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
        "$leer = 'import sys; sys.stderr.write(''a'' + chr(10) + chr(10) + ''b'' + chr(10))'\n"
        "$lines = Invoke-Native $py @('-c', $leer) -MergeStdErr\n"
        "Write-Output \"LEER=$($lines -join '|')\"\n"
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
    assert "LEER=a||b" in r.stdout                        # leere stderr-Zeile bleibt leer
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


# --- R4: Upstream-Klon mit der ersten Fassung des Truncation-Patches --------------------------

FIRST_TRUNCATION_PATCH_COMMIT = "c1e6359"   # K1b, Bootstrap von der Reset-Obs (Review R4)


@needs_ps51
@needs_git
@pytest.mark.skipif(not (UPSTREAM / ".git").exists(), reason="third_party/RLGymPPO_CPP fehlt")
def test_apply_patches_reset_replaces_the_first_truncation_patch(tmp_path):
    """Auf einem Klon mit der alten Patch-Fassung meldet apply_patches.ps1 einen Fehler statt
    stillschweigend weiterzubauen; mit -Reset wird die neue Fassung sauber angewendet."""
    old = _git("show", f"{FIRST_TRUNCATION_PATCH_COMMIT}:third_party/patches/rlgympppo_cpp_truncation.patch")
    assert old.returncode == 0, old.stderr
    old_patch = tmp_path / "old_truncation.patch"
    old_patch.write_bytes(old.stdout.encode("utf-8"))
    clone = tmp_path / "upstream"
    assert _git("clone", "-q", "--shared", "--no-checkout", str(UPSTREAM), str(clone)).returncode == 0
    assert _git("checkout", "-q", PINNED, cwd=clone).returncode == 0
    for patch in (PATCH_DIR / "rlgympppo_cpp_gcc_compat.patch", old_patch):
        r = _git("apply", str(patch), cwd=clone)
        assert r.returncode == 0, r.stderr

    script = str(ROOT / "tools" / "apply_patches.ps1")
    r = _ps("-File", script, "-Repo", str(clone))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "[FEHLER] rlgympppo_cpp_truncation.patch" in r.stdout
    assert "-Reset" in r.stdout

    r = _ps("-File", script, "-Reset", "-Repo", str(clone))
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("[angewendet]") == 2, r.stdout
    gym_h = (clone / "RLGymPPO_CPP" / "RLGymSim_CPP" / "src" / "RLGymSim_CPP" / "Gym.h").read_text(encoding="utf-8")
    assert "RLGSC_HAS_FINAL_OBS" in gym_h


# --- R13: bench_expbuffer.ps1 mit eigenem Seed je Wiederholung --------------------------------

@needs_ps51
def test_bench_expbuffer_uses_a_different_seed_per_repetition():
    """Echter Plan des Skripts (-DryRun): Innerhalb einer Wiederholung derselbe Seed für alle
    Varianten (gepaart), zwischen Wiederholungen verschiedene Seeds; Baseline je Wiederholung."""
    r = _ps("-File", str(ROOT / "tools" / "experiments" / "bench_expbuffer.ps1"),
            "-StartCheckpoint", str(ROOT), "-Repeats", "3", "-Seed", "123", "-DryRun")
    assert r.returncode == 0, r.stdout + r.stderr
    plan = [dict(kv.split("=", 1) for kv in line.split()[1:]) for line in r.stdout.splitlines()
            if line.startswith("PLAN ")]
    assert len(plan) == 9
    seeds_per_rep = {}
    for p in plan:
        seeds_per_rep.setdefault(p["rep"], set()).add(p["seed"])
    assert all(len(s) == 1 for s in seeds_per_rep.values()), seeds_per_rep      # gepaart
    assert len({next(iter(s)) for s in seeds_per_rep.values()}) == 3             # verschieden
    assert [next(iter(seeds_per_rep[k])) for k in ("1", "2", "3")] == ["123", "124", "125"]
    for p in plan:
        expected = "selbst" if p["variant"] == "h5_updates6_epochs2_buf3" else f"h5_updates6_epochs2_buf3_r{p['rep']}"
        assert p["baseline"] == expected


# --- R14: Start-Checkpoint außerhalb der Rotation ------------------------------------------

MAIN_RUN_CKPTS = ROOT / "runs" / "lucy_1v1" / "checkpoints"


def _newest_real_checkpoint():
    if not MAIN_RUN_CKPTS.exists():
        return None
    c = [p for p in MAIN_RUN_CKPTS.iterdir() if p.name.isdigit() and (p / "PPO_POLICY.lt").exists()]
    return max(c, key=lambda p: int(p.name)) if c else None


@needs_ps51
@pytest.mark.skipif(_newest_real_checkpoint() is None, reason="kein echter Checkpoint in runs/lucy_1v1")
@pytest.mark.skipif(not (ROOT / "build" / "cpp_cu128" / "train_bot.exe").exists(), reason="train_bot.exe fehlt")
def test_run_experiment_keeps_the_start_checkpoint_outside_the_rotation(tmp_path):
    """Echtes run_experiment.ps1 (-PrepareOnly) mit dem echten neuesten Checkpoint (nur gelesen):
    start/<steps> (Referenz fürs Duell) und checkpoints/<steps> (Trainer), beide byte-gleich."""
    import hashlib
    src = _newest_real_checkpoint()
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in src.iterdir() if f.is_file()}
    runs, results = tmp_path / "runs", tmp_path / "results"
    r = _ps("-File", str(ROOT / "tools" / "experiments" / "run_experiment.ps1"),
            "-Config", str(ROOT / "train" / "configs" / "experiments" / "baseline.json"),
            "-StartCheckpoint", str(src), "-Name", "pytest", "-MinFreeGB", "0",
            "-RunsRoot", str(runs), "-ResultsRoot", str(results), "-PrepareOnly")
    assert r.returncode == 0, r.stdout + r.stderr
    run_dirs = list(runs.glob("exp_pytest_*"))
    assert len(run_dirs) == 1
    for sub in ("start", "checkpoints"):
        copy = run_dirs[0] / sub / src.name
        got = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in copy.iterdir() if f.is_file()}
        assert got == before, sub
    # Original unverändert
    assert {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in src.iterdir() if f.is_file()} == before
    assert "start" in r.stdout and "Rotation" in r.stdout


# --- R18: run_all_checks.ps1 ---------------------------------------------------------------

def _read_log(path: Path) -> str:
    """Tee-Object unter PowerShell 5.1 schreibt UTF-16 (BOM FF FE), Set-Content -Encoding UTF8 mit BOM."""
    raw = path.read_bytes()
    return raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")


@needs_ps51
def test_run_all_checks_skips_dependent_steps_and_names_results_with_seconds(tmp_path):
    """Echtes Skript mit einem Build-Fehler (-Flavor ohne Binaries): Tests, Smoke und
    Deployment-Smoke dürfen dann nicht laufen (keine Tests auf alten Binaries); Git-Stand gibt
    Branch und Hash aus statt einen festen Branch zu verlangen; Ergebnisordner mit Sekunden."""
    import re
    r = _ps("-File", str(ROOT / "tools" / "local" / "run_all_checks.ps1"), "-Flavor", "nichtda", "-SkipBuild",
            "-Repeat", "1", "-ResultsRoot", str(tmp_path))
    assert r.returncode == 1, r.stdout + r.stderr
    dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(dirs) == 1 and re.fullmatch(r"local_check_\d{4}-\d{2}-\d{2}_\d{6}", dirs[0].name), dirs
    assert (tmp_path / f"{dirs[0].name}.zip").exists()
    res = dirs[0]
    git = _read_log(res / "git.txt")
    assert "Branch: " in git and re.search(r"Hash: [0-9a-f]{40}", git)
    rows = {}
    for line in _read_log(res / "SUMMARY.md").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 3 and cells[1][:1].isdigit():
            rows[cells[1].split()[0]] = cells[2]
    if "Uncommittet: nichts" in git:
        assert rows["1"].startswith("OK"), rows
        assert rows["2"].startswith("FEHLER") and "fehlt" in rows["2"], rows
    else:
        assert rows["1"].startswith("FEHLER") and "nicht sauber" in rows["1"], rows
        assert rows["2"].startswith("uebersprungen wegen Schritt 1"), rows
    for step in ("3", "5", "6"):
        assert rows[step].startswith("uebersprungen wegen Schritt"), (step, rows)
    assert rows["4"].startswith("OK"), rows
    assert not (res / "tests.log").exists()          # kein Testlauf auf alten/fehlenden Binaries


def test_results_folder_is_ignored_by_git():
    """run_all_checks verlangt ein sauberes Arbeitsverzeichnis; results/ darf es nicht verschmutzen."""
    r = _git("check-ignore", "-q", "results/local_check_x/SUMMARY.md")
    assert r.returncode == 0


@needs_ps51
@pytest.mark.skipif(sys.platform != "win32", reason="exklusive Dateisperre nur unter Windows")
def test_run_all_checks_zip_survives_a_briefly_locked_file(tmp_path):
    """Ein Virenscanner oder ein offenes Log kann eine Datei kurz exklusiv sperren. Das Zip soll es
    dann erneut versuchen statt das Paket abzubrechen (beim ersten echten Lauf so passiert)."""
    import ctypes
    import threading
    import time
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    held = {}

    def locker():
        deadline = time.time() + 120
        while time.time() < deadline:
            dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
            if dirs:
                break
            time.sleep(0.05)
        res = dirs[0]
        lock = res / "gesperrt.txt"
        lock.write_text("x", encoding="ascii")
        # GENERIC_READ, kein Teilen (share mode 0), OPEN_EXISTING
        h = kernel32.CreateFileW(str(lock), 0x80000000, 0, None, 3, 0x80, None)
        held["ok"] = h not in (None, wintypes.HANDLE(-1).value)
        while time.time() < deadline and not (res / "SUMMARY.md").exists():
            time.sleep(0.05)
        time.sleep(3)                                  # erster Zip-Versuch trifft die Sperre
        kernel32.CloseHandle(h)

    t = threading.Thread(target=locker)
    t.start()
    r = _ps("-File", str(ROOT / "tools" / "local" / "run_all_checks.ps1"), "-Flavor", "nichtda", "-SkipBuild",
            "-Repeat", "1", "-ResultsRoot", str(tmp_path))
    t.join()
    assert held.get("ok"), "Sperre konnte nicht gesetzt werden"
    res = next(d for d in tmp_path.iterdir() if d.is_dir())
    zip_path = tmp_path / f"{res.name}.zip"
    assert zip_path.exists(), r.stdout + r.stderr
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        assert "gesperrt.txt" in z.namelist() and "SUMMARY.md" in z.namelist()
    assert "Zip konnte nicht" not in r.stdout
