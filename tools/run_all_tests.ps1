# Führt die komplette Testsuite aus: C++-Unit-Tests, Golden-Fixtures prüfen, Python-Tests.
#   powershell -ExecutionPolicy Bypass -File tools\run_all_tests.ps1
#   powershell -ExecutionPolicy Bypass -File tools\run_all_tests.ps1 -Repeat 2
#
# Audit M3: Die Golden-Fixtures werden NICHT mehr überschrieben. Ein frischer Dump wird in eine
# temporäre Datei geschrieben und numerisch mit tests/fixtures/obs_golden.json verglichen.
# Weicht er ab, hat sich das Obs-Layout geändert und alle Checkpoints wären inkompatibel.
# Bewusst aktualisieren: tools\update_golden.ps1.
#
# Audit M2: Die Python-Suite muss mindestens -MinPythonTests Tests wirklich ausführen. Fehlende
# Pakete (rlbot, rlbot_flatbuffers, rlgym) würden sonst über pytest.importorskip still
# übersprungen und die Suite bliebe grün.
param(
    [int]$Repeat = 1,
    [string]$Flavor = "cu128",
    [int]$MinPythonTests = 60
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\.."
$Build = "$Root\build\cpp_$Flavor"
$Py = (& "$Root\.venv\Scripts\python.exe" -c "import sys; print(sys.base_prefix)").Trim()
$env:PYTHONHOME = $Py
$env:PATH = "$Py;$env:PATH"

$failed = 0
for ($run = 1; $run -le $Repeat; $run++) {
    Write-Host "`n=== Durchlauf $run von $Repeat ===" -ForegroundColor Cyan

    Write-Host "`n--- C++-Unit-Tests ---"
    & "$Build\rlbot_tests.exe" "$Root\collision_meshes" | Select-String -Pattern "FAIL|bestanden"
    if ($LASTEXITCODE -ne 0) { $failed++; Write-Host "C++-Tests fehlgeschlagen" -ForegroundColor Red }

    Write-Host "`n--- Golden-Fixtures pruefen (werden nicht ueberschrieben) ---"
    $tmpDump = Join-Path $env:TEMP "rlbot_obs_check_$run.json"
    & "$Build\dump_obs.exe" $tmpDump 30 | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0) {
        $failed++; Write-Host "dump_obs fehlgeschlagen" -ForegroundColor Red
    } else {
        & "$Root\.venv\Scripts\python.exe" "$Root\tools\check_golden.py" $tmpDump
        if ($LASTEXITCODE -ne 0) {
            $failed++
            Write-Host "Obs-Layout hat sich geaendert - bestehende Checkpoints sind INKOMPATIBEL." -ForegroundColor Red
            Write-Host "Wenn das beabsichtigt ist: tools\update_golden.ps1 ausfuehren." -ForegroundColor Yellow
        }
    }

    Write-Host "`n--- Python-Tests ---"
    $pyOut = & "$Root\.venv\Scripts\python.exe" -m pytest "$Root\tests" -q --no-header -p no:cacheprovider 2>&1
    $pyOut | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { $failed++; Write-Host "Python-Tests fehlgeschlagen" -ForegroundColor Red }

    $summary = ($pyOut | Select-String -Pattern "(\d+) passed") | Select-Object -Last 1
    $passed = 0
    if ($summary) { $passed = [int]$summary.Matches[0].Groups[1].Value }
    $skippedMatch = ($pyOut | Select-String -Pattern "(\d+) skipped") | Select-Object -Last 1
    $skipped = 0
    if ($skippedMatch) { $skipped = [int]$skippedMatch.Matches[0].Groups[1].Value }
    Write-Host "Python: $passed bestanden, $skipped uebersprungen (Minimum $MinPythonTests bestanden)"
    if ($passed -lt $MinPythonTests) {
        $failed++
        Write-Host "Zu wenige Python-Tests ausgefuehrt ($passed < $MinPythonTests). Fehlen rlbot / rlbot_flatbuffers / rlgym im .venv? (pytest.importorskip ueberspringt still)" -ForegroundColor Red
    }
}

Write-Host ""
if ($failed -eq 0) {
    Write-Host "Alle Durchläufe bestanden." -ForegroundColor Green
    exit 0
}
Write-Host "$failed Teilläufe fehlgeschlagen." -ForegroundColor Red
exit 1
