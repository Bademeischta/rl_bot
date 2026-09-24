# Führt die komplette Testsuite aus: C++-Unit-Tests, Golden-Fixtures neu erzeugen, Python-Tests.
#   powershell -ExecutionPolicy Bypass -File tools\run_all_tests.ps1
#   powershell -ExecutionPolicy Bypass -File tools\run_all_tests.ps1 -Repeat 2
param([int]$Repeat = 1, [string]$Flavor = "cu128")

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

    Write-Host "`n--- Golden-Fixtures neu erzeugen ---"
    & "$Build\dump_obs.exe" "$Root\tests\fixtures\obs_golden.json" 30 | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0) { $failed++; Write-Host "dump_obs fehlgeschlagen" -ForegroundColor Red }

    Write-Host "`n--- Python-Tests ---"
    & "$Root\.venv\Scripts\python.exe" -m pytest "$Root\tests" -q --no-header
    if ($LASTEXITCODE -ne 0) { $failed++; Write-Host "Python-Tests fehlgeschlagen" -ForegroundColor Red }
}

Write-Host ""
if ($failed -eq 0) {
    Write-Host "Alle Durchläufe bestanden." -ForegroundColor Green
    exit 0
}
Write-Host "$failed Teilläufe fehlgeschlagen." -ForegroundColor Red
exit 1
