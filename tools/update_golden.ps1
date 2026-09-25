# Erzeugt die Golden-Fixtures BEWUSST neu (Audit M3). run_all_tests.ps1 überschreibt sie nicht
# mehr; es prüft nur noch gegen sie. Dieses Skript ist der einzige Weg, die Referenz zu ändern,
# damit eine Obs-Layout-Änderung (= alle Checkpoints inkompatibel) als git-Diff sichtbar wird.
#   powershell -ExecutionPolicy Bypass -File tools\update_golden.ps1 [-Flavor cu128]
param([string]$Flavor = "cu128")

$ErrorActionPreference = "Stop"
# Native Programme nur über Invoke-Native (Review-Befund R2: stderr unter PowerShell 5.1)
. "$PSScriptRoot\NativeCommand.ps1"
$Root = Resolve-Path "$PSScriptRoot\.."
$Build = "$Root\build\cpp_$Flavor"
$Fixture = "$Root\tests\fixtures\obs_golden.json"

Write-Host "ACHTUNG: Die Referenz $Fixture wird ueberschrieben." -ForegroundColor Yellow
Write-Host "Aendert sich das Layout, sind ALLE bestehenden Checkpoints unbrauchbar (257 Eingaben in fester Bedeutung)."
Write-Host "Vorher pruefen mit: python tools\check_golden.py <dump.json>"

Invoke-Native "$Build\dump_obs.exe" @($Fixture, 30) -MergeStdErr -NoThrow | Select-Object -Last 1 | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) { throw "dump_obs.exe fehlgeschlagen (Exit $LASTEXITCODE)" }

Write-Host "`nGeschrieben. Unterschied zur versionierten Referenz:"
Invoke-Native git @('-C', $Root, 'diff', '--stat', '--', 'tests/fixtures/obs_golden.json') -MergeStdErr | ForEach-Object { Write-Host $_ }
Write-Host "Wenn das beabsichtigt war: Aenderung mit klarer Begruendung committen."
