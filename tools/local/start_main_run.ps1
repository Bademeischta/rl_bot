# Startet den Hauptlauf so, dass er weiterläuft, wenn die aufrufende Sitzung (z. B. Claude Code) endet
# (Hauptlauf-Betrieb B4). Die Aufgabenplanung startet tools\local\run_main.ps1 in einem eigenen,
# sichtbaren Fenster unter dem angemeldeten Benutzer, ohne Laufzeitgrenze (Standard wären 72 h).
#   powershell -ExecutionPolicy Bypass -File tools\local\start_main_run.ps1 [-Config <json>] [-DryRun]
#
# Vorher: Läuft schon ein train_bot.exe, bricht das Skript ab. Eine alte Stop-Datei (vom letzten
# sauberen Stopp) wird entfernt, sonst würde der Trainer den Start verweigern.
param(
    [string]$Config = "train\configs\lucy_1v1_zero_sum_drill_fast.json",
    [string]$StopFile = "runs\hauptlauf\STOP",
    [string]$TaskName = "RLbot Hauptlauf",
    [string]$TrainerName = "train_bot",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
$cfgPath = Join-Path $Root $Config
if (-not (Test-Path $cfgPath)) { Write-Host "FEHLER: Config fehlt: $cfgPath" -ForegroundColor Red; exit 2 }
if (-not (Test-Path (Join-Path $Root "build\cpp_cu128\train_bot.exe"))) { Write-Host "FEHLER: build\cpp_cu128\train_bot.exe fehlt" -ForegroundColor Red; exit 2 }

$running = Get-Process -Name $TrainerName -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "FEHLER: $TrainerName läuft schon (PID $($running.Id -join ', ')). Erst sauber stoppen: tools\local\stop_main_run.ps1" -ForegroundColor Red
    exit 3
}

$stopPath = if ([IO.Path]::IsPathRooted($StopFile)) { $StopFile } else { Join-Path $Root $StopFile }
if (Test-Path $stopPath) {
    Write-Host "Alte Stop-Datei $StopFile wird entfernt (kein Trainer läuft)."
    if (-not $DryRun) { Remove-Item $stopPath }
}

$runner = Join-Path $Root "tools\local\run_main.ps1"
$argument = "-NoExit -NoProfile -ExecutionPolicy Bypass -File `"$runner`" -Config `"$Config`" -StopFile `"$StopFile`""
Write-Host "Aufgabe '$TaskName': powershell.exe $argument"
Write-Host "Arbeitsordner $Root; Trainer mit --stop-file $StopFile --save-on-exit; Log runs\hauptlauf\train_<datum>.log"
if ($DryRun) { Write-Host "DryRun: nichts gestartet."; exit 0 }

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument -WorkingDirectory $Root
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

$deadline = (Get-Date).AddSeconds(90)
do {
    Start-Sleep -Seconds 2
    $proc = Get-Process -Name $TrainerName -ErrorAction SilentlyContinue
} while (-not $proc -and (Get-Date) -lt $deadline)
if (-not $proc) {
    Write-Host "FEHLER: $TrainerName ist nach 90 s nicht gestartet; Fenster 'RLbot Hauptlauf' und runs\hauptlauf\*.log prüfen" -ForegroundColor Red
    exit 1
}
Write-Host "Hauptlauf läuft: $TrainerName PID $($proc.Id -join ', '), gestartet $($proc[0].StartTime)"
Write-Host "Stand prüfen:    .\.venv\Scripts\python tools\local\main_run_status.py"
Write-Host "Sauber stoppen:  powershell -ExecutionPolicy Bypass -File tools\local\stop_main_run.ps1"
exit 0
