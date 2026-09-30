# Hauptlauf im eigenen Fenster (Hauptlauf-Betrieb B4). Wird von tools\local\start_main_run.ps1 über die
# Aufgabenplanung gestartet und läuft damit unabhängig von der Sitzung, die ihn gestartet hat.
#   powershell -NoExit -ExecutionPolicy Bypass -File tools\local\run_main.ps1 [-Config <json>]
#
# train_bot.exe bekommt --stop-file (sauber stoppen: tools\local\stop_main_run.ps1 legt die Datei an,
# der Trainer beendet die laufende Iteration) und --save-on-exit (End-Checkpoint, kein Fortschritt geht
# verloren). Die Ausgabe erscheint im Fenster und zusätzlich UTF-8 in runs\hauptlauf\train_<datum>.log.
param(
    [string]$Config = "train\configs\lucy_1v1_zero_sum_drill_fast.json",
    [string]$Exe = "build\cpp_cu128\train_bot.exe",
    [string]$StopFile = "runs\hauptlauf\STOP",
    [string]$LogDir = "runs\hauptlauf"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\NativeCommand.ps1"
Set-Location $Root
try { $Host.UI.RawUI.WindowTitle = "RLbot Hauptlauf ($Config)" } catch { }

New-Item -ItemType Directory -Force (Join-Path $Root $LogDir) | Out-Null
$log = Join-Path $Root "$LogDir\train_$(Get-Date -Format yyyy-MM-dd_HHmmss).log"
# Der Trainer bettet Python ein (Metrik-Sender); PYTHONHOME wie in run_experiment.ps1
$pyHome = (Invoke-Native "$Root\.venv\Scripts\python.exe" @('-c', 'import sys; print(sys.base_prefix)')).Trim()
$env:PYTHONHOME = $pyHome
$env:PATH = "$pyHome;$env:PATH"

$writer = New-Object System.IO.StreamWriter($log, $true, (New-Object System.Text.UTF8Encoding($false)))
$writer.AutoFlush = $true
$stopPath = if ([IO.Path]::IsPathRooted($StopFile)) { $StopFile } else { Join-Path $Root $StopFile }
$argList = @((Join-Path $Root $Config), '--stop-file', $stopPath, '--save-on-exit')
$writer.WriteLine("=== $(Get-Date -Format o) Start: $Exe $($argList -join ' ')")
Write-Host "Log: $log"
Write-Host "Sauber stoppen: powershell -ExecutionPolicy Bypass -File tools\local\stop_main_run.ps1"
try {
    Invoke-Native (Join-Path $Root $Exe) $argList -MergeStdErr -NoThrow | ForEach-Object {
        $line = "$_"
        $writer.WriteLine($line)
        $line
    }
    $code = $LASTEXITCODE
} finally {
    $writer.WriteLine("=== $(Get-Date -Format o) train_bot.exe beendet, Exit $code")
    $writer.Close()
}
Write-Host "train_bot.exe beendet (Exit $code). Log: $log"
exit $code
