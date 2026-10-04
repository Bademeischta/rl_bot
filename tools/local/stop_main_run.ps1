# Stoppt den Hauptlauf sauber (Hauptlauf-Betrieb B4): legt die Stop-Datei an; train_bot.exe beendet die
# laufende Iteration, schreibt mit --save-on-exit den End-Checkpoint und endet. Kein Kill.
#   powershell -ExecutionPolicy Bypass -File tools\local\stop_main_run.ps1 [-NoWait] [-TimeoutSeconds 900]
#
# Wartet (ohne -NoWait) auf das Prozessende und nennt den neuesten Checkpoint. Reagiert der Trainer nicht
# innerhalb von -TimeoutSeconds, meldet das Skript es nur (Exit 1) und beendet NICHTS hart.
# Den Checkpoint-Ordner liest es aus der Config des laufenden Trainers (B8), -CheckpointFolder überschreibt das.
param(
    [string]$StopFile = "runs\hauptlauf\STOP",
    [string]$CheckpointFolder = "runs\lucy_1v1\checkpoints",
    [string]$TrainerName = "train_bot",
    [int]$ProcessId = 0,
    [int]$TimeoutSeconds = 900,
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
$procs = if ($ProcessId) { Get-Process -Id $ProcessId -ErrorAction SilentlyContinue } else { Get-Process -Name $TrainerName -ErrorAction SilentlyContinue }
if (-not $procs) {
    Write-Host "Kein $TrainerName-Prozess gefunden, nichts zu stoppen."
    exit 0
}
# B8: Läuft der Hauptlauf mit einer anderen Config (z. B. runs\lucy_1v1_lr1e4), stand hier sonst der neueste
# Checkpoint des Standardordners. Der Ordner kommt deshalb aus der Config auf der Kommandozeile des Trainers.
if (-not $PSBoundParameters.ContainsKey('CheckpointFolder')) {
    foreach ($pr in $procs) {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($pr.Id)" -ErrorAction SilentlyContinue).CommandLine
        if ($cmd -match '"([^"]+\.json)"|(\S+\.json)') {
            $cfgFile = if ($Matches[1]) { $Matches[1] } else { $Matches[2] }
            if (-not [IO.Path]::IsPathRooted($cfgFile)) { $cfgFile = Join-Path $Root $cfgFile }
            try {
                $folder = (Get-Content $cfgFile -Raw -ErrorAction Stop | ConvertFrom-Json).learner.checkpoint_folder
                if ($folder) { $CheckpointFolder = $folder; break }
            } catch { }
        }
    }
}
$stopPath = if ([IO.Path]::IsPathRooted($StopFile)) { $StopFile } else { Join-Path $Root $StopFile }
New-Item -ItemType Directory -Force (Split-Path $stopPath) | Out-Null
Set-Content -Path $stopPath -Value "stop $(Get-Date -Format o)" -Encoding ASCII
Write-Host "Stop-Datei angelegt: $stopPath (PID $($procs.Id -join ', ')); der Trainer beendet die laufende Iteration und speichert."
if ($NoWait) { exit 0 }

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    $alive = $procs | Where-Object { -not $_.HasExited }
    if (-not $alive) { break }
    Start-Sleep -Seconds 2
    $procs | ForEach-Object { $_.Refresh() }
}
if ($procs | Where-Object { -not $_.HasExited }) {
    Write-Host "Trainer läuft nach $TimeoutSeconds s noch. Nichts beendet; Fenster 'RLbot Hauptlauf' prüfen." -ForegroundColor Yellow
    exit 1
}
$ck = if ([IO.Path]::IsPathRooted($CheckpointFolder)) { $CheckpointFolder } else { Join-Path $Root $CheckpointFolder }
if (Test-Path $ck) {
    $newest = Get-ChildItem $ck -Directory | Where-Object { $_.Name -match '^\d+$' } | Sort-Object { [long]$_.Name } | Select-Object -Last 1
    if ($newest) { Write-Host "Sauber beendet. Neuester Checkpoint: $($newest.Name) ($CheckpointFolder)" }
} else {
    Write-Host "Sauber beendet."
}
exit 0
