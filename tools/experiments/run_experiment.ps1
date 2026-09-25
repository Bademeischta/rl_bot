# Führt ein Stufe-3-Experiment lokal aus (Audit-Roadmap, Schritt 2) und packt die Ergebnisse.
#
#   powershell -ExecutionPolicy Bypass -File tools\experiments\run_experiment.ps1 `
#       -Config train\configs\experiments\baseline.json `
#       -StartCheckpoint runs\lucy_1v1\checkpoints\2704829056 -Steps 100000000 -Seed 123
#
#   Für jedes weitere Experiment dieselben -StartCheckpoint/-Steps/-Seed, dazu
#   -Baseline results\exp_baseline_<datum> für das Duell "Ende gegen Baseline-Ende".
#
# Ablauf:
#   1. freien Speicherplatz prüfen
#   2. Checkpoint-Ordner (inkl. RUNNING_STATS.json) nach runs\exp_<name>_<datum>\checkpoints\ KOPIEREN,
#      das Original bleibt unberührt; ein vorhandener Zielordner wird NIE überschrieben
#   3. Training bis Start-Steps + -Steps (learner.extra_steps), End-Checkpoint per save_on_exit;
#      Abbruchkriterien aus tools\experiments\check_abort.py werden alle -PollSeconds geprüft
#   4. Ladder (TrueSkill, eval\ladder.py) und Duell Ende gegen Start (und gegen Baseline-Ende)
#   5. alles nach results\exp_<name>_<datum>\ (summary.md/json, metrics.csv, Ladder, Duelle, Logs)
#      und als results\exp_<name>_<datum>.zip
#
# Voraussetzungen: gebauter Trainer (bench\cpp\build.ps1 -Flavor cu128), .venv mit requirements,
# collision_meshes\ im Repo-Wurzelordner.
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$StartCheckpoint,
    [long]$Steps = 100000000,
    [int]$Seed = 123,
    [string]$Name = "",
    [string]$Baseline = "",
    [int]$DuelGames = 100,
    [int]$LadderGames = 50,
    [int]$PollSeconds = 30,
    [int]$MinFreeGB = 20,
    [int]$AbortWarmup = 100,
    [string]$Flavor = "cu128",
    [switch]$SkipLadder
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
# Native Programme nur über Invoke-Native (Review-Befund R2: stderr unter PowerShell 5.1)
. "$Root\tools\NativeCommand.ps1"
$Py = "$Root\.venv\Scripts\python.exe"
$Build = "$Root\build\cpp_$Flavor"
$Trainer = "$Build\train_bot.exe"
$Duel = "$Build\duel.exe"

function Fail($msg) { Write-Host "FEHLER: $msg" -ForegroundColor Red; throw $msg }

# --- 0. Eingaben prüfen ---------------------------------------------------------
$Config = (Resolve-Path $Config).Path
$StartCheckpoint = (Resolve-Path $StartCheckpoint).Path
if (-not (Test-Path $Trainer)) { Fail "Trainer fehlt: $Trainer (bench\cpp\build.ps1 -Flavor $Flavor)" }
if (-not (Test-Path $Py)) { Fail ".venv fehlt: $Py (README, Einrichtung)" }
if (-not (Test-Path "$StartCheckpoint\PPO_POLICY.lt")) { Fail "Kein PPO_POLICY.lt in $StartCheckpoint" }
if (-not (Test-Path "$StartCheckpoint\RUNNING_STATS.json")) { Fail "Kein RUNNING_STATS.json in $StartCheckpoint" }
if (-not (Test-Path "$Root\collision_meshes")) { Fail "collision_meshes fehlt im Repo-Wurzelordner" }
$startSteps = [long](Split-Path $StartCheckpoint -Leaf)
if ($Name -eq "") { $Name = [IO.Path]::GetFileNameWithoutExtension($Config) }
$Date = Get-Date -Format "yyyy-MM-dd_HHmm"
$ExpName = "exp_${Name}_${Date}"
$RunDir = "$Root\runs\$ExpName"
$ResDir = "$Root\results\$ExpName"
if (Test-Path $RunDir) { Fail "Lauf-Ordner existiert bereits: $RunDir (wird nie ueberschrieben)" }
if (Test-Path $ResDir) { Fail "Ergebnis-Ordner existiert bereits: $ResDir" }
New-Item -ItemType Directory -Force -Path $ResDir | Out-Null
Start-Transcript -Path "$ResDir\run_experiment.log" | Out-Null
$startedAt = Get-Date

try {
    Write-Host "=== Experiment $ExpName ===" -ForegroundColor Cyan
    Write-Host "Config:           $Config"
    Write-Host "Start-Checkpoint: $StartCheckpoint ($startSteps Steps)"
    Write-Host "Steps:            $Steps   Seed: $Seed   Ziel: $($startSteps + $Steps)"
    $gitHead = Invoke-Native git @('-C', $Root, 'rev-parse', '--short', 'HEAD') -NoThrow -Quiet
    $gitDirty = Invoke-Native git @('-C', $Root, 'status', '--porcelain') -NoThrow -Quiet
    Write-Host "Git:              $gitHead $(if ($gitDirty) { '(dirty)' })"

    # --- 1. Speicherplatz --------------------------------------------------------
    $drive = (Get-Item $Root).PSDrive
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    $ckptMB = [math]::Round((Get-ChildItem $StartCheckpoint -Recurse | Measure-Object Length -Sum).Sum / 1MB, 1)
    $cfgJson = Get-Content $Config -Raw | ConvertFrom-Json
    $saves = [math]::Ceiling($Steps / [double]$cfgJson.learner.timesteps_per_save) + 2
    $needGB = [math]::Round(($ckptMB * $saves) / 1024 + 1, 1)
    Write-Host "Frei auf $($drive.Name): $freeGB GB, Checkpoint $ckptMB MB, geschaetzter Bedarf $needGB GB"
    if ($freeGB -lt $MinFreeGB) { Fail "Weniger als $MinFreeGB GB frei auf $($drive.Name):" }
    if ($freeGB -lt $needGB) { Fail "Geschaetzter Bedarf $needGB GB > frei $freeGB GB" }

    # --- 2. Checkpoint kopieren --------------------------------------------------
    $ckptDst = "$RunDir\checkpoints\$startSteps"
    New-Item -ItemType Directory -Force -Path $ckptDst | Out-Null
    Copy-Item "$StartCheckpoint\*" -Destination $ckptDst -Recurse
    if (-not (Test-Path "$ckptDst\PPO_POLICY.lt")) { Fail "Kopie fehlgeschlagen: $ckptDst" }
    Write-Host "Checkpoint kopiert nach $ckptDst (Original unveraendert)"

    # --- 3. Config ableiten und Training starten ----------------------------------
    $cfgJson.learner.checkpoint_folder = "runs/$ExpName/checkpoints"
    $cfgJson.learner.random_seed = $Seed
    $cfgJson.learner.timestep_limit = 0
    $cfgJson.learner | Add-Member -NotePropertyName extra_steps -NotePropertyValue $Steps -Force
    $cfgJson.learner | Add-Member -NotePropertyName save_on_exit -NotePropertyValue $true -Force
    $cfgJson.metrics.run = $ExpName
    $cfgPath = "$RunDir\config.json"
    $cfgJson | ConvertTo-Json -Depth 10 | Set-Content -Path $cfgPath -Encoding UTF8
    Write-Host "Abgeleitete Config: $cfgPath"

    $pyHome = (Invoke-Native $Py @('-c', 'import sys; print(sys.base_prefix)')).Trim()
    $env:PYTHONHOME = $pyHome
    $env:PATH = "$pyHome;$env:PATH"

    $baselineSps = $null
    if ($Baseline -ne "" -and (Test-Path "$Baseline\summary.json")) {
        $bs = Get-Content "$Baseline\summary.json" -Raw | ConvertFrom-Json
        if ($bs.metrics.last_20pct.sps) { $baselineSps = [double]$bs.metrics.last_20pct.sps }
    }

    Write-Host "Training startet ($(Get-Date -Format 'HH:mm:ss')); Log: $RunDir\train.log"
    $trainStart = Get-Date
    $proc = Start-Process -FilePath $Trainer -ArgumentList "`"$cfgPath`"" -WorkingDirectory $Root `
        -RedirectStandardOutput "$RunDir\train.log" -RedirectStandardError "$RunDir\train.err" `
        -PassThru -NoNewWindow
    $abortReason = $null
    $lastReport = Get-Date
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds $PollSeconds
        $checkArgs = @("$Root\tools\experiments\check_abort.py", "$RunDir\metrics.csv", "--warmup", $AbortWarmup)
        if ($baselineSps) { $checkArgs += @("--baseline-sps", $baselineSps) }
        $out = Invoke-Native $Py $checkArgs -MergeStdErr -NoThrow
        $code = $LASTEXITCODE
        if ($code -eq 3) {
            $abortReason = ($out | Out-String).Trim()
            Write-Host "ABBRUCH durch check_abort.py:`n$abortReason" -ForegroundColor Red
            Stop-Process -Id $proc.Id -Force
            break
        }
        if ($code -eq 4) { Write-Host ($out | Out-String).Trim() -ForegroundColor Yellow }
        if (((Get-Date) - $lastReport).TotalMinutes -ge 5) {
            Write-Host "$(Get-Date -Format 'HH:mm:ss') $(($out | Select-Object -Last 1))"
            $lastReport = Get-Date
        }
    }
    $wall = [math]::Round(((Get-Date) - $trainStart).TotalSeconds, 0)
    if (-not $abortReason -and $proc.ExitCode -ne 0) {
        $abortReason = "train_bot.exe Exit-Code $($proc.ExitCode), siehe $RunDir\train.err"
        Write-Host $abortReason -ForegroundColor Red
    }
    Write-Host "Training beendet nach $wall s"

    # --- 4. Ladder und Duelle --------------------------------------------------------
    $ckpts = Get-ChildItem "$RunDir\checkpoints" -Directory | Where-Object { $_.Name -match '^\d+$' } |
        Sort-Object { [long]$_.Name }
    $endCkpt = $ckpts | Select-Object -Last 1
    Write-Host "End-Checkpoint: $($endCkpt.FullName) ($($ckpts.Count) Checkpoints)"
    if ([long]$endCkpt.Name -eq $startSteps) { Write-Host "WARNUNG: kein neuer Checkpoint entstanden" -ForegroundColor Yellow }

    $duelStart = "$ResDir\duel_end_vs_start.json"
    $duelBase = "$ResDir\duel_end_vs_baseline.json"
    if ((Test-Path $Duel) -and [long]$endCkpt.Name -ne $startSteps) {
        Write-Host "Duell Ende gegen Start ($DuelGames Spiele)..."
        Invoke-Native $Duel @('--a', "$($endCkpt.FullName)\PPO_POLICY.lt", '--b', "$ckptDst\PPO_POLICY.lt", '--games', $DuelGames,
            '--meshes', "$Root\collision_meshes", '--out', $duelStart) -MergeStdErr | Select-Object -Last 2 | ForEach-Object { Write-Host $_ }
        if ($Baseline -ne "" -and (Test-Path "$Baseline\summary.json")) {
            $bs = Get-Content "$Baseline\summary.json" -Raw | ConvertFrom-Json
            if ($bs.end_checkpoint -and (Test-Path "$($bs.end_checkpoint)\PPO_POLICY.lt")) {
                Write-Host "Duell Ende gegen Baseline-Ende ($DuelGames Spiele)..."
                Invoke-Native $Duel @('--a', "$($endCkpt.FullName)\PPO_POLICY.lt", '--b', "$($bs.end_checkpoint)\PPO_POLICY.lt",
                    '--games', $DuelGames, '--meshes', "$Root\collision_meshes", '--out', $duelBase) -MergeStdErr | Select-Object -Last 2 | ForEach-Object { Write-Host $_ }
            } else {
                Write-Host "Baseline-Endcheckpoint nicht gefunden ($($bs.end_checkpoint)), Duell uebersprungen" -ForegroundColor Yellow
            }
        }
        if (-not $SkipLadder) {
            Write-Host "Ladder ($LadderGames Spiele je Paarung)..."
            Invoke-Native $Py @("$Root\eval\ladder.py", '--run', $RunDir, '--games', $LadderGames, '--exe', $Duel) -MergeStdErr | ForEach-Object { Write-Host $_ }
        }
    } else {
        Write-Host "duel.exe fehlt oder kein neuer Checkpoint: Duelle uebersprungen" -ForegroundColor Yellow
    }

    # --- 5. Ergebnisse sammeln ---------------------------------------------------------
    foreach ($f in @("metrics.csv", "config.json", "config_used.json", "train.log", "train.err", "ratings.json")) {
        if (Test-Path "$RunDir\$f") { Copy-Item "$RunDir\$f" -Destination $ResDir }
    }
    $sumArgs = @("$Root\tools\experiments\summarize.py", "--run", $RunDir, "--out", $ResDir, "--name", $Name,
                 "--config", $Config, "--start-checkpoint", $ckptDst, "--end-checkpoint", $endCkpt.FullName,
                 "--wall-seconds", $wall)
    if (Test-Path $duelStart) { $sumArgs += @("--duel-start", $duelStart) }
    if (Test-Path $duelBase) { $sumArgs += @("--duel-baseline", $duelBase) }
    if ($abortReason) { $sumArgs += @("--abort-reason", $abortReason) }
    Invoke-Native $Py $sumArgs -MergeStdErr | ForEach-Object { Write-Host $_ }
}
finally {
    Stop-Transcript | Out-Null
}

$zip = "$Root\results\$ExpName.zip"
Compress-Archive -Path "$ResDir\*" -DestinationPath $zip -Force
Write-Host "`nFertig nach $([math]::Round(((Get-Date) - $startedAt).TotalMinutes, 1)) min." -ForegroundColor Green
Write-Host "Ergebnisse: $ResDir"
Write-Host "Zip zum Zurueckgeben: $zip"
if ($abortReason) { Write-Host "Lauf wurde abgebrochen: $abortReason" -ForegroundColor Red; exit 3 }
exit 0
