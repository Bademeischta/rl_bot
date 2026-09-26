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
#   2. Checkpoint-Ordner (inkl. RUNNING_STATS.json) zweimal KOPIEREN und per Hash prüfen:
#      runs\exp_<name>_<datum>\start\<steps> (Referenz fürs Duell, außerhalb der Checkpoint-Rotation)
#      und ...\checkpoints\<steps> (lädt der Trainer). Das Original bleibt unberührt; vorhandene
#      Lauf-/Ergebnisordner und Zips werden NIE überschrieben
#   3. Training bis Start-Steps + -Steps (learner.extra_steps), End-Checkpoint per save_on_exit;
#      Abbruchkriterien aus tools\experiments\check_abort.py werden alle -PollSeconds geprüft.
#      Abbruch sauber über train_bot.exe --stop-file (Iteration zu Ende, End-Checkpoint schreiben);
#      Stop-Process -Force nur als Notfall nach -StopTimeoutSeconds (Review-Befund R15)
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
    # 1000 Spiele à 300 s: 95-%-KI der Tordifferenz +-0,053 Tore/Spiel bei der gemessenen
    # Streuung (SD 0,85, Nullmessung); ~6 min je Duell mit 8 Threads (AUDIT.md §7.8)
    [int]$DuelGames = 1000,
    [int]$LadderGames = 50,
    [int]$PollSeconds = 30,
    [int]$MinFreeGB = 20,
    [int]$AbortWarmup = 100,
    [string]$Flavor = "cu128",
    [switch]$SkipLadder,
    # So lange wartet der Runner nach dem Anlegen der Stop-Datei, bevor er den Trainer hart beendet
    [int]$StopTimeoutSeconds = 600,
    # Nur Schritte 0-2 (Prüfen, Checkpoint kopieren), kein Training; für Tests
    [switch]$PrepareOnly,
    # Andere Wurzel für runs\ und results\ (Tests); Standard: Repo-Wurzel
    [string]$RunsRoot = "",
    [string]$ResultsRoot = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
# Native Programme nur über Invoke-Native (Review-Befund R2: stderr unter PowerShell 5.1)
. "$Root\tools\NativeCommand.ps1"
# Stop-TrainerGracefully (Review-Befund R15)
. "$Root\tools\experiments\TrainerControl.ps1"
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
$Date = Get-Date -Format "yyyy-MM-dd_HHmmss"
$ExpName = "exp_${Name}_${Date}"
if ($RunsRoot -eq "") { $RunsRoot = "$Root\runs" }
if ($ResultsRoot -eq "") { $ResultsRoot = "$Root\results" }
$RunDir = "$RunsRoot\$ExpName"
$ResDir = "$ResultsRoot\$ExpName"
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
    # Zwei Kopien (Review-Befund R14): start\<steps> ist die unveränderliche Referenz für das Duell
    # "Ende gegen Start" und liegt außerhalb der Checkpoint-Rotation (checkpoints_to_keep löscht in
    # checkpoints\ die ältesten); checkpoints\<steps> lädt der Trainer beim Start.
    $startDst = "$RunDir\start\$startSteps"
    $ckptDst = "$RunDir\checkpoints\$startSteps"
    foreach ($dst in @($startDst, $ckptDst)) {
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        Copy-Item "$StartCheckpoint\*" -Destination $dst -Recurse
        foreach ($f in Get-ChildItem $StartCheckpoint -File) {
            $copy = Join-Path $dst $f.Name
            if (-not (Test-Path $copy) -or (Get-FileHash $copy).Hash -ne (Get-FileHash $f.FullName).Hash) {
                Fail "Kopie fehlerhaft: $copy"
            }
        }
    }
    Write-Host "Checkpoint kopiert und geprueft: $startDst (Referenz, ausserhalb der Rotation) und $ckptDst (Trainer); Original unveraendert"
    if ($PrepareOnly) {
        Write-Host "-PrepareOnly: kein Training" -ForegroundColor Yellow
        return
    }

    # --- 3. Config ableiten und Training starten ----------------------------------
    $cfgJson.learner.checkpoint_folder = ($RunDir -replace '\\', '/') + "/checkpoints"
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
    # --stop-file (R15): run_experiment beendet den Trainer über diese Datei, nicht per Kill
    $stopFile = "$RunDir\STOP"
    $proc = Start-Process -FilePath $Trainer -ArgumentList "`"$cfgPath`" --stop-file `"$stopFile`"" `
        -WorkingDirectory $Root -RedirectStandardOutput "$RunDir\train.log" -RedirectStandardError "$RunDir\train.err" `
        -PassThru -NoNewWindow
    $null = $proc.Handle   # PowerShell 5.1: ohne gecachtes Handle ist ExitCode nach dem Ende leer
    $abortReason = $null
    $stopMode = $null
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
            # Sauber über die Stop-Datei; Stop-Process -Force nur als Notfall nach dem Timeout (R15)
            $stopMode = Stop-TrainerGracefully -Process $proc -StopFile $stopFile -TimeoutSeconds $StopTimeoutSeconds
            if ($stopMode -eq "erzwungen") { $abortReason += " (Trainer musste nach $StopTimeoutSeconds s hart beendet werden)" }
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
    # End-Checkpoint nur ein nachweislich vollständiger (Review-Befund R16): alle Dateien, intakte
    # Archive, RUNNING_STATS.json passend, Policy lädt. Ein halb geschriebener neuester Ordner
    # (Notfall-Kill mitten im Save) wird mit Grund verworfen und der nächstältere genommen.
    $ckpts = Get-ChildItem "$RunDir\checkpoints" -Directory | Where-Object { $_.Name -match '^\d+$' }
    $endPath = Invoke-Native $Py @("$Root\tools\experiments\pick_checkpoint.py", "$RunDir\checkpoints") -NoThrow
    if ($LASTEXITCODE -ne 0 -or -not $endPath) { Fail "Kein vollstaendiger Checkpoint in $RunDir\checkpoints" }
    $endCkpt = Get-Item ("$endPath".Trim())
    Write-Host "End-Checkpoint (vollstaendig geprueft): $($endCkpt.FullName) ($($ckpts.Count) Checkpoint-Ordner)"
    if ([long]$endCkpt.Name -eq $startSteps) { Write-Host "WARNUNG: kein neuer Checkpoint entstanden" -ForegroundColor Yellow }

    $duelStart = "$ResDir\duel_end_vs_start.json"
    $duelBase = "$ResDir\duel_end_vs_baseline.json"
    if ((Test-Path $Duel) -and [long]$endCkpt.Name -ne $startSteps) {
        Write-Host "Duell Ende gegen Start ($DuelGames Spiele)..."
        Invoke-Native $Duel @('--a', "$($endCkpt.FullName)\PPO_POLICY.lt", '--b', "$startDst\PPO_POLICY.lt", '--games', $DuelGames,
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
                 "--config", $Config, "--start-checkpoint", $startDst, "--end-checkpoint", $endCkpt.FullName,
                 "--wall-seconds", $wall)
    if (Test-Path $duelStart) { $sumArgs += @("--duel-start", $duelStart) }
    if (Test-Path $duelBase) { $sumArgs += @("--duel-baseline", $duelBase) }
    if ($abortReason) { $sumArgs += @("--abort-reason", $abortReason) }
    Invoke-Native $Py $sumArgs -MergeStdErr | ForEach-Object { Write-Host $_ }
}
finally {
    Stop-Transcript | Out-Null
}
if ($PrepareOnly) { exit 0 }

$zip = "$ResultsRoot\$ExpName.zip"
if (Test-Path $zip) { Fail "Zip existiert bereits: $zip (wird nie ueberschrieben)" }
Compress-Archive -Path "$ResDir\*" -DestinationPath $zip
Write-Host "`nFertig nach $([math]::Round(((Get-Date) - $startedAt).TotalMinutes, 1)) min." -ForegroundColor Green
Write-Host "Ergebnisse: $ResDir"
Write-Host "Zip zum Zurueckgeben: $zip"
if ($abortReason) { Write-Host "Lauf wurde abgebrochen: $abortReason" -ForegroundColor Red; exit 3 }
exit 0
