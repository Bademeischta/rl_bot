# Lokales Prüfpaket (LOCAL_RUNBOOK.md): baut, testet und prüft alles, was in der Cloud-Session
# nicht möglich war, und legt die Ergebnisse als Zip ab.
#
#   powershell -ExecutionPolicy Bypass -File tools\local\run_all_checks.ps1
#   powershell -ExecutionPolicy Bypass -File tools\local\run_all_checks.ps1 -SkipBuild -Repeat 1
#
# Schritte (jeder wird protokolliert; ein Fehler bricht nur den Schritt ab, nicht das Paket):
#   1. Branch-Stand prüfen (erwarteter Branch, Commit, uncommittete Änderungen)
#   2. Upstream-Patches anwenden und cu128 bauen (bench\cpp\build.ps1)
#   3. Alle Tests (tools\run_all_tests.ps1 -Repeat 2)
#   4. Installierte Python-Versionen gegen requirements.txt (tools\local\check_python_versions.py)
#   5. Kurzer Smoke-Trainingslauf mit sanity.json in einen NEUEN Ordner (nie bestehende runs\ anfassen)
#   6. Deployment-Smoke: Export, Laden, Obs-Größe, Inferenz, Latenz (tools\local\deploy_smoke.py)
#      plus Bestandsaufnahme des Hauptlaufs (tools\local\inspect_run.py)
#   7. Alles nach results\local_check_<datum>\ und als Zip
param(
    [string]$Flavor = "cu128",
    [int]$Repeat = 2,
    [string]$Run = "runs\lucy_1v1",
    [string]$ExpectedBranch = "claude/rlbot-audit-roadmap-c8t7nl",
    [long]$SmokeSteps = 2000000,
    [switch]$SkipBuild,
    [switch]$SkipSmoke,
    [switch]$SkipDeploy
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
# Native Programme nur über Invoke-Native (Review-Befund R2: stderr unter PowerShell 5.1)
. "$Root\tools\NativeCommand.ps1"
$Py = "$Root\.venv\Scripts\python.exe"
$Build = "$Root\build\cpp_$Flavor"
$Date = Get-Date -Format "yyyy-MM-dd_HHmm"
$Res = "$Root\results\local_check_$Date"
New-Item -ItemType Directory -Force -Path $Res | Out-Null
Start-Transcript -Path "$Res\run_all_checks.log" | Out-Null
$startedAt = Get-Date
$status = [ordered]@{}

function Step($name, [scriptblock]$body) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $t0 = Get-Date
    try {
        $r = & $body
        if ($null -eq $r) { $r = "OK" }
        $status[$name] = "$r ($([math]::Round(((Get-Date) - $t0).TotalSeconds)) s)"
        Write-Host "-> $($status[$name])" -ForegroundColor Green
    } catch {
        $status[$name] = "FEHLER: $($_.Exception.Message) ($([math]::Round(((Get-Date) - $t0).TotalSeconds)) s)"
        Write-Host "-> $($status[$name])" -ForegroundColor Red
    }
}

try {
    if (-not (Test-Path $Py)) { throw ".venv fehlt: $Py (README, Einrichtung)" }
    $pyHome = (Invoke-Native $Py @('-c', 'import sys; print(sys.base_prefix)')).Trim()
    $env:PYTHONHOME = $pyHome
    $env:PATH = "$pyHome;$env:PATH"

    # --- 1. Branch-Stand -----------------------------------------------------------
    Step "1 Branch-Stand" {
        $branch = (Invoke-Native git @('-C', $Root, 'rev-parse', '--abbrev-ref', 'HEAD')).Trim()
        $commit = (Invoke-Native git @('-C', $Root, 'log', '-1', '--format=%h %ad %s', '--date=short')).Trim()
        $dirty = Invoke-Native git @('-C', $Root, 'status', '--porcelain') -Quiet
        $upstream = Invoke-Native git @('-C', "$Root\third_party\RLGymPPO_CPP", 'rev-parse', '--short', 'HEAD') -NoThrow -Quiet
        @("Branch: $branch", "Commit: $commit", "Uncommittet: $(if ($dirty) { "$(($dirty | Measure-Object).Count) Dateien" } else { 'nichts' })",
          "Upstream: $upstream") |
            Tee-Object -FilePath "$Res\git.txt" | ForEach-Object { Write-Host $_ }
        if ($branch -ne $ExpectedBranch) { throw "Branch ist '$branch', erwartet '$ExpectedBranch' (git checkout $ExpectedBranch && git pull)" }
        if ($dirty) { "OK, aber uncommittete Aenderungen vorhanden" } else { "OK ($commit)" }
    }

    # --- 2. Patches + Build ---------------------------------------------------------
    Step "2 Patches und Build ($Flavor)" {
        Invoke-Native powershell @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$Root\tools\apply_patches.ps1") -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\patches.log" | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE) { throw "apply_patches.ps1 Exit $LASTEXITCODE" }
        if ($SkipBuild) {
            foreach ($exe in @("train_bot.exe", "rlbot_tests.exe", "dump_obs.exe", "duel.exe", "dump_policy_actions.exe")) {
                if (-not (Test-Path "$Build\$exe")) { throw "$exe fehlt in $Build (ohne -SkipBuild bauen)" }
            }
            return "uebersprungen (-SkipBuild), Binaries vorhanden"
        }
        Invoke-Native powershell @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$Root\bench\cpp\build.ps1", '-Flavor', $Flavor) -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\build.log" | Select-Object -Last 3 | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE) { throw "build.ps1 Exit $LASTEXITCODE, siehe $Res\build.log" }
        "OK"
    }

    # --- 3. Tests ---------------------------------------------------------------------
    Step "3 Tests (-Repeat $Repeat)" {
        Invoke-Native powershell @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$Root\tools\run_all_tests.ps1", '-Repeat', $Repeat, '-Flavor', $Flavor) -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\tests.log" | Select-String -Pattern "bestanden|FAIL|fehlgeschlagen|passed|failed|Golden|Python:" | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE) { throw "run_all_tests.ps1 Exit $LASTEXITCODE, siehe $Res\tests.log" }
        "OK"
    }

    # --- 4. Python-Versionen --------------------------------------------------------------
    Step "4 Python-Versionen gegen Pins" {
        Invoke-Native $Py @("$Root\tools\local\check_python_versions.py", '--json', "$Res\python_versions.json") -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\python_versions.txt" | ForEach-Object { Write-Host $_ }
        switch ($LASTEXITCODE) {
            0 { "OK, alle Pins stimmen" }
            1 { "ABWEICHUNG (Information, siehe python_versions.txt)" }
            2 { "FEHLENDE PAKETE (siehe python_versions.txt)" }
            default { throw "check_python_versions.py Exit $LASTEXITCODE" }
        }
    }

    # --- 5. Smoke-Training --------------------------------------------------------------------
    Step "5 Smoke-Training sanity.json ($SmokeSteps Steps)" {
        if ($SkipSmoke) { return "uebersprungen (-SkipSmoke)" }
        $smokeRun = "$Root\runs\local_check_$Date\sanity"
        if (Test-Path $smokeRun) { throw "Ordner existiert schon: $smokeRun" }
        New-Item -ItemType Directory -Force -Path $smokeRun | Out-Null
        $cfg = Get-Content "$Root\train\configs\sanity.json" -Raw | ConvertFrom-Json
        $cfg.learner.checkpoint_folder = "runs/local_check_$Date/sanity/checkpoints"
        $cfg.learner.timestep_limit = $SmokeSteps
        $cfg.learner.timesteps_per_save = [long]($SmokeSteps / 2)
        $cfg.metrics.run = "local_check_sanity"
        $cfgPath = "$smokeRun\config.json"
        $cfg | ConvertTo-Json -Depth 10 | Set-Content -Path $cfgPath -Encoding UTF8
        Invoke-Native "$Build\train_bot.exe" @($cfgPath, '--save-on-exit') -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\smoke_train.log" | Select-String -Pattern "Timestep limit|FATAL|Exception|extra_steps|save_on_exit" | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE) { throw "train_bot.exe Exit $LASTEXITCODE, siehe $Res\smoke_train.log" }
        Copy-Item "$smokeRun\metrics.csv" -Destination "$Res\smoke_metrics.csv"
        Copy-Item "$smokeRun\config_used.json" -Destination "$Res\smoke_config_used.json"
        Invoke-Native $Py @("$Root\tools\experiments\check_abort.py", "$smokeRun\metrics.csv", '--warmup', 5) -MergeStdErr -NoThrow | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -eq 3) { throw "Abbruchkriterium im Smoke-Lauf verletzt (nan/inf?)" }
        $sumOut = Invoke-Native $Py @("$Root\tools\experiments\summarize.py", '--run', $smokeRun, '--out', "$Res\smoke", '--name', 'local_check_sanity') -MergeStdErr
        $sumOut | Select-Object -First 30 | ForEach-Object { Write-Host $_ }
        $ckpts = Get-ChildItem "$smokeRun\checkpoints" -Directory | Measure-Object
        "OK ($($ckpts.Count) Checkpoints, metrics in smoke_metrics.csv)"
    }

    # --- 6. Deployment-Smoke + Bestandsaufnahme ----------------------------------------------------
    Step "6 Deployment-Smoke und Bestandsaufnahme ($Run)" {
        if ($SkipDeploy) { return "uebersprungen (-SkipDeploy)" }
        if (-not (Test-Path "$Root\$Run")) { throw "Lauf fehlt: $Root\$Run" }
        $inspectOut = Invoke-Native $Py @("$Root\tools\local\inspect_run.py", '--run', "$Root\$Run", '--out', "$Res\run_inspect.json") -MergeStdErr
        $inspectOut | Select-Object -First 5 | ForEach-Object { Write-Host $_ }
        Invoke-Native $Py @("$Root\tools\local\deploy_smoke.py", '--run', "$Root\$Run", '--out', "$Res\deploy") -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\deploy_smoke.txt" | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE) { throw "deploy_smoke.py Exit $LASTEXITCODE (siehe deploy_smoke.txt)" }
        # Policy-Paritaet gegen den echten Checkpoint (dump_policy_actions.exe)
        $env:RLBOT_PARITY_RUN = "$Root\$Run"
        Invoke-Native $Py @('-m', 'pytest', "$Root\tests\test_policy_parity.py", '-q', '--no-header', '-p', 'no:cacheprovider') -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\policy_parity.txt" | Select-Object -Last 2 | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE) { throw "Policy-Paritaetstest fehlgeschlagen (policy_parity.txt)" }
        "OK"
    }
}
finally {
    # --- 7. Zusammenfassung + Zip -------------------------------------------------------------------
    $lines = @("# Lokales Pruefpaket $Date", "", "Dauer: $([math]::Round(((Get-Date) - $startedAt).TotalMinutes, 1)) min", "",
               "| Schritt | Ergebnis |", "|---|---|")
    foreach ($k in $status.Keys) { $lines += "| $k | $($status[$k]) |" }
    $lines += @("", "Dateien: git.txt, patches.log, build.log, tests.log, python_versions.txt/json, smoke_metrics.csv,",
                "smoke/summary.md, run_inspect.json, deploy/deploy_smoke.json, policy_parity.txt, run_all_checks.log")
    $lines -join "`n" | Set-Content -Path "$Res\SUMMARY.md" -Encoding UTF8
    Write-Host "`n$($lines -join "`n")"
    Stop-Transcript | Out-Null
    $zip = "$Root\results\local_check_$Date.zip"
    Compress-Archive -Path "$Res\*" -DestinationPath $zip -Force
    Write-Host "`nErgebnisse: $Res" -ForegroundColor Green
    Write-Host "Zip zum Zurueckgeben: $zip" -ForegroundColor Green
    $failed = ($status.Values | Where-Object { $_ -like "FEHLER*" }).Count
    if ($failed) { Write-Host "$failed Schritt(e) mit Fehler, Details im Log." -ForegroundColor Red }
}
