# Lokales Prüfpaket (LOCAL_RUNBOOK.md): baut, testet und prüft alles, was in der Cloud-Session
# nicht möglich war, und legt die Ergebnisse als Zip ab.
#
#   powershell -ExecutionPolicy Bypass -File tools\local\run_all_checks.ps1
#   powershell -ExecutionPolicy Bypass -File tools\local\run_all_checks.ps1 -SkipBuild -Repeat 1
#
# Schritte (jeder wird protokolliert; ein Fehler bricht den Schritt ab und überspringt alle
# Schritte, die von ihm abhängen, Review-Befund R18):
#   1. Git-Stand: Arbeitsverzeichnis sauber? Branch, Commit-Hash, Upstream-Hash ausgeben
#   2. Upstream-Patches anwenden und cu128 bauen (bench\cpp\build.ps1)            [braucht 1]
#      mit -SkipBuild: Binaries müssen existieren und jünger als der HEAD-Commit sein
#   3. Alle Tests (tools\run_all_tests.ps1 -Repeat 2)                              [braucht 2]
#   4. Installierte Python-Versionen gegen requirements.txt                        [unabhängig]
#   5. Kurzer Smoke-Trainingslauf mit sanity.json in einen NEUEN Ordner            [braucht 2]
#   6. Deployment-Smoke, Policy-Parität, Bestandsaufnahme des Hauptlaufs           [braucht 2]
#   7. Alles nach results\local_check_<datum_uhrzeit>\ und als Zip (nie überschrieben)
# Exit 0 nur, wenn kein Schritt FEHLER oder "übersprungen wegen" meldet.
param(
    [string]$Flavor = "cu128",
    [int]$Repeat = 2,
    [string]$Run = "runs\lucy_1v1",
    [long]$SmokeSteps = 2000000,
    [switch]$SkipBuild,
    [switch]$SkipSmoke,
    [switch]$SkipDeploy,
    # Andere Wurzel für results\ (Tests); Standard: Repo-Wurzel
    [string]$ResultsRoot = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
# Native Programme nur über Invoke-Native (Review-Befund R2: stderr unter PowerShell 5.1)
. "$Root\tools\NativeCommand.ps1"
$Py = "$Root\.venv\Scripts\python.exe"
$Build = "$Root\build\cpp_$Flavor"
# Sekunden im Namen, vorhandene Ordner/Zips werden nie überschrieben (R18)
$Date = Get-Date -Format "yyyy-MM-dd_HHmmss"
if ($ResultsRoot -eq "") { $ResultsRoot = "$Root\results" }
$Res = "$ResultsRoot\local_check_$Date"
$zip = "$ResultsRoot\local_check_$Date.zip"
if ((Test-Path $Res) -or (Test-Path $zip)) { throw "Ergebnis existiert schon: $Res (wird nie ueberschrieben)" }
New-Item -ItemType Directory -Path $Res | Out-Null
Start-Transcript -Path "$Res\run_all_checks.log" | Out-Null
$startedAt = Get-Date
$status = [ordered]@{}
$ok = @{}

# Führt einen Schritt aus, wenn alle Schritte in $DependsOn erfolgreich waren; sonst wird er mit
# Grund übersprungen (keine Tests auf alten Binaries, kein Smoke ohne frischen Build).
function Step($id, $name, [string[]]$DependsOn, [scriptblock]$body) {
    $label = "$id $name"
    Write-Host "`n=== $label ===" -ForegroundColor Cyan
    $missing = @($DependsOn | Where-Object { -not $ok[$_] })
    if ($missing.Count) {
        $status[$label] = "uebersprungen wegen Schritt $($missing -join ', ')"
        Write-Host "-> $($status[$label])" -ForegroundColor Yellow
        $ok[$id] = $false
        return
    }
    $t0 = Get-Date
    try {
        $r = & $body
        if ($null -eq $r) { $r = "OK" }
        $status[$label] = "$r ($([math]::Round(((Get-Date) - $t0).TotalSeconds)) s)"
        $ok[$id] = $true
        Write-Host "-> $($status[$label])" -ForegroundColor Green
    } catch {
        $status[$label] = "FEHLER: $($_.Exception.Message) ($([math]::Round(((Get-Date) - $t0).TotalSeconds)) s)"
        $ok[$id] = $false
        Write-Host "-> $($status[$label])" -ForegroundColor Red
    }
}

try {
    if (-not (Test-Path $Py)) { throw ".venv fehlt: $Py (README, Einrichtung)" }
    $pyHome = (Invoke-Native $Py @('-c', 'import sys; print(sys.base_prefix)')).Trim()
    $env:PYTHONHOME = $pyHome
    $env:PATH = "$pyHome;$env:PATH"

    # --- 1. Git-Stand ------------------------------------------------------------------
    # Keine feste Branch-Erwartung mehr (R18): Geprüft wird, dass das Ergebnis zu einem Commit
    # gehört (sauberes Arbeitsverzeichnis); Branch und Hash stehen in git.txt.
    Step "1" "Git-Stand" @() {
        $branch = (Invoke-Native git @('-C', $Root, 'rev-parse', '--abbrev-ref', 'HEAD')).Trim()
        $hash = (Invoke-Native git @('-C', $Root, 'rev-parse', 'HEAD')).Trim()
        $commit = (Invoke-Native git @('-C', $Root, 'log', '-1', '--format=%h %ad %s', '--date=short')).Trim()
        $dirty = @(Invoke-Native git @('-C', $Root, 'status', '--porcelain') -Quiet | Where-Object { $_ })
        $upstream = Invoke-Native git @('-C', "$Root\third_party\RLGymPPO_CPP", 'rev-parse', 'HEAD') -NoThrow -Quiet
        $lines = @("Branch: $branch", "Hash: $hash", "Commit: $commit",
                   "Uncommittet: $(if ($dirty.Count) { "$($dirty.Count) Dateien" } else { 'nichts' })",
                   "Upstream: $upstream")
        $lines += $dirty | ForEach-Object { "  $_" }
        $lines | Tee-Object -FilePath "$Res\git.txt" | ForEach-Object { Write-Host $_ }
        if ($dirty.Count) { throw "Arbeitsverzeichnis nicht sauber ($($dirty.Count) Dateien, siehe git.txt): erst committen" }
        "OK ($branch @ $($hash.Substring(0, 12)))"
    }

    # --- 2. Patches + Build ---------------------------------------------------------
    Step "2" "Patches und Build ($Flavor)" @("1") {
        Invoke-Native powershell @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$Root\tools\apply_patches.ps1") -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\patches.log" | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE) { throw "apply_patches.ps1 Exit $LASTEXITCODE" }
        $exes = @("train_bot.exe", "rlbot_tests.exe", "dump_obs.exe", "duel.exe", "dump_policy_actions.exe", "write_metrics_csv.exe")
        if ($SkipBuild) {
            # Keine Tests auf alten Binaries (R18): jede muss jünger als der HEAD-Commit sein
            $headTime = [DateTimeOffset]::FromUnixTimeSeconds([long](Invoke-Native git @('-C', $Root, 'log', '-1', '--format=%ct'))).LocalDateTime
            foreach ($exe in $exes) {
                if (-not (Test-Path "$Build\$exe")) { throw "$exe fehlt in $Build (ohne -SkipBuild bauen)" }
                if ((Get-Item "$Build\$exe").LastWriteTime -lt $headTime) { throw "$exe ist aelter als der HEAD-Commit ($headTime): ohne -SkipBuild bauen" }
            }
            return "uebersprungen (-SkipBuild), Binaries vorhanden und juenger als HEAD"
        }
        Invoke-Native powershell @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$Root\bench\cpp\build.ps1", '-Flavor', $Flavor) -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\build.log" | Select-Object -Last 3 | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE) { throw "build.ps1 Exit $LASTEXITCODE, siehe $Res\build.log" }
        "OK"
    }

    # --- 3. Tests ---------------------------------------------------------------------
    Step "3" "Tests (-Repeat $Repeat)" @("2") {
        Invoke-Native powershell @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$Root\tools\run_all_tests.ps1", '-Repeat', $Repeat, '-Flavor', $Flavor) -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\tests.log" | Select-String -Pattern "bestanden|FAIL|fehlgeschlagen|passed|failed|Golden|Python:" | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE) { throw "run_all_tests.ps1 Exit $LASTEXITCODE, siehe $Res\tests.log" }
        "OK"
    }

    # --- 4. Python-Versionen --------------------------------------------------------------
    Step "4" "Python-Versionen gegen Pins" @() {
        Invoke-Native $Py @("$Root\tools\local\check_python_versions.py", '--json', "$Res\python_versions.json") -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\python_versions.txt" | ForEach-Object { Write-Host $_ }
        switch ($LASTEXITCODE) {
            0 { "OK, alle Pins stimmen" }
            1 { "OK mit ABWEICHUNG (Information, siehe python_versions.txt)" }
            2 { "OK mit FEHLENDEN PAKETEN (Information, siehe python_versions.txt)" }
            default { throw "check_python_versions.py Exit $LASTEXITCODE" }
        }
    }

    # --- 5. Smoke-Training --------------------------------------------------------------------
    Step "5" "Smoke-Training sanity.json ($SmokeSteps Steps)" @("2") {
        if ($SkipSmoke) { return "uebersprungen (-SkipSmoke)" }
        $smokeRun = "$Root\runs\local_check_$Date\sanity"
        if (Test-Path $smokeRun) { throw "Ordner existiert schon: $smokeRun" }
        New-Item -ItemType Directory -Force -Path $smokeRun | Out-Null
        $cfg = Get-Content "$Root\train\configs\sanity.json" -Raw | ConvertFrom-Json
        $cfg.learner.checkpoint_folder = ($smokeRun -replace '\\', '/') + "/checkpoints"
        $cfg.learner.timestep_limit = $SmokeSteps
        $cfg.learner.timesteps_per_save = [long]($SmokeSteps / 2)
        # sanity.json hat kein metrics.run; eine Zuweisung an eine fehlende Eigenschaft bricht ab
        $cfg.metrics | Add-Member -NotePropertyName run -NotePropertyValue "local_check_sanity" -Force
        $cfgPath = "$smokeRun\config.json"
        $cfg | ConvertTo-Json -Depth 10 | Set-Content -Path $cfgPath -Encoding UTF8
        Invoke-Native "$Build\train_bot.exe" @($cfgPath, '--save-on-exit', '--collision-meshes', "$Root\collision_meshes") -MergeStdErr -NoThrow |
            Tee-Object -FilePath "$Res\smoke_train.log" | Select-String -Pattern "Timestep limit|FATAL|Exception|extra_steps|save_on_exit" | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE) { throw "train_bot.exe Exit $LASTEXITCODE, siehe $Res\smoke_train.log" }
        Copy-Item "$smokeRun\metrics.csv" -Destination "$Res\smoke_metrics.csv"
        Copy-Item "$smokeRun\config_used.json" -Destination "$Res\smoke_config_used.json"
        Invoke-Native $Py @("$Root\tools\experiments\check_abort.py", "$smokeRun\metrics.csv", '--warmup', 5) -MergeStdErr -NoThrow | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -eq 3) { throw "Abbruchkriterium im Smoke-Lauf verletzt (nan/inf/leer?)" }
        $sumOut = Invoke-Native $Py @("$Root\tools\experiments\summarize.py", '--run', $smokeRun, '--out', "$Res\smoke", '--name', 'local_check_sanity') -MergeStdErr
        $sumOut | Select-Object -First 30 | ForEach-Object { Write-Host $_ }
        $ckpts = Get-ChildItem "$smokeRun\checkpoints" -Directory | Measure-Object
        "OK ($($ckpts.Count) Checkpoints, metrics in smoke_metrics.csv)"
    }

    # --- 6. Deployment-Smoke + Bestandsaufnahme ----------------------------------------------------
    Step "6" "Deployment-Smoke und Bestandsaufnahme ($Run)" @("2") {
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
    Compress-Archive -Path "$Res\*" -DestinationPath $zip
    Write-Host "`nErgebnisse: $Res" -ForegroundColor Green
    Write-Host "Zip zum Zurueckgeben: $zip" -ForegroundColor Green
}
$bad = @($status.Values | Where-Object { $_ -like "FEHLER*" -or $_ -like "uebersprungen wegen*" })
if ($bad.Count -or $status.Count -lt 6) {
    Write-Host "$($bad.Count) Schritt(e) mit Fehler oder uebersprungen wegen eines Fehlers, Details im Log." -ForegroundColor Red
    exit 1
}
Write-Host "Alle Schritte OK." -ForegroundColor Green
exit 0
