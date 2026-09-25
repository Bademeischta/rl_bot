# Stufe 4, Audit H5: A/B über die Gradientenschritte pro Iteration (6 / 3 / 2), misst lokal
# SPS UND Lernkurve, mit Wiederholungen für die Streuung (bis zu 7 % zwischen identischen Läufen).
#
#   powershell -ExecutionPolicy Bypass -File tools\experiments\bench_expbuffer.ps1 `
#       -StartCheckpoint runs\lucy_1v1\checkpoints\2704829056 [-Steps 20000000] [-Repeats 2] [-Seed 123]
#
# Jede Variante läuft über tools\experiments\run_experiment.ps1 (gleicher Start-Checkpoint, Seed,
# Step-Zahl); die erste Variante (6 Updates = Status quo) ist die Baseline für compare.py.
# Ergebnis: results\bench_expbuffer_<datum>.md (Tabelle) und die einzelnen results\exp_h5_*-Ordner.
#
# Entscheidungsregel (AUDIT.md Roadmap 15): eine schnellere Variante nur behalten, wenn die
# Lernkurve (ep_end_goal, Ballkontakt, TrueSkill, Duell gegen Baseline-Ende) nicht schlechter ist.
param(
    [Parameter(Mandatory = $true)][string]$StartCheckpoint,
    [long]$Steps = 20000000,
    [int]$Repeats = 2,
    [int]$Seed = 123,
    [int]$DuelGames = 100,
    [string]$Flavor = "cu128"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
# Native Programme nur über Invoke-Native (Review-Befund R2: stderr unter PowerShell 5.1)
. "$Root\tools\NativeCommand.ps1"
$Py = "$Root\.venv\Scripts\python.exe"
$Runner = "$Root\tools\experiments\run_experiment.ps1"
$Variants = @("h5_updates6_epochs2_buf3", "h5_updates3_epochs1_buf3", "h5_updates2_epochs2_buf1")
$Date = Get-Date -Format "yyyy-MM-dd_HHmm"
$Log = "$Root\results\bench_expbuffer_$Date.log"
New-Item -ItemType Directory -Force -Path "$Root\results" | Out-Null
Start-Transcript -Path $Log | Out-Null

$folders = @()
$baselineFolder = ""
try {
    foreach ($rep in 1..$Repeats) {
        foreach ($variant in $Variants) {
            $name = "${variant}_r$rep"
            Write-Host "`n=== $name ($Steps Steps, Seed $Seed) ===" -ForegroundColor Cyan
            $runArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Runner,
                      "-Config", "$Root\train\configs\experiments\$variant.json",
                      "-StartCheckpoint", $StartCheckpoint, "-Steps", $Steps, "-Seed", $Seed,
                      "-Name", $name, "-DuelGames", $DuelGames, "-Flavor", $Flavor, "-SkipLadder")
            if ($baselineFolder -ne "") { $runArgs += @("-Baseline", $baselineFolder) }
            Invoke-Native powershell $runArgs -MergeStdErr -NoThrow | ForEach-Object { Write-Host $_ }
            $rc = $LASTEXITCODE
            $folder = Get-ChildItem "$Root\results" -Directory | Where-Object { $_.Name -like "exp_${name}_*" } |
                Sort-Object LastWriteTime | Select-Object -Last 1
            if ($rc -ne 0 -or -not $folder) {
                Write-Host "Variante $name fehlgeschlagen (Exit $rc)" -ForegroundColor Red
                continue
            }
            $folders += $folder.FullName
            if ($baselineFolder -eq "") { $baselineFolder = $folder.FullName }
        }
    }

    if ($folders.Count -ge 2) {
        $out = "$Root\results\bench_expbuffer_$Date.md"
        Invoke-Native $Py (@("$Root\tools\experiments\compare.py") + $folders + @('--baseline', $baselineFolder, '--out', $out)) -MergeStdErr | ForEach-Object { Write-Host $_ }
        Write-Host "`nVergleich: $out" -ForegroundColor Green
        Write-Host "SPS-Spalte = Durchsatz, ep_end_goal/Ballkontakt/Duell = Lernkurve. Streuung zwischen den"
        Write-Host "Wiederholungen (_r1/_r2) derselben Variante zeigt, ob ein Unterschied belastbar ist."
    }
}
finally {
    Stop-Transcript | Out-Null
}
Write-Host "Log: $Log"
