# Stufe 4, Audit H5: A/B über die Gradientenschritte pro Iteration (6 / 3 / 2), misst lokal
# SPS UND Lernkurve, mit Wiederholungen für die Streuung (bis zu 7 % zwischen identischen Läufen).
#
#   powershell -ExecutionPolicy Bypass -File tools\experiments\bench_expbuffer.ps1 `
#       -StartCheckpoint runs\lucy_1v1\checkpoints\<steps> [-Steps 20000000] [-Repeats 2] [-Seed 123]
#   ... -DryRun   zeigt nur den Plan (Variante, Wiederholung, Seed, Baseline), startet nichts
#
# Jede Variante läuft über tools\experiments\run_experiment.ps1 (gleicher Start-Checkpoint und
# Step-Zahl). Seeds (Review-Befund R13): Wiederholung r bekommt den Seed -Seed + (r - 1); innerhalb
# einer Wiederholung teilen alle Varianten denselben Seed (gepaarter Vergleich), zwischen den
# Wiederholungen unterscheidet er sich, sonst misst die Wiederholung nur dieselben Env-Zufälle noch
# einmal. Die erste Variante (6 Updates = Status quo) jeder Wiederholung ist die Baseline für die
# Duelle ihrer Wiederholung. Ergebnis: results\bench_expbuffer_<datum>.md und results\exp_h5_*-Ordner.
#
# Entscheidungsregel (AUDIT.md Roadmap 15): eine schnellere Variante nur behalten, wenn die
# Lernkurve (ep_end_goal, Ballkontakt, TrueSkill, Duell gegen Baseline-Ende) nicht schlechter ist.
param(
    [Parameter(Mandatory = $true)][string]$StartCheckpoint,
    [long]$Steps = 20000000,
    [int]$Repeats = 2,
    [int]$Seed = 123,
    [int]$DuelGames = 1000,
    [int]$LadderGames = 20,
    [string]$Flavor = "cu128",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
# Native Programme nur über Invoke-Native (Review-Befund R2: stderr unter PowerShell 5.1)
. "$Root\tools\NativeCommand.ps1"
$Py = "$Root\.venv\Scripts\python.exe"
$Runner = "$Root\tools\experiments\run_experiment.ps1"
$Variants = @("h5_updates6_epochs2_buf3", "h5_updates3_epochs1_buf3", "h5_updates2_epochs2_buf1")

# Plan: je Wiederholung ein eigener Seed, alle Varianten einer Wiederholung mit demselben
$plan = @()
foreach ($rep in 1..$Repeats) {
    foreach ($variant in $Variants) {
        $plan += [pscustomobject]@{ Rep = $rep; Variant = $variant; Name = "${variant}_r$rep"
                                    Seed = $Seed + ($rep - 1); IsRepBaseline = ($variant -eq $Variants[0]) }
    }
}
if ($DryRun) {
    foreach ($p in $plan) {
        Write-Output ("PLAN rep={0} variant={1} seed={2} baseline={3}" -f $p.Rep, $p.Variant, $p.Seed,
                      $(if ($p.IsRepBaseline) { "selbst" } else { "$($Variants[0])_r$($p.Rep)" }))
    }
    exit 0
}

$Date = Get-Date -Format "yyyy-MM-dd_HHmmss"
$Log = "$Root\results\bench_expbuffer_$Date.log"
New-Item -ItemType Directory -Force -Path "$Root\results" | Out-Null
Start-Transcript -Path $Log | Out-Null

$folders = @()
$firstBaseline = ""
try {
    $repBaseline = ""
    foreach ($p in $plan) {
        if ($p.IsRepBaseline) { $repBaseline = "" }
        Write-Host "`n=== $($p.Name) ($Steps Steps, Seed $($p.Seed)) ===" -ForegroundColor Cyan
        $runArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Runner,
                     "-Config", "$Root\train\configs\experiments\$($p.Variant).json",
                     "-StartCheckpoint", $StartCheckpoint, "-Steps", $Steps, "-Seed", $p.Seed,
                     "-Name", $p.Name, "-DuelGames", $DuelGames, "-Flavor", $Flavor, "-SkipLadder")
        if ($repBaseline -ne "") { $runArgs += @("-Baseline", $repBaseline) }
        Invoke-Native powershell $runArgs -MergeStdErr -NoThrow | ForEach-Object { Write-Host $_ }
        $rc = $LASTEXITCODE
        $folder = Get-ChildItem "$Root\results" -Directory | Where-Object { $_.Name -like "exp_$($p.Name)_*" } |
            Sort-Object LastWriteTime | Select-Object -Last 1
        if ($rc -ne 0 -or -not $folder) {
            Write-Host "Variante $($p.Name) fehlgeschlagen (Exit $rc)" -ForegroundColor Red
            continue
        }
        $folders += $folder.FullName
        if ($p.IsRepBaseline) { $repBaseline = $folder.FullName }
        if ($firstBaseline -eq "" -and $p.IsRepBaseline) { $firstBaseline = $folder.FullName }
    }

    if ($folders.Count -ge 2 -and $firstBaseline -ne "") {
        $out = "$Root\results\bench_expbuffer_$Date.md"
        Invoke-Native $Py (@("$Root\tools\experiments\compare.py") + $folders +
                           @('--baseline', $firstBaseline, '--out', $out, '--ladder-games', $LadderGames)) -MergeStdErr |
            ForEach-Object { Write-Host $_ }
        Write-Host "`nVergleich: $out" -ForegroundColor Green
        Write-Host "SPS-Spalte = Durchsatz, ep_end_goal/Ballkontakt/Duell = Lernkurve. Das Duell jeder Variante"
        Write-Host "läuft gegen die 6-Update-Variante derselben Wiederholung (gleicher Seed). Streuung zwischen"
        Write-Host "den Wiederholungen (_r1/_r2, verschiedene Seeds) zeigt, ob ein Unterschied belastbar ist."
    }
}
finally {
    Stop-Transcript | Out-Null
}
Write-Host "Log: $Log"
