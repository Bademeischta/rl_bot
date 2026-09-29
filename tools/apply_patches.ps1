# Wendet die Upstream-Patches aus third_party/patches/ auf den gepinnten Klon von RLGymPPO_CPP an
# (idempotent). Wird von bench/cpp/build.ps1 vor jedem Build aufgerufen.
#   powershell -ExecutionPolicy Bypass -File tools\apply_patches.ps1 [-Check] [-Reset] [-Repo <klon>]
#
# Patches (Reihenfolge = Anwendungsreihenfolge):
#   rlgympppo_cpp_gcc_compat.patch   Timer.h/gradscaler.hpp für GCC (Linux-Build der Tests);
#                                    auf MSVC wirkungslos, aber harmlos
#   rlgympppo_cpp_truncation.patch   Audit K1: Timeouts als Truncation (Gym::StepResult.truncated,
#                                    ThreadAgent, GAE-Bootstrap vom echten Folgezustand)
#   rlgympppo_cpp_speed.patch        Geschwindigkeit (G1 ff.): Zeitaufschlüsselung und die
#                                    Optimierungen aus AUDIT.md §9; baut auf dem Truncation-Patch auf
# Der libtorch-Patch (cuda.cmake) läuft getrennt über tools\patch_libtorch_cuda.ps1.
#
# Die Patches sind ein Stapel: Ein späterer Patch darf Zeilen ändern, die ein früherer eingeführt hat
# (Speed-Patch über dem Truncation-Patch). Der frühere lässt sich dann nicht mehr einzeln umkehren.
# Deshalb wird der Stand von hinten bestimmt: Lässt sich Patch k umkehren, sind er und alle davor
# angewendet. -Check prüft die fehlenden Patches gestapelt an einer Kopie der betroffenen Dateien.
# Neuen Stand des letzten Patches schreiben: python tools\export_upstream_patch.py
#
# -Repo:  anderer Klon als third_party\RLGymPPO_CPP (Tests: frischer Checkout des gepinnten Commits)
# -Reset: verwirft vorher alle Änderungen an versionierten Dateien des Klons (git checkout -- .),
#         z. B. wenn dort noch die erste Fassung des Truncation-Patches angewendet ist (Review R4).
#         Betrifft nur den Upstream-Klon, nie das RLbot-Repo.
param(
    [switch]$Check,
    [switch]$Reset,
    [string]$Repo = ""
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\NativeCommand.ps1"
$Root = Resolve-Path "$PSScriptRoot\.."
if ($Repo -eq "") { $Repo = "$Root\third_party\RLGymPPO_CPP" }
$Patches = @(
    "$Root\third_party\patches\rlgympppo_cpp_gcc_compat.patch",
    "$Root\third_party\patches\rlgympppo_cpp_truncation.patch",
    "$Root\third_party\patches\rlgympppo_cpp_speed.patch"
)
$Pinned = "ee4cc56fc8e43758cc898fdba92a9173a94e52f2"

if (-not (Test-Path "$Repo\.git")) { throw "Upstream-Klon fehlt: $Repo (siehe third_party\PINNED.md)" }
foreach ($patch in $Patches) { if (-not (Test-Path $patch)) { throw "Patch fehlt: $patch" } }

$head = (Invoke-Native git @('-C', $Repo, 'rev-parse', 'HEAD')).Trim()
if ($head -ne $Pinned) {
    Write-Host "WARNUNG: $Repo steht auf $head, gepinnt ist $Pinned (PINNED.md)." -ForegroundColor Yellow
}

if ($Reset) {
    if ($Check) { throw "-Check und -Reset schliessen sich aus" }
    Write-Host "Setze $Repo auf den sauberen Stand zurueck (git checkout -- .)" -ForegroundColor Yellow
    Invoke-Native git @('-C', $Repo, 'checkout', '--', '.') -MergeStdErr | ForEach-Object { Write-Host $_ }
}

# Stand von hinten bestimmen: der letzte Patch, der sich umkehren lässt, und alle davor sind angewendet.
# --check schreibt bei "passt nicht" nach stderr; das ist hier eine Antwort, kein Fehler.
$appliedUpTo = -1
for ($i = $Patches.Count - 1; $i -ge 0; $i--) {
    Invoke-Native git @('-C', $Repo, 'apply', '--check', '--reverse', $Patches[$i]) -NoThrow -Quiet
    if ($LASTEXITCODE -eq 0) { $appliedUpTo = $i; break }
}

# -Check: fehlende Patches gestapelt an einer Kopie der betroffenen Dateien prüfen, Klon bleibt unberührt
$checkDir = $null
if ($Check -and $appliedUpTo -lt $Patches.Count - 1) {
    $checkDir = Join-Path ([IO.Path]::GetTempPath()) ("rlbot_patchcheck_" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $checkDir | Out-Null
    for ($i = $appliedUpTo + 1; $i -lt $Patches.Count; $i++) {
        foreach ($line in Get-Content $Patches[$i]) {
            if ($line -match '^\+\+\+ b/(.+)$') {
                $src = Join-Path $Repo $matches[1]
                $dst = Join-Path $checkDir $matches[1]
                if ((Test-Path $src) -and -not (Test-Path $dst)) {
                    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
                    Copy-Item $src $dst
                }
            }
        }
    }
}

$applied = 0; $already = 0; $failed = 0
try {
    for ($i = 0; $i -lt $Patches.Count; $i++) {
        $patch = $Patches[$i]
        $name = Split-Path $patch -Leaf
        if ($i -le $appliedUpTo) {
            Write-Host "[bereits angewendet] $name"
            $already++
            continue
        }
        $target = if ($checkDir) { $checkDir } else { $Repo }
        # Ausserhalb eines Git-Repos arbeitet git apply wie patch (Prüfkopie)
        Invoke-Native git @('-C', $target, 'apply', '--check', $patch) -NoThrow -Quiet
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FEHLER] $name laesst sich weder anwenden noch ist er angewendet." -ForegroundColor Red
            Write-Host "         Upstream-Stand pruefen (git -C third_party\RLGymPPO_CPP status / diff)."
            Write-Host "         Zeilenenden: third_party\patches\*.patch muessen LF haben (.gitattributes)."
            Write-Host "         Aeltere Fassung des Patches angewendet? Dann: tools\apply_patches.ps1 -Reset"
            Write-Host "         Eigene Aenderungen am Klon? Dann: python tools\export_upstream_patch.py"
            $failed++
            # Die folgenden Patches bauen auf diesem auf
            break
        }
        Invoke-Native git @('-C', $target, 'apply', $patch) -NoThrow
        if ($LASTEXITCODE -ne 0) { $failed++; Write-Host "[FEHLER] git apply $name" -ForegroundColor Red; break }
        if ($Check) {
            Write-Host "[anwendbar, nicht angewendet] $name (ohne -Check wird er angewendet)"
            continue
        }
        Write-Host "[angewendet] $name"
        $applied++
    }
} finally {
    if ($checkDir) { Remove-Item -Recurse -Force $checkDir }
}

Write-Host "Patches: $applied angewendet, $already bereits vorhanden, $failed fehlgeschlagen."
if ($failed -gt 0) { exit 1 }
exit 0
