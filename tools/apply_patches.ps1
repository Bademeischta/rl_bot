# Wendet die Upstream-Patches aus third_party/patches/ auf den gepinnten Klon von RLGymPPO_CPP an
# (idempotent). Wird von bench/cpp/build.ps1 vor jedem Build aufgerufen.
#   powershell -ExecutionPolicy Bypass -File tools\apply_patches.ps1 [-Check] [-Repo <klon>]
#
# Patches (Reihenfolge = Anwendungsreihenfolge):
#   rlgympppo_cpp_gcc_compat.patch   Timer.h/gradscaler.hpp für GCC (Linux-Build der Tests);
#                                    auf MSVC wirkungslos, aber harmlos
#   rlgympppo_cpp_truncation.patch   Audit K1: Timeouts als Truncation (Gym::StepResult.truncated,
#                                    ThreadAgent, GAE-Bootstrap vom echten Folgezustand)
# Der libtorch-Patch (cuda.cmake) läuft getrennt über tools\patch_libtorch_cuda.ps1.
#
# -Repo: anderer Klon als third_party\RLGymPPO_CPP (Tests: frischer Checkout des gepinnten Commits)
param(
    [switch]$Check,
    [string]$Repo = ""
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\NativeCommand.ps1"
$Root = Resolve-Path "$PSScriptRoot\.."
if ($Repo -eq "") { $Repo = "$Root\third_party\RLGymPPO_CPP" }
$Patches = @(
    "$Root\third_party\patches\rlgympppo_cpp_gcc_compat.patch",
    "$Root\third_party\patches\rlgympppo_cpp_truncation.patch"
)
$Pinned = "ee4cc56fc8e43758cc898fdba92a9173a94e52f2"

if (-not (Test-Path "$Repo\.git")) { throw "Upstream-Klon fehlt: $Repo (siehe third_party\PINNED.md)" }

$head = (Invoke-Native git @('-C', $Repo, 'rev-parse', 'HEAD')).Trim()
if ($head -ne $Pinned) {
    Write-Host "WARNUNG: $Repo steht auf $head, gepinnt ist $Pinned (PINNED.md)." -ForegroundColor Yellow
}

$applied = 0; $already = 0; $failed = 0
foreach ($patch in $Patches) {
    $name = Split-Path $patch -Leaf
    if (-not (Test-Path $patch)) { throw "Patch fehlt: $patch" }

    # --check schreibt bei "passt nicht" nach stderr; das ist hier eine Antwort, kein Fehler.
    Invoke-Native git @('-C', $Repo, 'apply', '--check', '--reverse', $patch) -NoThrow -Quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[bereits angewendet] $name"
        $already++
        continue
    }
    Invoke-Native git @('-C', $Repo, 'apply', '--check', $patch) -NoThrow -Quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FEHLER] $name laesst sich weder anwenden noch ist er angewendet." -ForegroundColor Red
        Write-Host "         Upstream-Stand pruefen (git -C third_party\RLGymPPO_CPP status / diff)."
        Write-Host "         Zeilenenden: third_party\patches\*.patch muessen LF haben (.gitattributes)."
        $failed++
        continue
    }
    if ($Check) {
        Write-Host "[anwendbar, nicht angewendet] $name (ohne -Check wird er angewendet)"
        continue
    }
    Invoke-Native git @('-C', $Repo, 'apply', $patch) -NoThrow
    if ($LASTEXITCODE -ne 0) { $failed++; Write-Host "[FEHLER] git apply $name" -ForegroundColor Red; continue }
    Write-Host "[angewendet] $name"
    $applied++
}

Write-Host "Patches: $applied angewendet, $already bereits vorhanden, $failed fehlgeschlagen."
if ($failed -gt 0) { exit 1 }
exit 0
