# Sauberes Beenden eines laufenden train_bot.exe (Review-Befund R15). Wird von
# run_experiment.ps1 per Dot-Sourcing eingebunden:  . "$PSScriptRoot\TrainerControl.ps1"
#
# Stop-TrainerGracefully legt die Stop-Datei an, die train_bot.exe über --stop-file kennt. Der
# Trainer beendet dann die laufende Iteration, schreibt mit save_on_exit den End-Checkpoint und
# endet mit Exit 0. Erst wenn das nach -TimeoutSeconds nicht passiert ist, wird der Prozess als
# Notfall mit Stop-Process -Force beendet (dabei kann ein halb geschriebener Checkpoint entstehen;
# die Auswahl des End-Checkpoints prüft deshalb auf Vollständigkeit, R16).
#
# Rückgabe: "sauber" (von selbst beendet) oder "erzwungen" (Notfall).

function Stop-TrainerGracefully {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][string]$StopFile,
        [int]$TimeoutSeconds = 600
    )
    if ($Process.HasExited) { return "sauber" }
    Set-Content -Path $StopFile -Value "stop $(Get-Date -Format o)" -Encoding ASCII
    Write-Host "Stop-Datei angelegt: $StopFile (warte bis $TimeoutSeconds s auf das Ende der Iteration und den End-Checkpoint)"
    if ($Process.WaitForExit($TimeoutSeconds * 1000)) {
        Write-Host "Trainer hat sauber beendet (Exit $($Process.ExitCode))"
        return "sauber"
    }
    Write-Host "NOTFALL: Trainer hat nach $TimeoutSeconds s nicht reagiert, Stop-Process -Force" -ForegroundColor Red
    Stop-Process -Id $Process.Id -Force
    $Process.WaitForExit(30000) | Out-Null
    return "erzwungen"
}
