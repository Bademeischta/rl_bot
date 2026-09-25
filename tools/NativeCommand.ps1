# Gemeinsame Hilfsfunktion für native Programme (git, cmake, python, *.exe) in allen Skripten.
# Einbinden per Dot-Sourcing:  . "$PSScriptRoot\..\NativeCommand.ps1"  (Pfad je nach Ordner)
#
# Warum (Review-Befund R2): Windows PowerShell 5.1 verpackt jede stderr-Zeile eines nativen
# Programms in einen ErrorRecord (NativeCommandError). Mit $ErrorActionPreference = 'Stop' wird
# daraus ein Abbruch, auch mit 2>$null oder 2>&1, und auch wenn das Programm mit Exit 0 endet
# (git schreibt Warnungen und "patch failed" bei --check nach stderr, cmake und MSVC ebenso).
# Invoke-Native schaltet dafür lokal auf 'Continue', behandelt stderr als Text und entscheidet
# ausschließlich über den Exit-Code.
#
#   Invoke-Native git @('-C', $Root, 'rev-parse', 'HEAD')           # stdout als Rückgabe
#   Invoke-Native $Exe $argList -MergeStdErr | Tee-Object build.log  # stderr mit ins Log
#   Invoke-Native git @('apply', '--check', $p) -NoThrow -Quiet; if ($LASTEXITCODE) { ... }
#
# Parameter:
#   -MergeStdErr   stderr-Zeilen als Strings in die Ausgabe (für Logs); sonst per Write-Host
#                  auf die Konsole (landen im Transcript, aber nicht im Rückgabewert)
#   -Quiet         stderr verwerfen
#   -NoThrow       kein Abbruch bei Exit-Code außerhalb von -OkExitCodes; $LASTEXITCODE prüfen
#   -OkExitCodes   Exit-Codes, die kein Fehler sind (Default 0)

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Exe,
        [Parameter(Position = 1)][object[]]$ArgumentList = @(),
        [int[]]$OkExitCodes = @(0),
        [switch]$MergeStdErr,
        [switch]$Quiet,
        [switch]$NoThrow
    )

    # Ein fehlendes Programm soll klar abbrechen, statt einen alten $LASTEXITCODE zu hinterlassen.
    if (-not (Get-Command $Exe -CommandType Application -ErrorAction SilentlyContinue)) {
        throw "Programm nicht gefunden: $Exe"
    }

    $global:LASTEXITCODE = 0
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @ArgumentList 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $line = $_.ToString()
                if ($MergeStdErr) { $line }
                elseif (-not $Quiet) { Write-Host $line }
            } else {
                $_
            }
        }
        $code = $global:LASTEXITCODE
    } finally {
        $ErrorActionPreference = $saved
    }

    if (-not $NoThrow -and $OkExitCodes -notcontains $code) {
        throw "$Exe $($ArgumentList -join ' ') endete mit Exit-Code $code"
    }
}
