# Lint fuer Review-Befund R2 (wird von tests/test_build_scripts.py unter powershell.exe 5.1
# aufgerufen): Findet in allen uebergebenen .ps1 Aufrufe nativer Programme, die NICHT ueber
# Invoke-Native (tools/NativeCommand.ps1) laufen.
#
# Gemeldet wird jeder Befehl, dessen Name ein natives Programm ist (git, cmake, python, ...,
# *.exe) oder der mit dem Aufrufoperator & auf etwas anderes als einen Skriptblock zeigt.
# Ausnahmen: die Hilfsfunktion selbst und Skriptblock-Variablen aus $AllowedCallVariables.
# Ausgabe: eine Zeile "<datei>:<zeile>: <befehl>" je Fund; Exit 0 = keine Funde.
param([Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][string[]]$Files)

$ErrorActionPreference = 'Stop'
$NativeNames = @('git', 'cmake', 'ninja', 'cmd', 'powershell', 'pwsh', 'python', 'py', 'pip',
                 'nvidia-smi', 'nvcc', 'cl', 'link', 'where')
$AllowedCallVariables = @('body')   # run_all_checks.ps1: Step { ... } ruft & $body auf

$findings = @()
foreach ($file in $Files) {
    $file = (Resolve-Path $file).Path
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$tokens, [ref]$errors)
    if ($errors.Count) {
        $findings += "${file}:$($errors[0].Extent.StartLineNumber): Parserfehler $($errors[0].Message)"
        continue
    }
    $commands = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true)
    foreach ($cmd in $commands) {
        # Aufrufe innerhalb der Hilfsfunktion selbst sind erlaubt
        $parent = $cmd.Parent
        $insideHelper = $false
        while ($parent) {
            if ($parent -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $parent.Name -eq 'Invoke-Native') {
                $insideHelper = $true; break
            }
            $parent = $parent.Parent
        }
        if ($insideHelper) { continue }

        $first = $cmd.CommandElements[0]
        $name = $cmd.GetCommandName()
        $bad = $false
        if ($cmd.InvocationOperator -eq [System.Management.Automation.Language.TokenKind]::Ampersand) {
            if ($first -is [System.Management.Automation.Language.VariableExpressionAst] -and
                $AllowedCallVariables -contains $first.VariablePath.UserPath) {
                $bad = $false
            } elseif ($first -is [System.Management.Automation.Language.ScriptBlockExpressionAst]) {
                $bad = $false
            } else {
                $bad = $true
            }
        } elseif ($name) {
            $leaf = [System.IO.Path]::GetFileName($name)
            $base = [System.IO.Path]::GetFileNameWithoutExtension($leaf)
            if ($leaf -like '*.exe' -or $NativeNames -contains $base.ToLower()) { $bad = $true }
        }
        if ($bad) {
            $findings += "${file}:$($cmd.Extent.StartLineNumber): $($cmd.Extent.Text.Split("`n")[0].Trim())"
        }
    }
}
$findings | ForEach-Object { Write-Output $_ }
if ($findings.Count) { exit 1 }
exit 0
