$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $workspace '.venv-detection\Scripts\python.exe'
$statusPath = Join-Path $PSScriptRoot 'status.json'
$initial = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
$runId = $initial.run_id
while ($true) {
    Start-Sleep -Seconds 30
    try { $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json } catch { continue }
    if ($status.run_id -ne $runId) { exit }
    if ($status.state -eq 'complete') {
        $externalLog = Join-Path $PSScriptRoot 'external_test.log'
        & $pythonExe (Join-Path $PSScriptRoot 'test_external_pipeline.py') *> $externalLog
        $report = Get-Content -LiteralPath $status.final_report -Raw | ConvertFrom-Json
        $test = $report.test_metrics
        $message = "Maturity classification completed.`r`n"
        $message += "Winner: $($report.winner.model_name)`r`n"
        $message += ('Test accuracy: {0:P2}' -f $test.accuracy) + "`r`n"
        $message += ('Test macro-F1: {0:P2}' -f $test.macro_f1) + "`r`n"
        $message += ('Test balanced accuracy: {0:P2}' -f $test.balanced_accuracy) + "`r`n"
        $message += "External web-image results: C:\Reinforcement\classification\external_results"
        $message | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'completion-notification.txt')
        $shell = New-Object -ComObject WScript.Shell
        $null = $shell.Popup($message, 0, 'Gourd maturity classification', 64)
        exit
    }
    if ($status.state -eq 'failed') {
        $message = "Classification training stopped with an error. See C:\Reinforcement\classification\status.json."
        $shell = New-Object -ComObject WScript.Shell
        $null = $shell.Popup($message, 0, 'Gourd maturity classification', 16)
        exit
    }
}
