$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $workspace '.venv-detection\Scripts\python.exe'
$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and
    $_.CommandLine -match 'continue_mobile_then_efficientnet.py|train_classification.py'
}
if ($active) { throw "Classification training is already active." }
$env:PYTHONUNBUFFERED = '1'
$process = Start-Process -FilePath $pythonExe -ArgumentList '-u classification/continue_mobile_then_efficientnet.py' -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'training.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'training.stderr.log') -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'training.pid')
Write-Output "Started continued GPU classification. Launcher PID: $($process.Id)"
