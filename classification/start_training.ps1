$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $workspace '.venv-detection\Scripts\python.exe'
$statusFile = Join-Path $PSScriptRoot 'status.json'
if (Test-Path -LiteralPath $statusFile) {
    $status = Get-Content -LiteralPath $statusFile -Raw | ConvertFrom-Json
    if ($status.state -notin @('complete', 'failed', 'stopped_by_user') -and $status.pid) {
        $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $($status.pid)"
        if ($existing -and $existing.CommandLine -match 'train_classification.py') {
            throw "Classification training is already running as PID $($status.pid)."
        }
    }
}
$env:PYTHONUNBUFFERED = '1'
$process = Start-Process -FilePath $pythonExe -ArgumentList '-u classification/train_classification.py' -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'training.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'training.stderr.log') -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'training.pid')
Write-Output "Started GPU classification comparison. Launcher PID: $($process.Id)"
