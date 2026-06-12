Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   Antigravity VPS Local Setup Script    " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Environment Checks
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed on this VPS or not added to PATH." -ForegroundColor Red
    Write-Host "Please reinstall Python 3.10+ and check 'Add Python to PATH'."
    Read-Host "Press Enter to exit"
    Exit
}

$TargetDir = $PSScriptRoot | Split-Path -Parent
Set-Location $TargetDir

# 2. Virtual Environment
Write-Host "`n[1/3] Setting up isolated Python environment..." -ForegroundColor Yellow
if (!(Test-Path "venv")) {
    python -m venv venv
}
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\pip install -r backend\requirements.txt

# 3. Inject Secrets
Write-Host "`n[2/3] Injecting secure API keys..." -ForegroundColor Yellow
$NvidiaKey = "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"
$RunPodKey = "rpa_NFHMWT2NT9Z99FVUTJ9XM6LB6MELCLEHI2K1T6VUk485il"
$MT5Account = Read-Host "Enter your Vantage MT5 Account Number"
$MT5Password = Read-Host "Enter your Vantage MT5 Password"
$MT5Server = Read-Host "Enter your Vantage MT5 Server Name (e.g. VantageMarkets-Live)"

$EnvContent = @"
NVIDIA_API_KEY=$NvidiaKey
RUNPOD_API_KEY=$RunPodKey
MT5_ACCOUNT=$MT5Account
MT5_PASSWORD=$MT5Password
MT5_SERVER=$MT5Server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
"@
Set-Content -Path ".\.env" -Value $EnvContent

Write-Host "`n[3/3] Setup Complete! The AI is ready to wake up." -ForegroundColor Green
Write-Host "Starting the 24/7 Supervisor..." -ForegroundColor Cyan

.\venv\Scripts\python.exe backend\supervisor.py
Read-Host "Press Enter to exit"
