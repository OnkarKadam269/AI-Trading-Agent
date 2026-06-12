param (
    [string]$TargetDir = "C:\AI_AGENT"
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   Antigravity VPS 2-Minute Bootstrap    " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Environment Checks
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Git is not installed on this VPS." -ForegroundColor Red
    Write-Host "Please download and install Git for Windows, then run this script again."
    Exit
}
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed on this VPS." -ForegroundColor Red
    Write-Host "Please download and install Python 3.10+ for Windows, then run this script again."
    Exit
}

# 2. Clone the Repository
$GitUsername = Read-Host "Enter your GitHub Username"
$GitPAT = Read-Host "Enter your GitHub PAT (Token)"
$GitRepo = Read-Host "Enter your GitHub Repository URL (e.g., github.com/username/repo.git)"

$CloneUrl = "https://$GitUsername`:$GitPAT@$GitRepo"

Write-Host "`n[1/4] Cloning codebase to $TargetDir..." -ForegroundColor Yellow
if (Test-Path $TargetDir) { Remove-Item -Recurse -Force $TargetDir }
git clone $CloneUrl $TargetDir
Set-Location $TargetDir

# 3. Virtual Environment
Write-Host "`n[2/4] Setting up isolated Python environment..." -ForegroundColor Yellow
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\pip install -r backend\requirements.txt

# 4. Inject Secrets
Write-Host "`n[3/4] Injecting secure API keys..." -ForegroundColor Yellow
$GeminiKey = Read-Host "Paste your Google Gemini API Key"
$RunPodKey = Read-Host "Paste your RunPod API Key"
$RunPodEndpoint = Read-Host "Paste your RunPod Serverless Endpoint ID (Leave blank if not deployed yet)"

$EnvContent = @"
GEMINI_API_KEY=$GeminiKey
RUNPOD_API_KEY=$RunPodKey
RUNPOD_ENDPOINT_ID=$RunPodEndpoint
MT5_ACCOUNT=25504422
MT5_PASSWORD=Pass@123
MT5_SERVER=VantageMarkets-Demo
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
"@
Set-Content -Path ".\.env" -Value $EnvContent

Write-Host "`n[4/4] Setup Complete! The AI is ready to wake up." -ForegroundColor Green
Write-Host "Starting the 24/7 Supervisor..." -ForegroundColor Cyan

.\venv\Scripts\python.exe backend\supervisor.py
