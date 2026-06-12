param (
    [string]$Message = "Auto-Deploy updates from Antigravity"
)

Write-Host "Starting Auto-Deployment to GitHub..." -ForegroundColor Cyan

# Check if git is initialized
if (!(Test-Path ".git")) {
    Write-Host "Error: Git repository not initialized. Run 'git init' first." -ForegroundColor Red
    exit 1
}

# Add all changed files
git add .

# Commit changes
git commit -m $Message

# Push to the default branch
git push origin HEAD

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment successful! The Hetzner VPS Watchdog will pick this up within 60 seconds." -ForegroundColor Green
} else {
    Write-Host "Deployment failed. Ensure your GitHub remote is set up correctly (git remote add origin <URL>)." -ForegroundColor Red
}
