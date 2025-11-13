# ============================================================
# Auto Push to GitHub Script (PowerShell)
# This script automatically commits and pushes changes to GitHub
# ============================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Auto Push to GitHub - Plant Disease Detection Project" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in a git repository
try {
    $null = git rev-parse --git-dir 2>$null
} catch {
    Write-Host "ERROR: This is not a git repository!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Red
    pause
    exit 1
}

# Check if there are any changes
$changes = git status --short
if ([string]::IsNullOrWhiteSpace($changes)) {
    Write-Host "No changes detected. Repository is up to date." -ForegroundColor Green
    Write-Host ""
    pause
    exit 0
}

# Show current status
Write-Host "Current git status:" -ForegroundColor Yellow
git status --short
Write-Host ""

# Ask for commit message
$commitMsg = Read-Host "Enter commit message (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($commitMsg)) {
    $commitMsg = "Auto-update: Project changes"
}

Write-Host ""
Write-Host "Committing changes with message: $commitMsg" -ForegroundColor Yellow
git add .

try {
    git commit -m $commitMsg
    if ($LASTEXITCODE -ne 0) {
        throw "Commit failed"
    }
} catch {
    Write-Host "ERROR: Commit failed!" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "Pulling latest changes from GitHub..." -ForegroundColor Yellow
try {
    git pull origin main --rebase
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Pull had conflicts. Attempting to continue..." -ForegroundColor Yellow
    }
} catch {
    Write-Host "WARNING: Pull encountered issues. Continuing with push..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
try {
    git push origin main
    if ($LASTEXITCODE -ne 0) {
        throw "Push failed"
    }
} catch {
    Write-Host ""
    Write-Host "ERROR: Push failed! You may need to resolve conflicts manually." -ForegroundColor Red
    Write-Host "Try running: git pull origin main --rebase" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "SUCCESS! Changes have been pushed to GitHub." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Repository: https://github.com/Chaitanya-G-H/plant-disease" -ForegroundColor Cyan
Write-Host ""
pause

