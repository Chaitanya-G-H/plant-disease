@echo off
REM ============================================================
REM Auto Push to GitHub Script
REM This script automatically commits and pushes changes to GitHub
REM ============================================================

echo ============================================================
echo Auto Push to GitHub - Plant Disease Detection Project
echo ============================================================
echo.

REM Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo ERROR: This is not a git repository!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Check if there are any changes
git diff --quiet && git diff --cached --quiet
if errorlevel 1 (
    echo Changes detected! Proceeding with commit and push...
    echo.
) else (
    echo No changes detected. Repository is up to date.
    echo.
    pause
    exit /b 0
)

REM Show current status
echo Current git status:
git status --short
echo.

REM Ask for commit message
set /p COMMIT_MSG="Enter commit message (or press Enter for default): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Auto-update: Project changes

echo.
echo Committing changes with message: "%COMMIT_MSG%"
git add .
git commit -m "%COMMIT_MSG%"

if errorlevel 1 (
    echo ERROR: Commit failed!
    pause
    exit /b 1
)

echo.
echo Pushing to GitHub...
git pull origin main --rebase
git push origin main

if errorlevel 1 (
    echo.
    echo ERROR: Push failed! You may need to resolve conflicts manually.
    echo Try running: git pull origin main --rebase
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! Changes have been pushed to GitHub.
echo ============================================================
echo Repository: https://github.com/Chaitanya-G-H/plant-disease
echo.
pause

