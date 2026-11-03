@echo off

:: Open a command prompt in the repo root with the nicetoolbox environment activated
cd /d "%~dp0"

if not exist "envs\nicetoolbox\Scripts\activate.bat" (
    echo ERROR: NICE Toolbox environment not found. Please run install_windows.bat first.
    pause
    exit /b 1
)

:: /k keeps the window open and waits for commands
cmd /k "envs\nicetoolbox\Scripts\activate.bat"
