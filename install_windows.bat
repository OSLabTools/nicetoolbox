@echo off
setlocal enableextensions enabledelayedexpansion

:: Run from the repo root so relative paths work wherever the script is called from
pushd "%~dp0"

set "RC=0"
:: Interactive mode needs a person at the keyboard anyway, so pause at the end to keep
:: a double-clicked window open. -y/--yes, --detectors or --core-only mean no prompts.
set "PAUSE_AT_END=1"
echo( %* | findstr /i /r /c:" -y\>" /c:" --yes\>" /c:" --detectors\>" /c:" --core-only\>" >nul && set "PAUSE_AT_END="

echo.
echo ************************************************************
echo   NICE Toolbox Installer
echo ************************************************************
echo.

:: ---------------------------------------------------------------------------
:: Step 1 - Install uv if not already on PATH
:: ---------------------------------------------------------------------------
where uv >nul 2>&1
if not errorlevel 1 (
    echo [1/3] uv found: OK
    goto :have_uv
)

echo [1/3] uv not found. Installing uv...
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 (
    echo ERROR: uv installation failed. Please install uv manually:
    echo   https://docs.astral.sh/uv/getting-started/installation/
    set "RC=1"
    goto :end
)

:: Add uv to PATH for the rest of this session
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:: Verify
where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv was installed but is not on PATH.
    echo Please open a new terminal and run "%~f0" again with the same options.
    set "RC=1"
    goto :end
)
echo uv installed successfully.

:have_uv

:: ---------------------------------------------------------------------------
:: Step 2 - Use uv to install Python 3.10 (no separate Python installer needed)
:: ---------------------------------------------------------------------------
echo [2/3] Ensuring Python 3.10 is available via uv...
uv python install 3.10
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python 3.10 via uv.
    set "RC=1"
    goto :end
)
echo Python 3.10 ready.

:: ---------------------------------------------------------------------------
:: Step 3 - Run the main installer script
:: ---------------------------------------------------------------------------
echo [3/3] Running the installer...
echo.
uv run --no-project --python 3.10 python -m nicetoolbox.installation.install %*
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo Installation finished with errors. Check the output above for details.
)

:end
if defined PAUSE_AT_END pause
popd
endlocal & exit /b %RC%
