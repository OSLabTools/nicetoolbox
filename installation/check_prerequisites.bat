@echo off
setlocal

:: Check if Python is installed and show its version
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not found in the system PATH.
    echo Please install Python from https://www.python.org/downloads/
    pause
) else (
    echo Python is installed. Version:
	python --version
	pause
)

:: Check if FFmpeg is installed
ffmpeg -version >nul 2>nul
if %errorlevel% neq 0 (
    echo FFmpeg is not installed or not found in the system PATH.
    echo Please install FFmpeg from https://ffmpeg.org/download.html
    pause
) else (
    echo FFmpeg is installed.
    pause
)

:: Check if Conda is installed
call conda activate base >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda is not installed or not found in the system PATH.
    echo Please install Conda from https://github.com/conda-forge/miniforge
) else (
    echo Conda is installed.
    pause
)

endlocal
