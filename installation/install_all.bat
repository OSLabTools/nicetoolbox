@echo off

echo Starting main Installation Script...

echo Changing working directory
cd /d "%~dp0\.."


echo Calling install_nicetoolbox_venv.bat...
call .\installation\install_nicetoolbox_venv.bat
if errorlevel 1 (
    echo install_nicetoolbox_venv.bat encountered an error. Exiting...
    exit /b %errorlevel%
)

:: Call the second batch file
echo Calling install_xgaze_venv.bat...
call .\installation\install_xgaze_venv.bat
if errorlevel 1 (
    echo install_xgaze_venv.bat encountered an error. Exiting...
    exit /b %errorlevel%
)

:: Call the third batch file
echo Calling install_openmmlab_conda.bat...
call .\installation\install_openmmlab_conda.bat
if errorlevel 1 (
    echo install_openmmlab_conda.bat encountered an error. Exiting...
    exit /b %errorlevel%
)

echo All installation scripts completed successfully.
pause