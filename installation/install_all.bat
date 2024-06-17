@echo off
echo Starting main Installation Script...


echo Calling install_isatool_venv.bat...
call install_isatool_venv.bat
if errorlevel 1 (
    echo install_isatool_venv.bat encountered an error. Exiting...
    exit /b %errorlevel%
)

:: Call the second batch file
echo Calling install_xgaze_venv.bat...
call install_xgaze_venv.bat
if errorlevel 1 (
    echo install_xgaze_venv.bat encountered an error. Exiting...
    exit /b %errorlevel%
)

:: Call the third batch file
echo Calling install_openmmlab_conda.bat...
call install_openmmlab_conda.bat
if errorlevel 1 (
    echo install_openmmlab_conda.bat encountered an error. Exiting...
    exit /b %errorlevel%
)

echo All installation scripts completed successfully.
pause