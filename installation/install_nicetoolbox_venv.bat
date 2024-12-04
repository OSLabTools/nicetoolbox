@echo off
REM -- Check for virtual environment installation --
REM -- python -m pip install virtualenv
cd /d "%~dp0\.."

echo Create virtual environment
python -m venv ./envs/nicetoolbox

echo Activate virtual environment
call .\envs\nicetoolbox\Scripts\activate

echo Install requirements
pip install -r .\installation\requirements.txt

echo Nice toolbox environment setup completed successfully.

echo Deactivating virtual environment
deactivate
