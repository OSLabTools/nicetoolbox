@echo off
REM -- Check for virtual environment installation --
REM -- python -m pip install virtualenv

REM -- Create virtual environment --
python -m venv ./env

REM -- Activate virtual environment --
call .\env\Scripts\activate

REM -- Install requirements --
pip install -r .\installation\requirements.txt

REM -- Keep the terminal open after the script finishes --
cmd /k
