@echo off
REM -- Check for virtual environment installation --
REM -- python -m pip install virtualenv

REM -- Create virtual environment --
python -m venv ../envs

REM -- Activate virtual environment --
call ..\envs\Scripts\activate

REM -- Install requirements --
pip install -r requirements.txt

REM -- Keep the terminal open after the script finishes --
cmd /k
