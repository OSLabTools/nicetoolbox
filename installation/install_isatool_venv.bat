@echo off
REM -- Check for virtual environment installation --
REM -- python -m pip install virtualenv

REM -- Create virtual environment --
python -m venv envs

REM -- Activate virtual environment --
call envs\Scripts\activate

REM -- Install local package in editable mode (assuming the package setup.py is in a directory called 'mypackage' one level up) --
pip install -e ..\oslab_utils

REM -- Install requirements --
pip install -r requirements.txt

REM -- Keep the terminal open after the script finishes --
cmd /k
