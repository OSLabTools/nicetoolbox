@echo off

REM -- Create virtual environment (env) under /third_party/xgaze_3cams --
python -m venv ../third_party/xgaze_3cams/env

REM -- Activat virtual environment --
call ../third_party/xgaze_3cams/env/Scripts/activate

REM -- Install pytorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

REM install the requirements
python -m pip install -r ../third_party/xgaze_3cams/requirements.txt

