@echo off

REM -- Create virtual environment (env) under /third_party/xgaze_3cams --
python -m venv ../detectors/third_party/xgaze_3cams/env

REM -- Activat virtual environment --
call ../detectors/third_party/xgaze_3cams/env/Scripts/activate

REM -- Install pytorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

REM install the requirements
python -m pip install -r ../detectors/third_party/xgaze_3cams/requirements.txt
