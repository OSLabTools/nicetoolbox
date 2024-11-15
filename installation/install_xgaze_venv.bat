@echo off

cd /d "%~dp0\.."

echo Create virtual environment (env) under /third_party/xgaze_3cams --
python -m venv ./nicetoolbox/detectors/third_party/xgaze_3cams/env

echo Activate virtual environment --
call ./nicetoolbox/detectors/third_party/xgaze_3cams/env/Scripts/activate

echo Install pytorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo install the requirements
python -m pip install -r ./nicetoolbox/detectors/third_party/xgaze_3cams/requirements.txt

echo Deactivating virtual environment
deactivate