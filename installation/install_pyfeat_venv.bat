@echo off

cd /d "%~dp0\.."

echo Create virtual environment (env) under /third_party/py_feat --
python -m venv ./envs/py_feat

echo Activate virtual environment --
call ./envs/py_feat/Scripts/activate

echo Install pytorch with cu118
python -m pip install torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo install the requirements
python -m pip install -r ./nicetoolbox/detectors/third_party/py_feat/requirements.txt

echo Deactivating virtual environment
deactivate