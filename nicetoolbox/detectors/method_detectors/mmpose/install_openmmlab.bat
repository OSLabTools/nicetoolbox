@echo off

:: Stop on error
setlocal enableextensions enabledelayedexpansion

:: This script is called from the Makefile in the root directory (cd = repo root).
set ENV_PATH=.\envs\openmmlab

echo Creating virtual environment at %ENV_PATH% with Python 3.8 using uv...
uv venv "%ENV_PATH%" --python 3.8

set PYTHON_BIN=%ENV_PATH%\Scripts\python.exe

:: Install PyTorch with CUDA
echo Installing PyTorch and dependencies...
uv pip install --python "%PYTHON_BIN%" torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

:: Install nicetoolbox-core
echo Installing nicetoolbox-core dependencies...
uv pip install --python "%PYTHON_BIN%" -e ./nicetoolbox_core

:: Install MMPose and its dependencies
echo Installing MMPose and dependencies...
uv pip install --python "%PYTHON_BIN%" setuptools pip openmim fsspec mmengine pyparsing six toml protobuf==3.20.3

set MIM_BIN=%ENV_PATH%\Scripts\mim.exe
echo Installing openmmlab packages via mim...
"%MIM_BIN%" install "mmcv==2.1.0"
"%MIM_BIN%" install "mmdet>=3.1.0,<3.3.0"
"%MIM_BIN%" install "mmpretrain>=1.0.0rc8"

echo Installing MMPose from the OSLabTools fork...
uv pip install --python "%PYTHON_BIN%" --no-build-isolation -r ./nicetoolbox/detectors/method_detectors/mmpose/mmpose_requirements.txt

echo OPENMMLab Environment setup completed successfully with uv.
endlocal
