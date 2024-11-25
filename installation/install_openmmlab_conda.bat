@echo off
:: Stop on error
setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0\.."
:: Create a conda environment
echo Creating conda environment...
call conda create --name openmmlab python=3.8 -y

:: Activate conda environment
echo Activating conda environment...
call conda activate openmmlab

:: Verify activation
echo Environment activated successfully. Current environment: %CONDA_DEFAULT_ENV%
:: Install PyTorch with CUDA
echo Installing PyTorch and dependencies...
:: note conda installation did not work after conda-forge channel setup
call pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

:: Install MMPose and its dependencies
echo Installing MMPose and dependencies...
call pip install -U openmim
call conda install fsspec -c conda-forge -y

call pip install mmengine
call mim install "mmcv==2.1.0"
call mim install "mmdet>=3.1.0"
call mim install "mmpretrain>=1.0.0rc8"

echo Navigate Inside to mmpose directory
cd .\nicetoolbox\detectors\third_party\mmpose\mmpose

echo Installing requirements from MMPose...
call pip install -r requirements.txt
call pip install -e .

echo Installing additional dependencies...
call conda install -c conda-forge pyparsing -y
call conda install -c conda-forge six -y
call conda install -c conda-forge toml -y

call conda deactivate
echo OPENMMLab Environment setup completed successfully.
endlocal
