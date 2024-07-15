@echo off
:: Stop on error
setlocal enableextensions enabledelayedexpansion

:: Function to check for errors after each command
:checkError
if errorlevel 1 (
    echo An error occurred. Exiting...
    exit /b %errorlevel%
)

:: Create a conda environment
echo Creating conda environment...
conda create --name openmmlab python=3.8 -y
call :checkError

:: Activate conda environment
echo Activating conda environment...
call conda activate openmmlab
call :checkError

:: Install PyTorch with CUDA
echo Installing PyTorch and dependencies...
:: For CPU only version, uncomment the following line and comment out the CUDA version
:: conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch -y
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
call :checkError

:: Install MMPose and its dependencies
echo Installing MMPose and dependencies...
pip install -U openmim
call :checkError
pip install mmengine
call :checkError
mim install "mmcv>=2.0.1"
call :checkError
mim install "mmdet>=3.1.0"
call :checkError
mim install "mmpretrain>=1.0.0rc8"
call :checkError

echo Navigate Inside to mmpose directory
cd ..\detectors\third_party\mmpose\mmpose
call :checkError

echo Installing requirements from MMPose...
pip install -r requirements.txt
call :checkError
pip install -e .
call :checkError

echo Installing additional dependencies...
conda install -c conda-forge pyparsing -y
call :checkError
conda install -c conda-forge six -y
call :checkError
conda install -c conda-forge toml -y
call :checkError

call conda deactivate
echo OPENMMLab Environment setup completed successfully.
endlocal
