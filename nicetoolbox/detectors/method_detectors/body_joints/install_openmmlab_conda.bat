@echo off

:: Stop on error
setlocal enableextensions enabledelayedexpansion

:: Initializing conda environment
:: It should be done at least once after conda installation
echo Initializing conda...
call conda init

:: OPENMMLAB INSTALLATION 

:: Create a conda environment
echo Creating conda environment...
call conda create --name openmmlab python=3.8 -y

:: Activate conda environment
echo Activating conda environment...
call conda activate openmmlab

:: Install PyTorch with CUDA
:: note conda installation did not work after conda-forge channel setup
echo Installing PyTorch and dependencies...
call pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

:: Install nicetoolbox-core
:: note: this script is called from the Makefile in the root directory
echo Installing nicetoolbox-core dependencies...
call pip install -e ./nicetoolbox_core

:: Install MMPose and its dependencies
echo Installing MMPose and dependencies...
call pip install -U openmim
call conda install fsspec -c conda-forge -y
call pip install mmengine
call mim install "mmcv==2.1.0"
call mim install "mmdet>=3.1.0"
call mim install "mmpretrain>=1.0.0rc8" 
echo Navigate Inside to mmpose directory
cd ./submodules/mmpose/
echo Installing requirements from MMPose...
call pip install -r requirements.txt
call pip install -e .

:: Install additional dependencies required for nicetoolbox inference scripts
echo Installing additional dependencies...
call conda install -c conda-forge pyparsing -y
call conda install -c conda-forge six -y
call conda install -c conda-forge toml -y

:: TODO: Is this still a thing?
:: needed on Caro's machine: downgrade protobuf
call pip install protobuf==3.20.3

:: Finalize
call conda deactivate
echo OPENMMLab Environment setup completed successfully.
endlocal
