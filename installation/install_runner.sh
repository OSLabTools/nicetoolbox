#!/bin/bash

# Stop on error
set -e

###ISA Tool INSTALLATION###
echo "ISA-Tool Environment"
cd $CI_PROJECT_DIR
echo $CI_PROJECT_DIR
python3.10 -m venv envs
source envs/bin/activate
cd oslab_utils
pip install .
cd $CI_PROJECT_DIR
pip install -r requirements.txt
deactivate
echo "ISA-Tool Environment setup completed successfully."

###GAZE DETECTOR INSTALLATION###
echo "Setting up Python virtual environment for third_party/xgaze_3cams..."
python3.10 -m venv third_party/xgaze_3cams/env
source third_party/xgaze_3cams/env/bin/activate
#install correct torch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118


echo "Installing requirements for third_party/xgaze_3cams..."
python -m pip install -r third_party/xgaze_3cams/requirements.txt
deactivate
echo "XGaze Environment setup completed successfully."



###OPENMMLAB INSTALLATION###
# Create a conda environment
echo "Creating conda environment..."
cd $CI_PROJECT_DIR
conda create --name openmmlab_runner python=3.8 -y
# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"  # This line is crucial for conda activation to work in scripts
conda activate openmmlab_runner
# Install PyTorch with CUDA
echo "Installing PyTorch and dependencies..."
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y   ## need to specify the pytorch version because mmcv does not work with 2.2.0 (which is most updated version
# Install MMPose and its dependencies
echo "Installing MMPose and dependencies..."
pip install -U openmim
pip install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpretrain>=1.0.0rc8"  # required for Vitpose
echo "Navigate Inside to mmpose directory"
cd third_party/mmpose/mmpose
echo "Installing requirements from MMPose..."
pip install -r requirements.txt
pip install .
echo "Installing additional dependencies..."
conda install -c conda-forge pyparsing -y
conda install -c conda-forge six -y
conda install -c conda-forge toml -y
pip install h5py
conda deactivate

echo " OPENMMLab Environment setup completed successfully."
