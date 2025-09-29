#!/bin/bash

# Stop on error
set -e

# Initializing conda enviorment
# It should be done at least once after conda installation
echo "Initializing conda..."
conda init

###OPENMMLAB INSTALLATION###
# Create a conda environment
echo "Creating conda environment..."
conda create --name openmmlab python=3.8 -y
# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"  # This line is crucial for conda activation to work in scripts
conda activate openmmlab
# Install PyTorch with CUDA
echo "Installing PyTorch and dependencies..."
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# # Install MMPose and its dependencies
echo "Installing MMPose and dependencies..."
pip install -U openmim
conda install fsspec -c conda-forge -y
pip install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet>=3.1.0"
mim install "mmpretrain>=1.0.0rc8"  # required for Vitpose
echo "Navigate Inside to mmpose directory"
cd ./submodules/mmpose/
echo "Installing requirements from MMPose..."
pip install -r requirements.txt
pip install -e .
echo "Installing additional dependencies..."
conda install -c conda-forge pyparsing six toml -y
# needed on Caro's machine: downgrade protobuf
python -m pip install protobuf==3.20.3
conda deactivate
echo " OPENMMLab Environment setup completed successfully."
