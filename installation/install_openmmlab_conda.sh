#!/bin/bash

# Stop on error
set -e

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
#conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch  ##cpu only version
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y   ## need to specify the pytorch version because mmcv does not work with 2.2.0 (which is most updated version
# Install MMPose and its dependencies
echo "Installing MMPose and dependencies..."
pip install -U openmim
pip install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpretrain>=1.0.0rc8"  # required for Vitpose
echo "Navigate Inside to mmpose directory"
cd ../detectors/third_party/mmpose/mmpose
echo "Installing requirements from MMPose..."
pip install -r requirements.txt
pip install -e .
echo "Installing additional dependencies..."
conda install -c conda-forge pyparsing -y
conda install -c conda-forge six -y
conda install -c conda-forge toml -y
conda deactivate
echo " OPENMMLab Environment setup completed successfully."
