#!/bin/bash

# Stop on error
set -e

# This script is called from the Makefile in the root directory (pwd = repo root).
CONDA_ENV_PATH="./envs/openmmlab"

# Initializing conda
echo "Initializing conda..."
conda init

###OPENMMLAB INSTALLATION###
# Create a conda environment
echo "Creating conda environment at $CONDA_ENV_PATH..."
conda create -p "$CONDA_ENV_PATH" python=3.8 -y

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"  # This line is crucial for conda activation to work in scripts
conda activate "$CONDA_ENV_PATH"

# Install PyTorch with CUDA
echo "Installing PyTorch and dependencies..."
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install nicetoolbox-core
echo "Installing nicetoolbox-core dependencies..."
pip install -e ./nicetoolbox_core

# Install MMPose and its dependencies
echo "Installing MMPose and dependencies..."
pip install -U openmim
conda install fsspec -c conda-forge -y
pip install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet>=3.1.0"
mim install "mmpretrain>=1.0.0rc8"  # required for Vitpose
echo "Installing MMPose from the OSLabTools fork..."
pip install -r ./nicetoolbox/detectors/method_detectors/mmpose/mmpose_requirements.txt

# Install additional dependencies required for nicetoolbox inference scripts
echo "Installing additional dependencies..."
conda install -c conda-forge pyparsing six toml -y

# TODO: Is this still a thing?
# needed on Caro's machine: downgrade protobuf
python -m pip install protobuf==3.20.3

# Finalize
conda deactivate
echo "OPENMMLab Environment setup completed successfully."
