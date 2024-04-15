#!/bin/bash

# Ensure all scripts are executable
chmod +x install_isatool_venv.sh
chmod +x install_xgaze_venv.sh
chmod +x install_openmmlab_conda.sh

echo "Setting up Isa-tool environment..."
./install_isatool_venv.sh

echo "Setting up XGaze environment..."
./install_xgaze_venv.sh

echo "Setting up OpenMMlab conda environment..."
./install_openmmlab_conda.sh

echo "All environments have been set up."
