#!/bin/bash

cd $CI_PROJECT_DIR

# Ensure all scripts are executable
chmod +x ./installation/install_isatool_venv.sh
chmod +x ./installation/install_xgaze_venv.sh
chmod +x ./installation/install_openmmlab_conda.sh

echo "Setting up Isa-tool environment..."
./installation/install_isatool_venv.sh

echo "Setting up XGaze environment..."
./installation/install_xgaze_venv.sh

echo "Setting up OpenMMlab conda environment..."
./installation/install_openmmlab_conda.sh

echo "All environments have been set up."
