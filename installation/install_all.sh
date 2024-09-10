#!/bin/bash

cd "$(dirname "$0")/.."

# Ensure all scripts are executable
chmod +x ./installation/install_nicetoolbox_venv.sh
chmod +x ./installation/install_xgaze_venv.sh
chmod +x ./installation/install_openmmlab_conda.sh

echo "Setting up Nice Toolbox environment..."
./installation/install_nicetoolbox_venv.sh

echo "Setting up XGaze environment..."
./installation/install_xgaze_venv.sh

echo "Setting up OpenMMlab conda environment..."
./installation/install_openmmlab_conda.sh

echo "All environments have been set up."
