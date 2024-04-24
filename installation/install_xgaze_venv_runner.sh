#!/bin/bash

# Stop on error
set -e

###GAZE DETECTOR INSTALLATION###
echo "Setting up Python virtual environment for third_party/xgaze_3cams..."
echo $PWD
cd $CI_PROJECT_DIR
python3.10 -m venv third_party/xgaze_3cams/env
source third_party/xgaze_3cams/env/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies in the virtual environment
echo "Installing requirements for third_party/xgaze_3cams..."
python -m pip install -r third_party/xgaze_3cams/requirements.txt

echo "XGaze Environment setup completed successfully."