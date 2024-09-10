#!/bin/bash

# Stop on error
set -e

cd "$(dirname "$0")/.."

###GAZE DETECTOR INSTALLATION###
echo "Setting up Python virtual environment for third_party/xgaze_3cams..."
python3.10 -m venv ./detectors/third_party/xgaze_3cams/env
source ./detectors/third_party/xgaze_3cams/env/bin/activate

# Install dependencies in the virtual environment
echo "Installing requirements for third_party/xgaze_3cams..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r ./detectors/third_party/xgaze_3cams/requirements.txt

echo "XGaze Environment setup completed successfully."

