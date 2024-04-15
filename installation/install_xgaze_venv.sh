#!/bin/bash

# Stop on error
set -e

###GAZE DETECTOR INSTALLATION###
echo "Setting up Python virtual environment for third_party/xgaze_3cams..."
python3.10 -m venv ../third_party/xgaze_3cams/env
source ../third_party/xgaze_3cams/env/bin/activate

# Install dependencies in the virtual environment
echo "Installing requirements for third_party/xgaze_3cams..."
python -m pip install -r ../third_party/xgaze_3cams/requirements.txt

echo "XGaze Environment setup completed successfully."