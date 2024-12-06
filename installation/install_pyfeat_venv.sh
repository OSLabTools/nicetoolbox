#!/bin/bash

# Stop on error
set -e

cd "$(dirname "$0")/.."

###PYFEAT DETECTOR INSTALLATION###
echo "Setting up Python virtual environment for third_party/py_feat..."
python3 -m venv ./envs/py_feat
source ./envs/py_feat/bin/activate

# Install dependencies in the virtual environment
echo "Installing requirements for third_party/py_feat..."
python3 -m pip install torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install py-feat
python3 -m pip install -r ./nicetoolbox/detectors/third_party/py_feat/requirements.txt

echo "Py-Feat Environment setup completed successfully."