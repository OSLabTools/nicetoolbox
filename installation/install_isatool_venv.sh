#!/bin/bash

# Stop on error
set -e

python3.10 -m venv ../envs
source ../envs/bin/activate
pip install -e ../../oslab_utils
pip install -r ../requirements.txt
deactivate
echo "ISA-Tool Environment setup completed successfully."