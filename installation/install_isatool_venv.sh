#!/bin/bash

# Stop on error
set -e

cd $CI_PROJECT_DIR

python3.10 -m venv ./env
source ./env/bin/activate
pip install -r ./installation/requirements.txt
deactivate
echo "ISA-Tool Environment setup completed successfully."