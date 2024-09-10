#!/bin/bash

# Stop on error
set -e

cd "$(dirname "$0")/.."

python3.10 -m venv ./env
source ./env/bin/activate
pip install -r ./installation/requirements.txt
deactivate
echo "NICE Toolbox environment setup completed successfully."