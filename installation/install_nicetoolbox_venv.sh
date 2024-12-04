#!/bin/bash

# Stop on error
set -e

cd "$(dirname "$0")/.."

python3.10 -m venv ./envs/nicetoolbox
source ./envs/nicetoolbox/bin/activate
pip install -r ./installation/requirements.txt
deactivate
echo "NICE Toolbox environment setup completed successfully."
