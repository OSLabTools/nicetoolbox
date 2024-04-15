#!/bin/bash

# Stop on error
set -e

###OISA-TOOL INSTALLATION###
#git clone https://gitlab.tuebingen.mpg.de/cschmitt/isa-tool.git
#git clone https://gitlab.tuebingen.mpg.de/cschmitt/oslab_utils.git
#cd isa-tool
#sudo apt install python3.10-venv
python3.10 -m venv ../envs
source ../envs/bin/activate
pip install -e ../../oslab_utils
pip install -r ../requirements.txt
deactivate
echo "ISA-Tool Environment setup completed successfully."