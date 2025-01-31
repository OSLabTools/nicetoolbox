# Makefile

# Define variables
TOOL_NAME = nicetoolbox
VENV = nicetoolbox
VENV_DIR = ./envs/$(VENV)
DEV = false
CONDA_DIR = $(shell conda info --base)
MMPOSE = ./nicetoolbox/detectors/method_detectors/body_joints/install_openmmlab_conda.sh
MACHINE_SPECIFICS = machine_specific_paths.toml


# Download data variables
EXAMPLE_DATASET = communication_multiview
ASSETS = assets

OUTPUTS_DIR = ../outputs
DATASETS_DIR = ../datasets
ASSETS_DIR = nicetoolbox/detectors

EXAMPLE_DATASET_URL = https://keeper.mpdl.mpg.de/f/b1b91c42fcf849fb90a1/?dl=1
ASSETS_URL = https://keeper.mpdl.mpg.de/f/d11765550fc349649346/?dl=1


# -----------------------------------
# Full setup: installation + download
# -----------------------------------
.PHONY: all
all: create_machine_specifics download_assets download_dataset install

# ------------------------
# Clean up an installation
# ------------------------
.PHONY: clean
clean:
#	@echo "Cleaning pycache."
#	@rm -rf __pycache__
	@echo "Deleting virtual environment $(VENV_DIR)."
	@rm -rf $(VENV_DIR)

# ------------------------
# Create machine specifics
# ------------------------
create_machine_specifics: $(MACHINE_SPECIFICS)

$(MACHINE_SPECIFICS):
	@touch $(MACHINE_SPECIFICS)
	@echo "# Absolute path to the directory in which all datasets are stored (str)" > $(MACHINE_SPECIFICS)
	@echo "datasets_folder_path = '$(DATASETS_DIR)'" >> $(MACHINE_SPECIFICS)
	@echo "" >> $(MACHINE_SPECIFICS)
	@echo "# Directory for saving toolbox output as an absolute path (str)" >> $(MACHINE_SPECIFICS)
	@echo "output_folder_path = '$(OUTPUTS_DIR)'" >> $(MACHINE_SPECIFICS)
	@echo "" >> $(MACHINE_SPECIFICS)
	@echo "# Where to find your conda (miniconda or anaconda) installation as absolute path (str)" >> $(MACHINE_SPECIFICS)
	@echo "conda_path = '$(CONDA_DIR)'" >> $(MACHINE_SPECIFICS)
	@echo "Created machine specifics paths file"

# ----------------------
# Download keeper assets
# ----------------------
download_assets: $(ASSETS_DIR)/$(ASSETS)

$(ASSETS_DIR)/$(ASSETS):
	@echo "Downloading keeper assets..."
	@mkdir -p $(ASSETS_DIR)
	@wget --progress=bar:force $(ASSETS_URL) -O $(ASSETS).zip
	@unzip $(ASSETS).zip -d $(ASSETS_DIR)
	@rm $(ASSETS).zip
	@echo "Checkpoint files downloaded to $(ASSETS_DIR)/$(ASSETS)"

# -----------------------
# Download keeper example
# -----------------------
download_dataset: $(DATASETS_DIR)/$(EXAMPLE_DATASET)

$(DATASETS_DIR)/$(EXAMPLE_DATASET):
	@echo "Downloading keeper example dataset..."
	@mkdir -p $(DATASETS_DIR)
	@wget --progress=bar:force $(EXAMPLE_DATASET_URL) -O $(EXAMPLE_DATASET).zip
	@unzip $(EXAMPLE_DATASET).zip -d $(DATASETS_DIR)
	@rm $(EXAMPLE_DATASET).zip
	@echo "Example dataset downloaded to $(DATASETS_DIR)/$(EXAMPLE_DATASET)."

# -------------------
# Install nicetoolbox
# -------------------
install: $(VENV_DIR)/bin/activate

#	Install xgaze if not already installed
ifeq ("$(wildcard ./envs/xgaze_3cams/bin/activate)","")
	@make install_xgaze
endif

#	Install pyfeat if not already installed
ifeq ("$(wildcard ./envs/py_feat/bin/activate)","")
	@make install_pyfeat
endif

#	 check for conda installation
ifeq ($(which conda),"")
	@echo "No CONDA installation found. Check the documentation for instructions: https://nicetoolbox.readthedocs.io/en/docs/installation.html."
else
	@$(eval CONDA_DIR=$(CONDA_DIR))

#	Install mmpose if not already installed
ifeq ("$(wildcard $(CONDA_DIR)/envs/openmmlab)", "")
	@make install_mmpose
endif
endif


# Install the virtual environment
$(VENV_DIR)/bin/activate: pyproject.toml
#	start clean
	@make clean

#	create virtual environment
	@echo "Creating virtual environment in $(VENV_DIR)..."
	@python3.10 -m venv $(VENV_DIR)
	@echo "Virtual environment created in $(VENV_DIR)"
	
ifeq ($(DEV), false)
# 	basic installation
	@echo "Installing $(TOOL_NAME)..."
	@$(VENV_DIR)/bin/pip install .
else
# 	developer installation
	@echo "Installing $(TOOL_NAME) editable for developers..."
#   torch is required for the evaluation (part of optional dependencies)
	@$(VENV_DIR)/bin/pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
	@$(VENV_DIR)/bin/pip install -e ".[dev,eval,visual]"
endif
	@echo "$(TOOL_NAME) installed in $(VENV_DIR) successfully."


# Install the venv for xgaze
.PHONY: install_xgaze
install_xgaze:
	@echo "Installing virtual environment for third_party code 'XGaze'..."

	@echo "Creating virtual environment..."
	@python3.10 -m venv ./envs/xgaze_3cams
	@echo "Virtual environment created in ./envs/xgaze_3cams"

	@echo "Installing requirements for 'XGaze'..."
	@./envs/xgaze_3cams/bin/pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
	@./envs/xgaze_3cams/bin/pip install -r ./submodules/ETH-XGaze/requirment.txt  # codespell:ignore requirment
	@./envs/xgaze_3cams/bin/pip install tensorboard h5py
	@echo "'XGaze' environment setup completed successfully."


# Install the venv for pyfeat
.PHONY: install_pyfeat
install_pyfeat:
	@echo "Installing virtual environment for third_party code 'Py-Feat'..."

	@echo "Creating virtual environment..."
	@python3.10 -m venv ./envs/py_feat
	@echo "Virtual environment created in ./envs/py_feat"

	@echo "Installing requirements for 'Py-Feat'..."
	@./envs/py_feat/bin/pip install torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
	@./envs/py_feat/bin/pip install py-feat
	@./envs/py_feat/bin/pip install -r ./nicetoolbox/detectors/method_detectors/emotion_individual/py_feat_requirements.txt
	@echo "'Py-Feat' environment setup completed successfully."


# Install the venv for mmpose
.PHONY: install_mmpose
install_mmpose:
	@echo "Installing virtual environment for third_party code 'MMPose'..."
	@chmod +x $(MMPOSE) && $(MMPOSE)
	@echo "'MMPose' environment setup completed successfully."
