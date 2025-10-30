# Makefile

# Define variables
TOOL_NAME = nicetoolbox
VENV = nicetoolbox
VENV_ROOT_DIR = ./envs
VENV_DIR = $(VENV_ROOT_DIR)/$(VENV)
DEV = false
MACHINE_SPECIFICS = machine_specific_paths.toml

# Define variables for third party venvs
ifeq ($(OS), Windows_NT)
	PYTHON_EXE = python
    CONDA_DIR := $(shell conda info --base | tr '\\\\' '/')
	MMPOSE = ./nicetoolbox/detectors/method_detectors/body_joints/install_openmmlab_conda.bat
	VENV_EXE_DIR = $(VENV_DIR)/Scripts
	MULTIVIEW_XGAZE_EXE_DIR = ./envs/multiview_eth_xgaze/Scripts
	PYFEAT_EXE_DIR = ./envs/py_feat/Scripts
	SPIGA_EXE_DIR = ./envs/spiga/Scripts
else
	PYTHON_EXE = python3.10
    CONDA_DIR := $(shell conda info --base)
	MMPOSE = ./nicetoolbox/detectors/method_detectors/body_joints/install_openmmlab_conda.sh
	VENV_EXE_DIR = $(VENV_DIR)/bin
	MULTIVIEW_XGAZE_EXE_DIR = ./envs/multiview_eth_xgaze/bin
	PYFEAT_EXE_DIR = ./envs/py_feat/bin
	SPIGA_EXE_DIR = ./envs/spiga/bin
endif


# Download data variables
EXAMPLE_DATASET = communication_multiview
ASSETS = assets

OUTPUTS_DIR = ../outputs
DATASETS_DIR = ../datasets
ASSETS_DIR = nicetoolbox/detectors

EXAMPLE_DATASET_URL = https://keeper.mpdl.mpg.de/f/9b2b44339a5d48a2a61f/?dl=1
ASSETS_URL = https://keeper.mpdl.mpg.de/f/6f6d3a030e514ae4b973/?dl=1


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
# Clean all virtual environments
# ------------------------
.PHONY: clean_all
clean_all:
	@echo "Deleting all virtual environments from $(VENV_ROOT_DIR)."
	@rm -rf $(VENV_ROOT_DIR)

# ------------------------
# Create a separator
# ------------------------
.PHONY: create_separator
create_separator:
	@echo ""
	@echo "*********************************************"
	@echo ""

# ------------------------
# Create machine specifics
# ------------------------
create_machine_specifics: $(MACHINE_SPECIFICS)

$(MACHINE_SPECIFICS):
	@make create_separator
	@touch $(MACHINE_SPECIFICS)
	@echo "# Absolute path to the directory in which all datasets are stored (str)" > $(MACHINE_SPECIFICS)
	@echo "datasets_folder_path = '$(DATASETS_DIR)'" >> $(MACHINE_SPECIFICS)
	@echo "" >> $(MACHINE_SPECIFICS)
	@echo "# Directory for saving toolbox output as an absolute path (str)" >> $(MACHINE_SPECIFICS)
	@echo "output_folder_path = '$(OUTPUTS_DIR)'" >> $(MACHINE_SPECIFICS)
	@echo "" >> $(MACHINE_SPECIFICS)
ifeq ($(OS), Windows_NT)
	@echo "# Where to find your conda (miniconda or anaconda) installation as absolute path (str)" >> $(MACHINE_SPECIFICS)
	@echo "conda_path = '$(CONDA_DIR)'" >> $(MACHINE_SPECIFICS)
	@echo "Created machine specifics paths file"
else
	@echo "Looking for valid conda envs_dirs..."
	@VALID_CONDA_PATH=$$(conda config --show envs_dirs | grep -v '\.conda/envs' | grep -E '/envs$$' | head -n 1 | tr -d ' -'); \
	if [ -z "$$VALID_CONDA_PATH" ]; then \
		echo "Error: Only **/.conda/ installation found. Nicetoolbox requires a visible conda installation (e.g., /home/<user>/miniconda)."; \
		echo "Please reconfigure conda with:"; \
		echo "  conda config --add envs_dirs /path/to/visible/conda/installation/"; \
		exit 1; \
	fi; \
	echo "# Where to find your conda (miniconda or anaconda) installation as absolute path (str)" >> $(MACHINE_SPECIFICS); \
	echo "conda_path = '$$(realpath $$VALID_CONDA_PATH/..)'">> $(MACHINE_SPECIFICS); \
	echo "Using conda installation at: $$VALID_CONDA_PATH"; \
	echo "Created machine specifics paths file"
endif

# ----------------------
# Download keeper assets
# ----------------------
download_assets: $(ASSETS_DIR)/$(ASSETS)

$(ASSETS_DIR)/$(ASSETS):
	@make create_separator
	@echo "Downloading keeper assets..."
	@mkdir -p $(ASSETS_DIR)
ifeq ($(OS), Windows_NT)
	@curl -L -o $(ASSETS).zip $(ASSETS_URL)
else
	@wget --progress=bar:force $(ASSETS_URL) -O $(ASSETS).zip
endif
	@unzip $(ASSETS).zip -d $(ASSETS_DIR)
	@rm $(ASSETS).zip
	@echo "Checkpoint files downloaded to $(ASSETS_DIR)/$(ASSETS)"
	
# -----------------------
# Download keeper example
# -----------------------
download_dataset: $(DATASETS_DIR)/$(EXAMPLE_DATASET)

$(DATASETS_DIR)/$(EXAMPLE_DATASET):
	@make create_separator
	@echo "Downloading keeper example dataset..."
	@mkdir -p $(DATASETS_DIR)
ifeq ($(OS), Windows_NT)
	@curl -L -o $(EXAMPLE_DATASET).zip $(EXAMPLE_DATASET_URL)
else
	@wget --progress=bar:force $(EXAMPLE_DATASET_URL) -O $(EXAMPLE_DATASET).zip
endif
	@unzip $(EXAMPLE_DATASET).zip -d $(DATASETS_DIR)
	@rm $(EXAMPLE_DATASET).zip
	@echo "Example dataset downloaded to $(DATASETS_DIR)/$(EXAMPLE_DATASET)."

# -------------------
# Install nicetoolbox
# -------------------
install: $(VENV_EXE_DIR)/activate

#	Install xgaze if not already installed
ifeq ("$(wildcard $(MULTIVIEW_XGAZE_EXE_DIR)/activate)","")
	@make install_multiview_eth_xgaze
endif

#	Install pyfeat if not already installed
ifeq ("$(wildcard $(PYFEAT_EXE_DIR)/activate)","")
	@make install_pyfeat
endif

#	Install SPIGA if not already installed
ifeq ("$(wildcard $(SPIGA_EXE_DIR)/activate)","")
	@make install_spiga
endif

#	check for conda installation
ifeq ($(which conda),"")
	@echo "No CONDA installation found. Check the documentation for instructions: https://nicetoolbox.readthedocs.io/en/stable/installation.html."
else
	@$(eval CONDA_DIR=$(CONDA_DIR))

#	Install mmpose if not already installed
ifeq ("$(wildcard $(CONDA_DIR)/envs/openmmlab)", "")
	@make install_mmpose
endif
endif


# Install the virtual environment
$(VENV_EXE_DIR)/activate: pyproject.toml
#	start clean
	@make clean

#	create virtual environment
	@make create_separator
	@echo "Creating virtual environment in $(VENV_DIR)..."
	@$(PYTHON_EXE) -m venv $(VENV_DIR)
	
ifeq ($(DEV), false)
# 	basic installation
	@echo "Installing $(TOOL_NAME)..."
	@$(VENV_EXE_DIR)/pip install .
else
# 	developer installation
	@echo "Installing $(TOOL_NAME) editable for developers..."
#   torch is required for the evaluation (part of optional dependencies)
	@$(VENV_EXE_DIR)/pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
	@$(VENV_EXE_DIR)/pip install -e ".[dev,visual]"
endif
	@echo "$(TOOL_NAME) installed in $(VENV_DIR) successfully."


# Install the venv for multiview-xgaze
.PHONY: install_multiview_eth_xgaze
install_multiview_eth_xgaze:
	@make create_separator
	@echo "Creating virtual environment for submodule 'Multiview ETH-XGaze'..."
	@$(PYTHON_EXE) -m venv ./envs/multiview_eth_xgaze
	@echo "Virtual environment created in ./envs/multiview_eth_xgaze"

	@echo ""Installing requirements for 'Multiview ETH-XGaze'...""
	@$(MULTIVIEW_XGAZE_EXE_DIR)/pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
	@$(MULTIVIEW_XGAZE_EXE_DIR)/pip install submodules/multiview_eth_xgaze -c submodules/multiview_eth_xgaze/constraints.txt

	@echo "Multiview ETH-XGaze' environment setup completed successfully."

# Install the venv for pyfeat
.PHONY: install_pyfeat
install_pyfeat:
	@make create_separator
	@echo "Installing virtual environment for algorithm 'Py-Feat'..."

	@echo "Creating virtual environment..."
	@$(PYTHON_EXE) -m venv ./envs/py_feat
	@echo "Virtual environment created in ./envs/py_feat"

	@echo "Installing requirements for 'Py-Feat'..."
	@$(PYFEAT_EXE_DIR)/pip install torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
	@$(PYFEAT_EXE_DIR)/pip install py-feat
	@$(PYFEAT_EXE_DIR)/pip install -r ./nicetoolbox/detectors/method_detectors/emotion_individual/py_feat_requirements.txt
	@echo "'Py-Feat' environment setup completed successfully."

.PHONY: install_spiga
install_spiga:
	@make create_separator
	@echo "Installing virtual environment for algorithm 'SPIGA'..."

	@echo "Creating virtual environment..."
	@$(PYTHON_EXE) -m venv ./envs/spiga
	@echo "Virtual environment created in ./envs/spiga"

	@echo "Installing requirements for 'SPIGA'..."
	@$(SPIGA_EXE_DIR)/pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
	@$(SPIGA_EXE_DIR)/pip install -r ./nicetoolbox/detectors/method_detectors/head_orientation/spiga_requirements.txt
	@echo "'SPIGA' environment setup completed successfully."


# Install the venv for mmpose
.PHONY: install_mmpose
install_mmpose:
	@make create_separator
	@echo "Installing virtual environment for submodule 'MMPose'..."
ifeq ($(OS), Windows_NT)
	@bash -c "$(MMPOSE)"
else
	@chmod +x $(MMPOSE) && $(MMPOSE)
endif
	@echo "'MMPose' environment setup completed successfully."

