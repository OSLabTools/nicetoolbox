# Makefile

# Define variables
TOOL_NAME = nicetoolbox
VENV = nicetoolbox
VENV_ROOT_DIR = ./envs
VENV_DIR = $(VENV_ROOT_DIR)/$(VENV)
DEV = false
# `make all DEV=true` — editable install. Many people type `dev=TRUE`; forward that to DEV.
ifneq ($(strip $(dev)),)
	DEV := $(dev)
endif
MACHINE_SPECIFICS = machine_specific_paths.toml
PROJECT_CONFIG = nice_project.toml
.DEFAULT_GOAL := install_full

# Define variables for third party venvs
ifeq ($(OS), Windows_NT)
	PYTHON_EXE = python
	CONDA_DIR := $(shell conda info --base | tr '\\\\' '/')
	MMPOSE = ./nicetoolbox/detectors/method_detectors/mmpose/install_openmmlab_conda.bat
	VENV_EXE_DIR = $(VENV_DIR)/Scripts
	ETH_XGAZE_EXE_DIR = ./envs/eth_xgaze/Scripts
	PYFEAT_EXE_DIR = ./envs/py_feat/Scripts
	SPIGA_EXE_DIR = ./envs/spiga/Scripts
	INSIGHT_FACE_EXE_DIR = ./envs/insight_face/Scripts
	WHISPERX_EXE_DIR = ./envs/whisperx/Scripts
	SAM3D_BODY_EXE_DIR = $(VENV_ROOT_DIR)/sam_3d_body/Scripts
	CRISPER_WHISPER_EXE_DIR = ./envs/crisper_whisper/Scripts
	UNIGAZE_EXE_DIR = ./envs/unigaze/Scripts
else
	PYTHON_EXE = python3.10
	CONDA_DIR := $(shell conda info --base)
	MMPOSE = ./nicetoolbox/detectors/method_detectors/mmpose/install_openmmlab_conda.sh
	VENV_EXE_DIR = $(VENV_DIR)/bin
	ETH_XGAZE_EXE_DIR = ./envs/eth_xgaze/bin
	PYFEAT_EXE_DIR = ./envs/py_feat/bin
	SPIGA_EXE_DIR = ./envs/spiga/bin
	INSIGHT_FACE_EXE_DIR = ./envs/insight_face/bin
	WHISPERX_EXE_DIR = ./envs/whisperx/bin
	SAM3D_BODY_EXE_DIR = $(VENV_ROOT_DIR)/sam_3d_body/bin
	CRISPER_WHISPER_EXE_DIR = ./envs/crisper_whisper/bin
	UNIGAZE_EXE_DIR = ./envs/unigaze/bin
endif

# Download data variables
EXAMPLE_DATASET = communication_multiview
ASSETS = assets

CONFIGS_DIR = <project_folder_path>/configs
OUTPUTS_DIR = ../outputs
DATASETS_DIR = ../datasets
ASSETS_DIR = nicetoolbox/detectors

EXAMPLE_DATASET_URL = https://keeper.mpdl.mpg.de/seafhttp/f/bbf0b44a2df34af685af/?op=view


# -----------------------------------
# Minimalistic setup
# -----------------------------------
.PHONY: install
install: create_machine_specifics create_project install_nicetoolbox_venv download_dataset
	@make create_separator
	@echo "Core NICE Toolbox installed. Third-party detectors are NOT installed."
	@echo "  Install all:  make install_all_detectors"
	@echo "  Install one:  make install_eth_xgaze   (see the Makefile for the full list)"

# -----------------------------------
# Install all toolbox third-party detectors + download models
# -----------------------------------
.PHONY: install_full
install_full: install install_all_detectors download_assets
	@make create_separator
	@echo "NICE Toolbox and all detectors installed."

# ------------------------
# Clean a specific virtual environment
# Usage: make clean_venv NAME=<venv_name>
# ------------------------
.PHONY: clean_venv
clean_venv:
	@if [ -z "$(NAME)" ]; then echo "Usage: make clean_venv NAME=<venv_name>"; exit 1; fi
	@echo "Deleting virtual environment $(VENV_ROOT_DIR)/$(NAME)."
	@rm -rf $(VENV_ROOT_DIR)/$(NAME)

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
ifeq ($(OS), Windows_NT)
	@echo "# Where to find your conda (miniconda or anaconda) installation as absolute path (str)" > $(MACHINE_SPECIFICS)
	@echo "conda_path = '$(CONDA_DIR)'" >> $(MACHINE_SPECIFICS)
	@echo "" >> $(MACHINE_SPECIFICS)
	@echo "# Optional Hugging Face token for gated Hub models (e.g. sam_3d_body). Leave empty if unused." >> $(MACHINE_SPECIFICS)
	@echo "hugging_face_token = ''" >> $(MACHINE_SPECIFICS)
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
	echo "# Where to find your conda (miniconda or anaconda) installation as absolute path (str)" > $(MACHINE_SPECIFICS); \
	echo "conda_path = '$$(realpath $$VALID_CONDA_PATH/..)'">> $(MACHINE_SPECIFICS); \
	echo "" >> $(MACHINE_SPECIFICS); \
	echo "# Optional Hugging Face token for gated Hub models (e.g. sam_3d_body). Leave empty if unused." >> $(MACHINE_SPECIFICS); \
	echo "hugging_face_token = ''" >> $(MACHINE_SPECIFICS); \
	echo "Using conda installation at: $$VALID_CONDA_PATH"; \
	echo "Created machine specifics paths file"
endif

# --------------------
# Create project config
# --------------------
create_project: $(PROJECT_CONFIG)

$(PROJECT_CONFIG):
	@make create_separator
	@echo "# Project-specific paths configuration." > $(PROJECT_CONFIG)
	@echo "# Use <project_folder_path> to reference paths relative to this file's folder." >> $(PROJECT_CONFIG)
	@echo "" >> $(PROJECT_CONFIG)
	@echo "# Path to the directory in which all configuration files are stored" >> $(PROJECT_CONFIG)
	@echo "configs_folder_path = '$(CONFIGS_DIR)'" >> $(PROJECT_CONFIG)
	@echo "" >> $(PROJECT_CONFIG)
	@echo "# Path to the directory in which all datasets are stored" >> $(PROJECT_CONFIG)
	@echo "datasets_folder_path = '$(DATASETS_DIR)'" >> $(PROJECT_CONFIG)
	@echo "" >> $(PROJECT_CONFIG)
	@echo "# Directory for saving toolbox output" >> $(PROJECT_CONFIG)
	@echo "output_folder_path = '$(OUTPUTS_DIR)'" >> $(PROJECT_CONFIG)
	@echo "Created project config file"

# ----------------------
# Download keeper assets
# ----------------------
# Smart download based on the run file
.PHONY: download_assets
download_assets:
	@make create_separator
	@echo "Running AssetManager to verify and download required models..."
	@$(VENV_EXE_DIR)/download_assets

# Download everything
.PHONY: download_all_assets
download_all_assets:
	@make create_separator
	@echo "Running AssetManager to download ALL available models..."
	@$(VENV_EXE_DIR)/download_assets --all
	
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

# Install nicetoolbox venv
.PHONY: install_nicetoolbox_venv
install_nicetoolbox_venv:
#	create virtual environment
	@make create_separator
	@make clean_venv NAME=$(VENV)

	@echo "Creating virtual environment in $(VENV_DIR)..."
	@$(PYTHON_EXE) -m venv $(VENV_DIR)

#	install nicetoolbox-core
	@echo "Installing nicetoolbox-core dependencies..."
	@$(VENV_EXE_DIR)/pip install -e ./nicetoolbox_core

ifeq ($(DEV), false)
#	basic installation
	@echo "Installing $(TOOL_NAME)..."
	@$(VENV_EXE_DIR)/pip install .
else
#	developer installation
	@echo "Installing $(TOOL_NAME) editable for developers..."
	@$(VENV_EXE_DIR)/pip install -e ".[dev]"
endif
	@echo "$(TOOL_NAME) installed in $(VENV_DIR) successfully."


# -----------------------------------
# Install all third-party detectors
#
# These are not maintained by the NICE Toolbox authors and are provided as-is by
# their respective owners, under their own licenses. See LICENSES_ALGORITHMS.md.
# -----------------------------------
.PHONY: install_all_detectors
install_all_detectors:
# 	detectors venv installations
	-@make install_eth_xgaze
	-@make install_unigaze
	-@make install_spiga
	-@make install_insight_face
	-@make install_whisperx
	-@make install_sam3d_body
	-@make install_crisper_whisper
# 	detectors conda installations
	-@make install_pyfeat
	-@make install_mmpose
	@make create_separator
	@echo "Detectors installation finished."

# Install the venv for eth-xgaze
.PHONY: install_eth_xgaze
install_eth_xgaze:
	@make create_separator
	@make clean_venv NAME=eth_xgaze
	@echo "Creating virtual environment for submodule 'ETH-XGaze'..."
	@$(PYTHON_EXE) -m venv ./envs/eth_xgaze
	@echo "Virtual environment created in ./envs/eth_xgaze"

	@echo "Installing requirements for 'ETH-XGaze'..."
	@$(ETH_XGAZE_EXE_DIR)/pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
	@$(ETH_XGAZE_EXE_DIR)/pip install -r ./nicetoolbox/detectors/method_detectors/eth_xgaze/eth_xgaze_requirements.txt
	@$(ETH_XGAZE_EXE_DIR)/pip install -e ./nicetoolbox_core

	@echo "ETH-XGaze' environment setup completed successfully."

# Install the conda env for pyfeat
.PHONY: install_pyfeat
install_pyfeat:
	@make create_separator
	@make clean_venv NAME=py_feat
	@echo "Installing conda environment for algorithm 'Py-Feat'..."
# Py-Feat v2 requires Python 3.11+, so we use conda
	@echo "Creating conda environment..."
	@conda create -p ./envs/py_feat python=3.11 -y
	@echo "Conda environment created in ./envs/py_feat"

	@echo "Installing requirements for 'Py-Feat'..."
	@$(PYFEAT_EXE_DIR)/pip install --no-warn-script-location torch==2.11.0+cu126 torchvision==0.26.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
# py-feat requires torchcodec>=0.11 and its linux wheels link against CUDA 13
# but we are still CUDA 12 and we don't even use it, so lets fix it to cpu
	@$(PYFEAT_EXE_DIR)/pip install --no-warn-script-location torchcodec==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
	@$(PYFEAT_EXE_DIR)/pip install --no-warn-script-location -r ./nicetoolbox/detectors/method_detectors/py_feat/py_feat_requirements.txt
	@$(PYFEAT_EXE_DIR)/pip install -e ./nicetoolbox_core
	@echo "'Py-Feat' environment setup completed successfully."

.PHONY: install_spiga
install_spiga:
	@make create_separator
	@make clean_venv NAME=spiga
	@echo "Installing virtual environment for algorithm 'SPIGA'..."

	@echo "Creating virtual environment..."
	@$(PYTHON_EXE) -m venv ./envs/spiga
	@echo "Virtual environment created in ./envs/spiga"

	@echo "Installing requirements for 'SPIGA'..."
	@$(SPIGA_EXE_DIR)/pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
	@$(SPIGA_EXE_DIR)/pip install -r ./nicetoolbox/detectors/method_detectors/spiga/spiga_requirements.txt
	@$(SPIGA_EXE_DIR)/pip install -e ./nicetoolbox_core
	@echo "'SPIGA' environment setup completed successfully."


# Install the venv for InsightFace
.PHONY: install_insight_face
install_insight_face:
	@make create_separator
	@make clean_venv NAME=insight_face
	@echo "Installing virtual environment for algorithm 'InsightFace'..."

	@echo "Creating virtual environment..."
	@$(PYTHON_EXE) -m venv ./envs/insight_face
	@$(INSIGHT_FACE_EXE_DIR)/python -m pip install --upgrade pip
	@echo "Virtual environment created in ./envs/insight_face"

# 	need bundled cuda and cudnn for onnxruntime to use GPU
	@echo "Installing Pytorch..."
	@$(INSIGHT_FACE_EXE_DIR)/pip install torch==2.9.0+cu129 --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple
	@echo "Installing requirements for 'InsightFace'..."
	@$(INSIGHT_FACE_EXE_DIR)/pip install -r ./nicetoolbox/detectors/method_detectors/insight_face/insight_face_requirements.txt

# 	need this hack to make gpu work - delete cpu version of onnx, keep only cuda
	@$(INSIGHT_FACE_EXE_DIR)/pip uninstall -y onnxruntime
	@$(INSIGHT_FACE_EXE_DIR)/pip install --no-cache-dir onnxruntime-gpu==1.23.2

	@$(INSIGHT_FACE_EXE_DIR)/pip install -e ./nicetoolbox_core
	@echo "'InsightFace' environment setup completed successfully."


# Install the venv for whisperx
.PHONY: install_whisperx
install_whisperx:
	@make create_separator
	@make clean_venv NAME=whisperx
	@echo "Installing virtual environment for algorithm 'WhisperX'..."

	@echo "Creating virtual environment..."
	@$(PYTHON_EXE) -m venv ./envs/whisperx
	@echo "Virtual environment created in ./envs/whisperx"

	@echo "Installing requirements for 'WhisperX'..."
	@$(WHISPERX_EXE_DIR)/pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
# 	whisperx alignment needs the nltk punkt tokenizer
	@$(WHISPERX_EXE_DIR)/python -m nltk.downloader -d ./envs/whisperx/nltk_data punkt_tab
	@$(WHISPERX_EXE_DIR)/pip install -r ./nicetoolbox/detectors/method_detectors/whisperx/whisperx_requirements.txt
	@$(WHISPERX_EXE_DIR)/pip install -e ./nicetoolbox_core
	@echo "'WhisperX' environment setup completed successfully."

# Install the venv for crisper-whisper
.PHONY: install_crisper_whisper
install_crisper_whisper:
	@make create_separator
	@make clean_venv NAME=crisper_whisper
	@echo "Installing virtual environment for algorithm 'CrisperWhisper'..."

	@echo "Creating virtual environment..."
	@$(PYTHON_EXE) -m venv ./envs/crisper_whisper
	@echo "Virtual environment created in ./envs/crisper_whisper"

	@echo "Installing requirements for 'CrisperWhisper'..."
	@$(CRISPER_WHISPER_EXE_DIR)/pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
	@$(CRISPER_WHISPER_EXE_DIR)/pip install -r ./nicetoolbox/detectors/method_detectors/crisper_whisper/crisper_whisper_requirements.txt
	@$(CRISPER_WHISPER_EXE_DIR)/pip install -e ./nicetoolbox_core
	@echo "'CrisperWhisper' environment setup completed successfully."


# Install the venv for mmpose
.PHONY: install_mmpose
install_mmpose:
	@make create_separator
	@make clean_venv NAME=openmmlab
	@echo "Installing Conda environment for submodule 'MMPose'..."
ifeq ($(OS), Windows_NT)
	@bash -c "$(MMPOSE)"
else
	@chmod +x $(MMPOSE) && $(MMPOSE)
endif
	@echo "'MMPose' environment setup completed successfully."

# SAM 3D repository path
SAM3D_BODY_REPO_DIR = $(VENV_ROOT_DIR)/sam_3d_body/src/sam-3d-body

.PHONY: install_sam3d_body
install_sam3d_body:
	@make create_separator
	@make clean_venv NAME=sam_3d_body
	@echo "Creating SAM 3D Body venv at sam_3d_body ..."
	@$(PYTHON_EXE) -m venv ./envs/sam_3d_body
	@echo "Cloning SAM 3D Body fork..."
	@git clone https://github.com/OSLabTools/sam-3d-body $(SAM3D_BODY_REPO_DIR)
	@git -C $(SAM3D_BODY_REPO_DIR) checkout --detach b5c765a0d89d789985e186d396315e7590887b94
	@echo "Installing PyTorch (2.8.0, cu129)..."
	@$(SAM3D_BODY_EXE_DIR)/pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple
	@echo "Installing SAM 3D Body dependencies..."
	@$(SAM3D_BODY_EXE_DIR)/pip install -r nicetoolbox/detectors/method_detectors/sam_3d_body/sam_3d_body_pip_requirements.txt
	@echo "Installing Detectron2..."
	@$(SAM3D_BODY_EXE_DIR)/pip install "detectron2==0.6+fd27788pt2.8.0cu129" --extra-index-url https://miropsota.github.io/torch_packages_builder --no-deps
	@$(SAM3D_BODY_EXE_DIR)/pip install -e ./nicetoolbox_core
	@echo "'SAM 3D Body' environment setup completed successfully."

# Install the venv for UniGaze
.PHONY: install_unigaze
install_unigaze:
	@make create_separator
	@make clean_venv NAME=unigaze
	@echo "Creating virtual environment for algorithm 'UniGaze'..."
	@$(PYTHON_EXE) -m venv ./envs/unigaze
	@echo "Virtual environment created in ./envs/unigaze"

	@echo "Installing requirements for 'UniGaze'..."
	@$(UNIGAZE_EXE_DIR)/pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
	@$(UNIGAZE_EXE_DIR)/pip install -r ./nicetoolbox/detectors/method_detectors/unigaze/unigaze_requirements.txt
	@$(UNIGAZE_EXE_DIR)/pip install -e ./nicetoolbox_core

	@echo "UniGaze environment setup completed successfully."