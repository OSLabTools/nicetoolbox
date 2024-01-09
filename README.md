# Template Detector


## Installation

1. Clone the repository and navigate into it:
```
git clone git@gitlab.tuebingen.mpg.de:cschmitt/isa-tool.git
cd isa-tool/
```
2. (Recommended) Create and activate a virtual environment and install the 
`requirements.txt`:
```
# using conda
conda create --name isa-tool python=3.10 -y
conda activate isa-tool
## install requirements.txt

# using venv
python3.10 -m venv envs/isa-tool
source envs/isa-tool/bin/activate
python -m pip install -r requirements.txt
```

3. Install additional virtual environments for the third-party codes, as 
described next.

    TODO: update `env_setup.bat` and `install.sh` scripts to do all following 
steps automated.


### Pose Detector

Create a new virtual environment, either using conda `openmmlab` or venv 
`third_party/mmpose/env`. Install the packages listed in 
`third_party/mmpose/requirements.txt`. Additionally, we need to install 
[MMPose](https://mmpose.readthedocs.io/en/latest/installation.html):

#### Installing MMPose on Windows using conda:
```
# 1. Create conda environment
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
## install requirements.txt?

# 2. Install pytorch
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install MMEngine and MMCV using MIM
pip install -U openmim
mim install mmengine
k# Install MMDetection
mim install "mmdet>=3.0.0"
mim install "mmpretrain>=1.0.0rc8"  ## required for Vitpose

# 4. Build MMPOSE from source
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .    
```
Notes: 
"-v" means verbose, or more output
"-e" means installing a project in editable mode,
thus any local modifications made to the code will take effect without reinstallation.

#### Installation on Linux using conda:
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch

# install MMPose
pip install -U openmim
pip install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
git clone https://github.com/open-mmlab/mmpose.git
cd third_party/mmpose/mmpose/
pip install -r requirements.txt 
pip install -e .

# further packages needed
conda install -c conda-forge pyparsing
conda install -c conda-forge six
conda install -c conda-forge toml
```
update Nov 2nd: before installing MMPose, I needed to install first:
```
conda install -c conda-forge webencodings
conda install -c conda-forge attrs
conda install -c conda-forge toml
conda install -c conda-forge tensorboard
conda install -c conda-forge gdown
```

#### Installation on Linux using venv: -- NOT WORKING
```
python -m venv third_party/mmpose/env
source third_party/mmpose/env/bin/activate
python -m pip install -r third_party/mmpose/requirements.txt
python -m pip install torch==2.0.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install -U openmim
python -m mim install mmengine
python -m mim install "mmdet>=3.0.0"

# build MMPose from source
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
# instead, install directly:
python -m mim install "mmpose>=1.1.0"
```

### **Update** MMPose to be able to use Vitpose
pip install mmpretrain

changed line 61 in pose config from mmcls.VisionTransformer to mmpretrain.VisionTransformer

update the mmpose:
mim install "mmpose>=1.1.0"

### Gaze Detector

Create a new virtual environment, again, using conda or venv. Install the 
packages listed in `third_party/xgaze_3cams/requirements.txt`. For example:
```
python -m venv third_party/xgaze_3cams/env
source third_party/xgaze_3cams/env/bin/activate
python -m pip install -r third_party/xgaze_3cams/requirements.txt
```


### Facial Expression Detection

```
python3.10 -m venv third_party/emoca/env
source third_party/emoca/env/bin/activate
python -m pip install -r third_party/emoca/requirements.txt
python -m pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt1131/download.html
```

#### Checkpoints
```
bash pull_submodules.sh
cd emoca/gdl_apps/EmotionRecognition/demos/
bash download_assets.sh
```
This creates and fills the folder `isa-tool/third_party/emoca/emoca/assets`.

IMPORTANT: When loading the submodules and checkpoints from the official site
(as above), the code does not find the checkpoint for EMOCAv2. Quick fix: in 
`./third_party/emoca/emoca/assets/EmotionRecognition/face_reconstruction_based/EMOCA-emorec/cfg.yaml` 
line 30, set 
`deca_checkpoint: ./third_party/emoca/emoca/assets/EMOCA/models/EMOCA/detail/checkpoints/deca-epoch=03-val_loss/dataloader_idx_0=9.44489288.ckpt` 
to the old checkpoint. 

Alternative: Download a minimal `assets` folder from 
`ISA_Data_Share/14_MVP/data/emoca` and paste it to 
`isa-tool/third_party/emoca/emoca/assets`.

#### Testing
``` 
python gdl_apps/EmotionRecognition/demos/test_emotion_recognition_on_images.py --modeltype 3dmm --model_name EMOCA-emorec
```


### Active Speaker Detector

```
cd third_party/active_speaker
python3.8 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
cd -
```


## Documentation

Using 
- docstrings (""" """) for functions, methods and modules
- comments (#) for lines or paragraphs
Stick to PEP8 and NumPy style docstrings.

Automate documentation using [Sphinx](https://www.sphinx-doc.org/en/master/). Setup:
```
sudo apt-get install python3-sphinx
cd template-detector/doc/
# sphinx-quickstart  # if building a documentation from scratch
```
To build the documentation:
```
sphinx-build -b html . ./doc  # or
make html                     # if the 'Makefile' is in the same folder
```
and open `./doc/_build/html/index.html` in your browser.


Create a pdf instead of a html: 
```
make latex
cd _build/latex/
pdflatex detector.tex
```


## Testing

Following examples from [ASPP 2019](https://github.com/cscmt/testing_debugging_profiling/tree/master).
Install py.test on Ubuntu:

```
pip install -U pytest
```
From within the code folder, run
```
pytest -v
```
All test functions that start with `test_` are automatically detected by pytest. We collect them in the `./tests/` folder.
Example tests are given in `./tests/test_main.py`.


## Gitlab CI

Added a CI/CD pipeline on gitlab. Use the template `.gitlab-ci.yml` file with some modifications. 

### Runners

From the official docs: *Runners are processes that pick up and execute CI/CD jobs for GitLab. Register as many runners as you want. You can register runners as separate users, on separate servers, and on your local machine.*

[Install Gitlab runner](https://docs.gitlab.com/runner/install/) on Ubuntu:
```
curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh" | sudo bash
sudo apt-get install gitlab-runner
```
And [register a runner](https://docs.gitlab.com/runner/register/index.html): 
```
sudo gitlab-runner register
```
- GitLab instance URL: `https://gitlab.tuebingen.mpg.de/`
- Registration token: `GR13489411eCZ5zNzjBb5QsxSiVzD`
- Description for the runner: `Test runner on Caro's machine`
- Tags for the runner (*When you register a runner, its default behavior is to only pick tagged jobs.* ): `pytests`
- optiional maintenace note: empty
- Runner executor (*GitLab Runner implements a number of executors that can be used to run your builds in different environments.*): `shell`

In the CI/CD settings, find your runner and set `Can run untagged jobs` to `Yes`.

Find your local runners' configuration file in `/etc/gitlab-runner/config.toml` (needs sudo rights to see)

### TODO venv vs. conda vs pip install

I did not get it working easily with conda. Instead, I use a python virtual machine on my runner and istall the `requirements.txt`. For that, I installed 
```
sudo apt install python3.8-venv
```
Now these `requirements.txt` double the requirements specified in `pyproject.toml`. This is bad. TO FIX!

## Collaboration

### Naming conventions
- Group names: 
personL, personR, dyad

### Shared data formats
1. keypoints: np.arrays saved as hdf5\  
shape: [num_frames, num_keypoints, 2d/3dcoords] in 2d the third score is confidence level\

2. gaze: np.arrays - saved as hdf5\
shape: [num_frames, 6d] 6d: 3d position, 3d coords\

3. nodding/smiling/etc  -might be\
shape: [num_frames, label]\
shape: [num_frames, num_emotions/num_categories, confidence scores]\
shape: [num_annotation/num_segments, [starttime, endtime, label]]\

### Calibration Mapping
```
{
    'PIS_ID_000':'2020-08-11',
    'PIS_ID_01': '2020-08-11',
    'PIS_ID_02': '2020-08-11',
    'PIS_ID_03': '2020-08-11',
    'PIS_ID_04': '2020-08-12',
    'PIS_ID_05': '2020-08-12',
    'PIS_ID_06': '2020-08-12',
    'PIS_ID_07': '2020-08-12',
    'PIS_ID_08': '2020-08-12',
    'PIS_ID_09': '2020-08-13',
    'PIS_ID_666': '2020-08-13'
}
```
### Used ChAruCo Calibration Board
BOARD_SIZE = (12, 9)
SQUARE_SIZE = 60 mm
DICT_5X5

