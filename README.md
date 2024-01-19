# ISA - Tool


<span style="color:red">
Important: Please update your local machine_specific_path.toml file! 
Add the `conda_path` - See instructions below in `Getting started`.
</span>


## Getting Started

First, please create a local file `configs/machine_specific_path.toml` by
copy-pasting and renaming the file `configs/machine_specific_path_template.toml`.
Note: the `machine_specific_path.toml` - file is part of the `.gitignore` and 
will not be pushed to git as it contains all file and folder locations that 
are specific to your local machine. 

Edit the `machine_specific_path.toml` - file, by specifying the following:
- `pis_folder_path`: Please copy all needed data from the network path 
`/ps/project/pis/` to your local, while preserving the folder structure. 
Add this local path here for easy local-remote interchange.
- `datasets_folder_path`: The path on your local machine where the datasets
that you want to run the ISA-Tool on are stored.
- `methods_folder_path`: The path to your local folder that contains all third
party methods and their checkpoints -- TODO: move all assets to keeper and 
update installation files/instructions.
- `conda_path`: The path of your conda installation. E.g., for me, this is   
`/is/sg2/cschmitt/source/anaconda3`. On your system, open a command line and 
type `which conda` to help find this path.


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



## Development vs. Production

[On assertions in Python from RealPython](https://realpython.com/python-assert-statement/)

*The goal of assertion should be to uncover programmers’ errors rather 
than users’ errors.*
*A proper use of assertions is to inform developers about unrecoverable errors 
in a program. Assertions shouldn’t signal an expected error, like a 
FileNotFoundError, where a user can take a corrective action and try again.*

*Assertions are great during development, but in production, they can affect the 
code’s performance. 
Assertions take time to run, and they consume memory, so it’s advisable to 
disable them in production.*

Normal mode is typically the mode that you use during development (indicated by 
the build-in constant `__debug__ = True`), while optimized mode is what you 
should use in production (here, `__debug__ = False`).
In production mode, set `__debug__ = False` by running python with `-0` or `-00`
options or setting the `PYTHONOPTIMIZE` environment variable:
- `python -0` or `PYTHONOPTIMIZE=1`: remove the assert statements and any code 
that you’ve explicitly introduced under a conditional targeting `__debug__`. 
- `python -00` or `PYTHONOPTIMIZE=2`: Same as `-0` and also discards docstrings.


### Set class attributes correctly

To prevent the user from giving wrong/unsupported values to a class attribute, 
you can turn the attribute into a managed attribute using the `@property` 
decorator.
Managed attribute provide `setter` and `getter` methods. These are called 
whenever the class changes or retrieves the value of the attribute.
This way, you can move the validation code for the attribute to the `setter`
method, see the example from [RealPython](https://realpython.com/python-assert-statement/#running-python-with-the-o-or-oo-options):
```
class Circle:
   def __init__(self, radius):
       self.radius = radius

   @property
   def radius(self):
       return self._radius

   @radius.setter
   def radius(self, value):
       if value < 0:
           raise ValueError("positive radius expected")
       self._radius = value
```


### Data validation

Instead of testing user input using an assertion, implement a conditional in
the function that raises an error for a wrong input. Then wrap any call to this
function in a `try ... except` block to catch the error and return an 
informative message to the user.

Make the assertion much more descriptive/verbose by specifying which error
and exception is raised, e..g., `ValueError`.


### Exceptions & Logging: When to raise an exception?

From the 
[Python docs](https://docs.python.org/3/howto/logging.html#exceptions-raised-during-logging):
*The logging package is designed to swallow exceptions which occur while logging 
in production.* 

*SystemExit and KeyboardInterrupt exceptions are never swallowed. 
Other exceptions which occur during the `emit()` method of a Handler subclass 
are passed to its `handleError()` method. The default implementation of 
`handleError()` in Handler checks to see if a module-level variable, 
`raiseExceptions`, is set. If set, a traceback is printed to `sys.stderr`. 
If not set, the exception is swallowed.*

*The default value of raiseExceptions is True. This is because during 
development, you typically want to be notified of any exceptions that occur. 
It’s advised that you set raiseExceptions to False for production usage.*


### Error Handling

[Python docs](https://docs.python.org/3/tutorial/errors.html)
*There are (at least) two distinguishable kinds of errors: syntax errors and 
exceptions.*

- Syntax errors denote parsing errors, due to invalid syntax.
- *Errors detected during execution are called exceptions. Even if a statement 
or expression is syntactically correct, it may cause an error when an attempt is 
made to execute it.*

One can handle exceptions by implementing `try ... except` statements. 



