# ISA - Tool


<span style="color:red">
Important: Please download the assets/ folder from KEEPER - See instructions 
below in `Getting started` and `1. Download assets`.
</span>


## Getting Started

### 1. Download assets

Assets are uploaded to [keeper](https://keeper.mpdl.mpg.de/d/a9f91e7e60e84da69fc0/).
Please download the folder an put it in `isa-tool/assets`.

### 2. Configure machine_specific_paths

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

### 3. Installation

1. Clone the Isa-tool and oslab_utils (if not already installed) repositories and navigate into Isa-tool:
```
git clone git@gitlab.tuebingen.mpg.de:cschmitt/isa-tool.git
git clone https://gitlab.tuebingen.mpg.de/cschmitt/oslab_utils.git 
cd isa-tool/
```
2. (in Linux) Install the necessary libraries 

<span style="color:red">
**Important - Prerequisites:**
</span>

1. Python 3.10
2. Conda is installed
3. Cuda 11.8

The installation scripts presume that Conda, CUDA 11.8, and Python 3.10 are already installed on your system. 
Installation via Conda is mandatory because of oepnmmlab installation (pose detector).
However, if you wish to use different versions of Python and CUDA, 
you can modify the corresponding lines in the installation files.

The installation script (installation/install_all): 
1. Setup isa-tool environment
2. Setup conda environment for Openmmlab (pose detector)
3. Setup venv environment for Gaze Detector

**How-to:**

Open a terminal and navigate to main directory where isa-tool then type the followings into terminal

```
cd isa-tool/installation
chmod +x installation_all.sh # to add executable permission to the script
./install_all.sh # to install all necessary libraries
```
4. (in Windows) Install the necessary libraries 
#TODO - update env_setup.bat file & update the README file 

5. Install additional virtual environments for the third-party codes, as 
described next.

5.1. **Facial Expression Detection**

```
python3.10 -m venv third_party/emoca/env
source third_party/emoca/env/bin/activate
python -m pip install -r third_party/emoca/requirements.txt
python -m pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt1131/download.html
```

**Checkpoints**
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

**Testing**
``` 
python gdl_apps/EmotionRecognition/demos/test_emotion_recognition_on_images.py --modeltype 3dmm --model_name EMOCA-emorec
```


5.2. **Active Speaker Detector**

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

### Troubleshooting

Very helpful: [Sphinx getting started](https://www.sphinx-doc.org/en/master/tutorial/getting-started.html)

For html:
- `sphinx-build -M html docs/source/ docs/build/`

For pdfs:
- `sudo apt install texlive-formats-extra`
- `sphinx-build -M latexpdf docs/source/ docs/build/`
- `cd docs/` and `make latexpdf`
- `pip install --upgrade myst-parser`

If anything went wrong before try `make clean` before running `make html`.


Error: `ImportError: cannot import name 'soft_unicode' from 'markupsafe'`. Apparently, newer `markupsafe` versions do not suppport `soft_unicode` anymore.
```
sudo apt remove python3-sphinx
sudo apt autoremove

pip uninstall markupsafe
pip install markupsafe==2.0.1
```


### Doxygen

[Installation instructions](https://www.doxygen.nl/manual/install.html)

Binaries failed on my machine with the error 
/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by doxygen)

Instead: download source code and follow instructions. Additionally add
```
sudo apt install flex
sudo apt install bison
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

### Components

Each component as folder plus -- entities within the `.npz` file.

- hand_joints      -- 2d, 3d, bounding_box
- body_joints      -- 2d, 3d, bounding_box
- face_landmarks   -- 2d, 3d, bounding_box
- gaze_individual  -- vector, head_avg, bounding_box, (2d / 3d?)
- gaze_interaction -- look_at, mutual, gaze_head_distance
- kinematics       -- displacement_body, velocity_body, displacement_face, velocity_face
- leaning          -- body_angle
- proximity        -- body_distance

Inside each folder, save results in `<algorithm>.npz` - one file per detector/feature.
The toolkit behind each algorithm is defined within the detector_config, not necessarily in the file/folder names. 

Entities in data_description:
- 'coordinate_x', 'coordinate_y', 'coordinate_z', 'coordinate_u', 'coordinate_v'
- 'distance', 'angle_deg', 'gradient_angle', 'confidence_score', 'velocity'
- 'to_face_<subject_descr>', 'look_at_<subject_descr>', 'with_<subject_descr>'

#### Components outputs

Each `<algorithm>.npz` file contains several output files saved as numpy arrays ('.npy'). 
All these numpy arrays share a common structure in their first 3 dimension :
[number_of_subjects, number_of_cameras, number_of_frames]

Output files: 
 - **body_joints**/**hand_joints**/**face_landmarks**:
   - 2d: -x/y coordinates of body/hand joints or face landmarks & their confidence score. It saves the raw output of the algorithm 
     - shape: [..., number_of_bodyjoints, 3] last dimension: 'coordinate_x', 'coordinate_y', 'confidence_score' 
   - 2d_filtered: if user set filtered true in detectors_config - applied Savitzky-Golay filter to algorithm output. 
   window_length and polyorder parameters can be adjusted in detectors_config. 
     - shape - same as 2d results
   - 2d_interpolated: applied a correction on algorithm output (raw/filtered) -- 
   the detections with a low confidence score were removed. The missing values were interpolated if the number of consecutive missing value is below 10
     - shape - same as 2d results
   - bbox_2d: coordinates of the bounding box of the full body of the subject.
     - shape: shape: [..., 1, 5] last dimension: 'top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y', 'confidence_score'
   - 3d: if more than 1 camera, calculates 3d coordinates of the body joints by triangulating two camera views. 
   Note: using 2d_interpolated results
     - shape: [..., number_of_bodyjoints, 3] last dimension: 'coordinate_x', 'coordinate_y', 'coordinate_z'

    
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

## Calibration 

### Calibration file
Calibration parameters for all cameras were provided as a nested dictionary (saved as `camera_params_...json`).

**Example:**

```json
{
  "cam4": {
    "camera_name": "cam4",
    "image_size": [1000, 700],
    "intrinsic_matrix": [
      [1193.6641930950077, 0.0, 503.77365693839107],
      [0.0, 1193.410339778219, 352.12891433016244],
      [0.0, 0.0, 1.0]
    ],
    "distortions": [-0.1412521384983322, 0.14702510007618264, 0.00010429739735286396, -0.0004644593818576435],
    "rvec": [0.0, 0.0, 0.0],
    "rotation_matrix": [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ],
    "translation": [[0.0], [0.0], [0.0]],
    "extrinsics_matrix": [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0]
    ],
    "projection_matrix": [
      [1193.6641930950077, 0.0, 503.77365693839107, 0.0],
      [0.0, 1193.410339778219, 352.12891433016244, 0.0],
      [0.0, 0.0, 1.0, 0.0]
    ]
  },
  "cam3": {
    // camera parameters...
  }, 
   "cam2": {
    // camera parameters...
  }
}
```
### Calibration parameters

1. **camera_name (str):** the name of the camera, same as dict key

2. **image_size (list):** the width and the height of the image that the camera captures. The list format is [image_width, image_height]

3. **intrinsics_matrix (list-of-lists):** This matrix, also known as the camera matrix, contains intrinsic parameters of the camera. 
The list format is: [[fx,s,cx],[0,fy,cy],[0,0,1]]. It is structured as follows: 
- fx/fy: focal length (if fx/fy is not specified, f=fx, fy=fx*ar) 
- s skewness, mostly 0
- cx/cy: principle points

4. **distortions (list):**  the distortion coefficients which correct for lens distortion in the captured images. 
These coefficients follow the OpenCV model and usually include [k1, k2, p1, p2, k3] 
- k1, k2, k3 : Radial distortion coefficients.
- p1, p2: Tangential distortion coefficients.

5. **rvec (list):** rotation vector - cv2.rodrigues(rotation_matrix) - a 3-element vector, a compact representation of rotation matrix. 
Its direction represents the axis of rotation and whose magnitude represents the angle of rotation in radians. also known as axis-angle representation.

6. **rotation_matrix (list-of-lists):** 3x3 rotation matrix - cv2.rodrigues(rvec)

7. **translation (list-of-lists):** the translation of the camera from the origin of the coordinate system. 

8. **extrinsics_matrix (list-of-lists):** This is a 3x4 matrix that combines the rotation matrix and translation vector 
to describe the camera's position and orientation in space. It is optained by stacking rotation matrix with tranlsation vector: [R|t]

9. **projection_matrix (list-of-lists):** 4x4 matrix that projects 3D points in the camera's coordinate system into 2D points in the image coordinate system. 
It is obtained by multiplying the intrinsic matrix by the extrinsic matrix: np.matmul(intrinsic_matrix, extrinsic_matrix)

The extrinsic parameters represent a rigid transformation from 3-D world coordinate system to the 3-D camera’s coordinate system. 
The intrinsic parameters represent a projective transformation from the 3-D camera’s coordinates into the 2-D image coordinates.
For more information, see https://de.mathworks.com/help/vision/ug/camera-calibration.html


### Calibration Mapping (Dyadic_Communication dataset)
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
### Used ChAruCo Calibration Board (Dyadic_Communication dataset)
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



