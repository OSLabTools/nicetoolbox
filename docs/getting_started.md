# Getting Started

For a quick start, [Getting Started](#getting-started) explains how to run the NICE Toolbox on the videos of a single camera, without multi-view captures. Multi-view requires calibrated cameras - this is covered in [Advanced Usage](#advanced-usage).

## Contents  <!-- omit in toc -->

- [Getting Started](#getting-started)
  - [1. Prepare the dataset](#1-prepare-the-dataset)
    - [Create a calibration file](#create-a-calibration-file)
  - [2. Create your machine-specific config](#2-create-your-machine-specific-config)
  - [3. Define the experiment to run](#3-define-the-experiment-to-run)
  - [4. Run the code](#4-run-the-code)
  - [Advanced Usage](#advanced-usage)
    - [Understanding the Config Files](#understanding-the-config-files)
    - [Choosing Detectors](#choosing-detectors)
    - [Defining Output Files](#defining-output-files)
    - [Calibration File](#calibration-file)
    - [Multi-View Dataset](#multi-view-dataset)

## 1. Prepare the dataset

The NICE Toolbox supports datasets with video or image input data, multiple camera views, different number of subjects (1 or 2 currently), as well as various folder structures. These dataset-specific details are defined in `./detectors/configs/dataset_properties.toml`. To add a new dataset, edit the toml file by creating a new dictionary within:

```toml
[<dataset_name>]
session_IDs = ['']        # identifiers for each session (list of str)
sequence_IDs = ['']       # identifiers for individual sequences (list of str)
cam_front = ''            # name of folder that contains data of the most frontal camera view (str)
cam_top = ''              # folder name of a frontal camera view from top (str, optional)
cam_face1 = ''            # folder name of a camera view of one subject's face (str, optional)
cam_face2 = ''            # folder name of a camera view of the second subject's face (str, optional)
subjects_descr = []       # define an identifier for the subjects in each video or frame (list of str)
cam_sees_subjects = {}    # define which camera view records which subject (dict: (cam_name, list of int))
path_to_calibrations = "" # file path with placeholders for the calibration files (str, optional)
data_input_folder = ""    # folder path with placeholders to the video or image files (str)
start_frame_index = 0     # how does the dataset index its data? usually, starting with 0 or 1 (int)
fps = 30                  # frame-rate of video data (int, optional)
```

- `cam_front` should contain the name of the camera view that observes the scene from the front. Best, it faces the subjects at about eye-height. 
- `cam_top`, `cam_face1`, and `cam_face2` are the names of optional additional camera views for multi-view predictions.
- `subjects_descr` 
- `cam_sees_subjects` is a dictionary and its keys are the camera_names from above. For each camera, define the subjects it observes from left to right. Each subject is represented by its index in subjects_descr, where indexing starts with 0.
- `path_to_calibrations` and `data_input_folder` may (or in most cases must) contain placeholders. Placeholders can be `<session_ID>`, `<sequence_ID>`, or `<datasets_folder_path>`.

The code expects the data folder to have a pre-defined folder structure:
`dataset_name/session_name/sequence_name(optional)/camera_name(optional)`
Supported data formats are `.mp4`, `.avi`, `.png`, `.jpg`, `.jpeg`. Examples for valid folder structures are:

```
dataset_name/
├── session_name/
│   ├── sequence_name/
│   │   ├── camera_name/
│   │   │   ├── image1.png
│   │   │   ├── image2.png
...
```

```
dataset_name/
├── session_name/
│   ├── camera_name.mp4
...
```

Note: the `calibration_file` (ehich also belongs to the dataset) does not have a specific location, as its filepath is defined in `./detectors/configs/dataset_properties.toml`, see above.

### Example  <!-- omit in toc -->

Assume we have a dataset that contains video sequences from multiple sessions and capture days. The setup of the data is as follows: a single camera records two people sitting next to each other and talking. The camera captures at a framerate of 30 frames per second and the dataset provides frames that are indexed starting from 0.
Further suppose that the dataset directory the following folder structure:

```
test_dataset/
├── session_1/
│   ├── sequence_1/
│   │   ├── view_1/
│   │   └── ...
│   ├── sequence_2/
│   └── ...
├── session_2/
│   ├── ...
├── ...
└── calibration.npz
```

To add this dataset to the NICE Toolbox, we need to add the following to `./detectors/configs/dataset_properties.toml`:

```toml
[test_dataset]                                    # folder name of the dataset
session_IDs = ["session_1", "session_2", ...]     # folder name of the sessions
sequence_IDs = ['sequence_1', 'sequence_1', ...]  # folder name of the sequences
cam_front = 'view_1'                              # a single camera recording from the front
cam_top = ''                                      # no other cameras, leave empty strings
cam_face1 = ''
cam_face2 = ''
subjects_descr = ["personL", "personR"]           # there are 2 people visible in the video
cam_sees_subjects = {view_1 = [0, 1]}             # order of the subjects named in 'subjects_descr'
path_to_calibrations = "<datasets_folder_path>/test_dataset/calibration.npz"
data_input_folder = "<datasets_folder_path>/test_dataset/<session_ID>/<sequence_ID>/"
start_frame_index = 0                             # the dataset provides frames that are indexed starting from 0
fps = 30                                          # The camera captures at a framerate of 30 frames per second
```

### Create a calibration file

Try our calibration converter GUI!

On linux:

```bash
cd isa-tool/
source envs/isa2/bin/activate
python utils/calibration_gui/calibration_converter.py
```

The calibration converter offers multiple options to create, load, or change a calibration file for the NICE toolbox. It outputs the calibration in two files: `calibrations.npz` which is required to run the NICE toolbox and `calibrations.json` which displays the same calibration data in a human-readable (and changeable) file.  
##TODO! Change json to toml

If loading an existing calibration file, the calibration dictionary may look like this in the case of using OpenCV's camera matrix representation:
```toml
[<session_ID>__<sequence_ID>.<camera_name>]
camera_name = "<camera_name>"
image_size = [ <width>, <height>]
mtx = [ [<f_x>, 0.0, <c_x>], [0.0, <f_y>, <c_y>], [ 0.0, 0.0, 1.0]]
dist = [ 0.0, 0.0, 0.0, 0.0, 0.0]
rvec = [ 0.0, 0.0, 0.0]
tvec = [ [ 0.0], [ 0.0], [ 0.0]]
```


**For a single camera**
In case your dataset has a single camera only (no multi-view setup), feel free to leave the rotation matrix (usually `R` or `rvec`) and the translation matrix (commonly denoted with `t` or `tcev`) to the defaults of identity:

```
"rotation_matrix" or "R":       [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
"vector" or "rvec":             [[0.0], [0.0], [0.0]]
"translation", "t", or "tvec":  [0.0, 0.0, 0.0]
```

Similarly, if you do not know the distortion coefficients, set them to `0.0`:
```
"distortions" or "d":           [0.0, 0.0, 0.0, 0.0, 0.0]
```

## 2. Create your machine-specific config

Create a file `./machine_specific_paths.toml`, you can also copy and rename the file `./machine_specific_paths_template.toml`.
These machine specific configuration file should contain the following dictionary:

```toml
# Absolute path to the directory in which all datasets are stored (str)
datasets_folder_path = ''

# Directory for saving toolbox output as an absolute path (str)
output_folder_path = ''

# Where to find your conda (miniconda or anaconda) installation as absolute path (str)
conda_path = ''
```

### Placeholders instead of absolute paths  <!-- omit in toc -->

Note: It is best practice not to use absolute paths in any other files in the ?ISA-Tool? Though absolute pahts do not cause errors, they hinder collaboration and greatly decrease the readability of code. 
Instead, `datasets_folder_path` and `conda_path` are available in the other config files in `./detectors/configs/` as placeholders - use as `<datasets_folder_path>` and `<conda_path>` directly in strings.

## 3. Define the experiment to run

The main config file to run a specific experiment is `./detectors/configs/run_file.toml`. For a first run of an experiment, there are only a few things to adjust:

```toml
visualize = false  # save image/video visualizations of detectors
...

[run.dataset_name]              # change 'dataset_name' to your dataset
components = ["body_joints", "gaze_individual", "gaze_interaction", "kinematics", "proximity", "leaning"]
videos = [                      # define which data to run on
    {
    session_ID = "",            # select the session_ID (str)
    sequence_ID="",             # select the sequence_ID (str, may be empty)
    video_start = 0,            # start of the video in frames, 0 for starting from beginnning (int)
    video_length = 100,         # number of frames to run, defines the length of the video (int)
    video_skip_frames = false   # whether to skip frames or run on all frames (int)
    },
    ...
]

[io]
experiment_name = "<yyyymmdd>"  # optionally, change the name of the experiment, default: date (str)
out_folder = "<pis_folder_path>/experiments/<experiment_name>"  # define where to save the experiment output (str)
...
```

- `visualize` enables saving of intermediate results per detector. Disable for a faster run time, enable for test runs of smaller data subsets and debugging.
- `run.dataset_name.components` lists all the components that should run.
- `run.dataset_name.videos` is a list of dictionaries. Each dictionary defines one video snipped to run. Possibility to extend as needed.
- `io.experiment_name` defaults to the current date (in format YYYMMDD). `<yyyymmdd>` is a placeholder that gets filled automatically.
- `io.out_folder` is the experiment output directory. It supports placeholders for all keys in `io`, all keys in `./machine_specific_paths.toml`, as well as the options `<git_hash>`, `<me>`, `<today>`, `<yyyymmdd>`, `<time>`, and `<pwd>`.

More entries of the `./detectors/configs/run_file.toml` are discussed in ?SECTION???.

## 4. Run the code

To run the code, open a terminal or the API of your choice and do:

```bash
cd /path/to/isa-tool/
source ./env/bin/activate

python detectors/main.py --run_config detectors/configs/run_file.toml --detectors_config detectors/configs/detectors_config.toml --machine_specifics machine_specific_paths.toml
```

The outputs will be saved in the folder defined in `./detectors/configs/run_file.toml` under `io.out_folder` (with filled-in placeholders).
To watch the experiment run, check the log file `.../out_folder/ISA-Tool.log`.

Congratulations! You got your first experiment running :)

## Advanced Usage

There are many more options to individualize and specify the experiment runs. We will cover them in the following.

### Understanding the Config Files

- `./detectors/configs/run_file.toml`
- `./detectors/configs/dataset_properties.toml`
- `./machine_specific_paths.toml`
- `./detectors/configs/detectors_config.toml`
- `./detectors/configs/predictions_mapping.toml`


### Choosing Detectors

Each component can be assigned to and run with multiple different algorithms. This is defined in `./detectors/configs/run_file.toml`:

```toml
[component_algorithm_mapping]
gaze_individual = ['xgaze_3cams']
gaze_interaction = ['gaze_distance']
body_joints = ['hrnetw48', 'vitpose']
hand_joints = ['hrnetw48']
face_landmarks = ['hrnetw48']
kinematics = ['velocity_body']
proximity = ['body_distance']
leaning = ['body_angle']
```

### Defining Output Files

There is one part of the `./detectors/configs/run_file.toml` we did not explain in detail yet, the `[io]`. This part of the config specifies where any out put of the NICE Toolbox gets saved.

```toml
[io]
experiment_name = "<yyyymmdd>"
out_folder = "<pis_folder_path>/experiments/<experiment_name>"
out_sub_folder = "<out_folder>/<dataset_name>_<session_ID>_s<video_start>_l<video_length>"
dataset_config = "detectors/configs/dataset_properties.toml"
assets = "<code_folder>/detectors/assets"

process_data_to = "data_folder"  # options are 'data_folder' and 'tmp_folder'
data_folder = "<pis_folder_path>/raw_processed/isa_tool_input/<dataset_name>_<session_ID>_<sequence_ID>"
tmp_folder = "<pis_folder_path>/experiments/tmp"
detector_out_folder = "<out_sub_folder>/<component_name>/<algorithm_name>/detector_output"
detector_visualization_folder = "<out_sub_folder>/<component_name>/<algorithm_name>/visualization"
detector_additional_output_folder = "<out_sub_folder>/<component_name>/<algorithm_name>/additional_output"
detector_tmp_folder = "<tmp_folder>/<component_name>/<algorithm_name>"
detector_run_config_path = "<out_sub_folder>/<component_name>/<algorithm_name>"
detector_final_result_folder = "<out_sub_folder>/<component_name>"
code_folder = "<pwd>"
conda_path = "<conda_path>"
```
- `out_folder` and `out_sub_folder` (str):
- `dataset_config` (str):
- `assets` (str):
- `process_data_to` (str):
- `data_folder`, `data_folder` (str):
- `detector_out_folder`, `detector_visualization_folder`, `detector_additional_output_folder`, `detector_tmp_folder`, `detector_run_config_path`, `detector_final_result_folder` (str):
- `code_folder` ():
- `conda_path` ():

### Calibration File

This is an extension of the dataset directory. Calibration parameters for all cameras are provided as a nested dictionary, its file path is saved in `./detectors/configs/dataset_properties.toml` under `path_to_calibrations` as `.json` or `.npz`.

**Example:**

```json
{
  "camera_1": {
    "camera_name": "camera_1",
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
  "camera_2": {
    // camera parameters...
  }
}
```

- `camera_name` (str): the name of the camera, same as dict key.

- `image_size` (list): the width and the height of the image that the camera captures. The list format is [image_width, image_height]

- `intrinsics_matrix` (list-of-lists): This matrix, also known as the camera matrix, contains intrinsic parameters of the camera.
The list format is: `[[fx,s,cx],[0,fy,cy],[0,0,1]]`. It is structured as follows:
  - `fx/fy`: focal length (if fx/fy is not specified, f=fx, fy=fx*ar)
  - `s`: skewness, mostly 0
  - `cx/cy`: principle points

- `distortions` (list): The distortion coefficients which correct for lens distortion in the captured images. These coefficients follow the OpenCV model and usually include [k1, k2, p1, p2, k3]
  - `k1, k2, k3` : Radial distortion coefficients.
  - `p1, p2`: Tangential distortion coefficients.

- `rvec` (list): rotation vector - `cv2.rodrigues(rotation_matrix)` - a 3-element vector, a compact representation of rotation matrix.
Its direction represents the axis of rotation and whose magnitude represents the angle of rotation in radians. also known as axis-angle representation.

- `rotation_matrix` (list-of-lists): 3x3 rotation matrix `R` - `cv2.rodrigues(rvec)`

- `translation` (list-of-lists): the translation `t` of the camera from the origin of the coordinate system.

- `extrinsics_matrix` (list-of-lists): This is a 3x4 matrix that combines the rotation matrix and translation vector to describe the camera's position and orientation in space. It is optained by stacking rotation matrix with tranlsation vector: `[R|t]`.

- `projection_matrix` (list-of-lists): 4x4 matrix that projects 3D points in the camera's coordinate system into 2D points in the image coordinate system.
It is obtained by multiplying the intrinsic matrix by the extrinsic matrix: `np.matmul(intrinsic_matrix, extrinsic_matrix)`.

The extrinsic parameters represent a rigid transformation from 3-D world coordinate system to the 3-D camera’s coordinate system.
The intrinsic parameters represent a projective transformation from the 3-D camera’s coordinates into the 2-D image coordinates.
For more information, see a great introduction from [MathWorks](https://de.mathworks.com/help/vision/ug/camera-calibration.html).

### Calibration Mapping -- OUTDATED? <!-- omit in toc -->

The ??ISA-Tool?? allows for adding a mapping between, e.g., session_IDs and calibration file paths. The idea is that during a data capture the camera system is recalibrated occasionally.
//TODO In which file to add this mapping? Is this still up-to-date?

```json
{
    "session_1": "calibration_1",
    "session_2": "calibration_1",
    "session_3": "calibration_2",
    ...
}
```

### Multi-View Dataset

We assume that the cameras are time-synchronized and calibrated intrinsically and extrinsically. After adding the calibration file as described in the previous section [Calibration file](#calibration-file).
To run on multi-view camera input, we now need to review the dataset config `./detectors/configs/dataset_properties.toml`. Make sure to update the following keys for your dataset:

```toml
cam_top = ''              # folder name of a frontal camera view from top (str, optional)
cam_face1 = ''            # folder name of a camera view of one subject's face (str, optional)
cam_face2 = ''            # folder name of a camera view of the second subject's face (str, optional)
subjects_descr = []       # define an identifier for the subjects in each video or frame (list of str)
cam_sees_subjects = {}    # define which camera view records which subject (dict: (cam_name, list of int))
path_to_calibrations = "" # file path with placeholders for the calibration files (str, optional)
```

Now, running the experiment will use all cameras provided. Check the log-file to see it working.
