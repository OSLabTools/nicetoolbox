# Understanding the Config Files


There are many more options to individualize and specify the experiment runs. We will cover them in the following.

WORK IN PROGRESS...

## Contents

- `./detectors/configs/run_file.toml`
- `./detectors/configs/dataset_properties.toml`
- `./machine_specific_paths.toml`
- `./detectors/configs/detectors_config.toml`
- `./detectors/configs/predictions_mapping.toml`


## Run file

```toml
visualize = false               # save image/video visualizations of detectors
...

[run.dataset_name]              # change 'dataset_name' to your dataset
components = ["body_joints", "gaze_individual", "gaze_interaction", "kinematics", "proximity", "leaning"]
videos = [
    {                           # define which data to run on
    session_ID = "",            # select the session_ID (str)
    sequence_ID="",             # select the sequence_ID (str, may be empty)
    video_start = 0,            # start of the video in frames, 0 for starting from beginnning (int)
    video_length = 100,         # number of frames to run, defines the length of the video (int)
    video_skip_frames = false   # whether to skip frames or run on all frames, default: false (bool)
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



## Dataset properties

Defined in `./detectors/configs/dataset_properties.toml`.


```toml
[<dataset_name>]
session_IDs = ['']        # identifiers for each session (list of str)
sequence_IDs = ['']       # identifiers for individual sequences (list of str)
cam_front = ''            # name of the camera with the most frontal view (str)
cam_top = ''              # camera name of a frontal view from top (str, optional)
cam_face1 = ''            # camera name of a view of one subject's face (str, optional)
cam_face2 = ''            # caemra name of a view of a second subject's face (str, optional)
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