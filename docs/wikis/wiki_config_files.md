# Understanding the Config Files


There are many more options to individualize and specify the experiment runs. We will cover them in the following.

WORK IN PROGRESS...

## Contents

- `./detectors/configs/run_file.toml`
- `./detectors/configs/dataset_properties.toml`
- `./machine_specific_paths.toml`
- `./detectors/configs/detectors_config.toml`
- `./detectors/configs/predictions_mapping.toml`
- `./visual/configs/visualizer_config.toml`


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


## Visualizer Config

Defined in `./visual/configs/visualizer_config.toml`.

Visualizer Config consists of three main part io, media, and component specifications. 

```toml
[io]
dataset_folder = "<datasets_folder_path>"                                 # main dataset folder
nice_tool_input_folder = "<output_folder_path>/raw_processed/isa_tool_input/<dataset_name>_<session_ID>_<sequence_ID>" # raw_processed input data
experiment_folder = "<output_folder_path>/experiments/20240906_mm"        # NICE Toolbox experiment output
experiment_video_folder = "<experiment_folder>/<video_name>"              # NICE Toolbox output folder for the specific video.
experiment_video_component = "<experiment_video_folder>/<component_name>" # NICE Toolbox output folder for the specific component

[media]                                # each Media session shows one video results.
dataset_name = 'mpi_inf_3dhp'          # dataset of the video
video_name = 'mpi_inf_3dhp_S1_s20_l20' # name of video result folder
multi_view = true                      # true if you have multiple cameras, otherwise set it to false
[media.visualize]                      # specify what will be visualized
components = [..]                     # list of components
camera_position = true                 # true if you want to visuailize camera position -- requires extrinsic information of the camera
start_frame = 0                        # starting frame for the visualization
end_frame = -1                         # end frame for the visualization, -1 means process until the end of the video
visualize_interval = 1                 # 1 means visualize every frame; change the parameter accordingly if you want to visualize every x frames
```
### Configuring Component Data Display in Rerun Windows
You can control which data will be shown in specific rerun windows by adjusting the 'media.component.canvas' items
The keys (like '3d' or '2d_interpolated') represent different type of data provided by that component.
The value lists define which canvases (rerun windows) will show the data
1. '3D_Canvas': This shows data in the 3D canvas. It is only for multi-view datasets. (Do not change the canvas name).
2. Cameras: Data will be visualized on that specific camera image. The camera name must match the camera placeholder names in dataset_properties.toml
3. Metrics - The displays the data as plots.
4. Empty list: If you don't want the data to be visualized, leave the list empty.

**Configuring Algorithm Display**
Under 'media.component', the algorithms parameter let you choose which algorithms to display.
For example, if you have multiple algorithms (e.g., hrnetw48 and vitpose in the body_joints component),
you can specify which algorithm’s results to show.
If you want to see results from both algorithms, list both names.

**Configuring Apearance**
Under 'media.component.appearance', you can configure the color and radii (the size of the dots and lines).

```toml
# Component: gaze individual - An example for 3D_Canvas and Camera Canvases
[media.gaze_individual]
algorithms = ['xgaze_3cams']            # list of algorithms
[media.gaze_individual.canvas]
3d_filtered = ["3D_Canvas", "<cam_face1>", "<cam_face2>", "<cam_top>", "<cam_front>"] ## key options 3d, 3d_filtered ## value options: [3D_Canvas], [3D_Canvas, camera names], [camera names], []
                                                                                      ## Note: Delete '3D_Canvas' if you don't have a multi-view setup.
[media.gaze_individual.appearance]
colors = [[0,150, 90]]                  # define the color of individual gaze
radii = {'3d'= 0.01, 'camera_view'= 4}  # define the size of gaze arrow in 3D_Canvas and camera views

# Component: kinematics  - An example for Metrics Display 
[media.kinematics]
algorithms = ['velocity_body']

[media.kinematics.canvas]
velocity_body_3d = ["metric_velocity"] # if don't have multi-view, use velocity_body_2d
#velocity_body_2d = ["metric_velocity"]

[media.kinematics.joints]              # visualize the mean velocity for the given bodyparts.
"head" = ["nose","left_eye","right_eye","left_ear","right_ear"]
"upper_body" = ["left_shoulder","right_shoulder","left_elbow", "right_elbow", "left_wrist", "right_wrist"]  # Indices of keypoints belonging to the upper body
"lower_body" = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

```
### Configuring Rerun Viewer and Blueprint in Rerun
When Rerun is initiated, it automatically creates a heuristic view for the windows. 
You can manually change this by dragging the windows or adding new ones using the plus sign next to the Blueprint menu.

![../graphics/rerun_blueprint.png](../graphics/rerun_blueprint.png)  

This Blueprint can be saved using the 'Save blueprint...' menu option and reopened later using the 
'Open' option. Once you configure the Rerun viewer, it will use the same blueprint for future sessions. 
You can reset the layout by clicking 'Reset Blueprint.'

![../graphics/rerun_viewer.png](../graphics/rerun_viewer.png)

If your new video does not have certain windows that the old dataset had, unused empty windows may appear. 
To get a fresh heuristic layout, reset the blueprint.

 