# Tutorials

## Run the example

**1. Download the example data**
The example data is uploaded to [keeper](https://keeper.mpdl.mpg.de/d/d38179804e1144a5880d/). Please download the folder `communication_multiview` and put it in your `datasets_folder_path` - directory which is defined in your file `./machine_specific_paths.toml`. For more information, see [machine-specific config](getting_started.md#2-create-your-machine-specific-config).


**2. Define the dataset's properties**
Check that the file `./detectors/configs/dataset_properties.toml` contains the following dictionary:
```toml
[communication_multiview]
session_IDs = ["session_xyz"]
sequence_IDs = ['']
cam_front = 'view_center'
cam_top = 'view_top'
cam_face1 = 'view_left'
cam_face2 = 'view_right'
subjects_descr = ["person_left", "person_right"]
cam_sees_subjects = {view_center = [0, 1], view_top = [0, 1], view_left = [0], view_right = [1]} 
path_to_calibrations = "<datasets_folder_path>/communication_multiview/calibrations.npz"
data_input_folder = "<datasets_folder_path>/communication_multiview/<session_ID>/"
start_frame_index = 0
fps = 30
```
A detailed description of this file can be found here: [prepare the dataset](getting_started.md#1-prepare-the-dataset).


**3. Add the experiment to run**
To run the NICE toolbox on our new dataset, we need to specify what exactly we want to run in our experiment. Open `./detectors/configs/run_file.toml` and make sure that the `[run]` dictionary details the following:

```toml
[run]
[run.communication_multiview]
components = ["body_joints", "gaze_individual", "gaze_interaction", "kinematics", "proximity", "leaning"]
videos = [
   {session_ID = "session_xyz", sequence_ID='', video_start = 0, video_length = 99, video_skip_frames = false},
]

```
More details can be found in [define the experiment](getting_started.md#3-define-the-experiment-to-run).


**4. Run the NICE Toolbox**
To [run](getting_started.md#4-run-the-code) the toolbox, open a terminal or the API of your choice and execute:

```
# navigate to the NICE toolbox source code folder
cd /path/to/isa-tool/

# activate the python environment
source ./env/bin/activate

# run the toolbox
python detectors/main.py
```

The outputs will be saved in the folder defined in `./detectors/configs/run_file.toml` under `io.out_folder` (with filled-in placeholders). 
To watch the experiment run, check the log file `.../out_folder/ISA-Tool.log`. Expect the tool to take about 6min for this experiment.


