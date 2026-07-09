# Understanding the config files


The configuration files are Python dictionaries saved in `.toml` - files that contain details about the machine on which the toolbox is running, the experiment that runs, the data the toolbox runs on, the detectors and algorithms it is using, a mapping between different data identifiers, as well as the visualization that shows the toolbox's results. Most of these configuration dictionaries may contain placeholders which are explained first.

```{contents} Contents
:depth: 3
```


## Placeholders

Placeholders can be put into strings and are filled automatically during run-time. All configs support the use of placeholders, where reasonable.
Placeholders are indicated by enclosing characters `<` and `>` and may take the following values:
1. All keys in `./machine_specific_paths.toml` and `./nice_project.toml`, as well as `<project_folder_path>` (the path to the project folder passed via `--project_folder_path`),
2. All keys from `io` as defined in `./configs/detectors_run_file.toml`,
3. The keys `<cur_dataset_name>`, `<cur_component_name>`, `<cur_algorithm_name>`, `<cur_sequence_id>`, `<cur_video_start>`, and `<cur_video_length>` that define the current experiment run are filled during program execution based on the specifications in the run file `./configs/detectors_run_file.toml`,
4. The options `<git_hash>`, `<me>`, `<today>`, `<yyyymmdd>`, `<time>`, and `<pwd>`.
5. Inside a `[dataset_name.template]` block (see [dataset properties](#dataset-properties)), any user-defined key can be referenced from other fields in the same sequence. For example, defining `dataset_root = "<datasets_folder_path>/my_dataset"` lets you write `<dataset_root>` in every path below it.


Some examples:

- The default output folder path defined in the [run file's io](#defining-input-and-output-files) is `"<output_folder_path>/experiments/<experiment_name>"`. During run time, the placeholder `<output_folder_path>` is filled from the [project config](#project-config) and the `<experiment_name>` is replaced by the value defined in the same dictionary as the output folder path, the [run file's io](#defining-input-and-output-files).
- A typical camera path inside a dataset template is `"<dataset_root>/<sequence_id>/view_center.mp4"`. The `<dataset_root>` is a user-defined variable declared elsewhere in the template, and `<sequence_id>` is composed per-sequence from that sequence's own fields when the config is loaded.

## Machine specifics

The `./machine_specific_paths.toml` file configures paths specific to the machine running NICE Toolbox. It is part of `.gitignore` and must be created on each machine. Generate it with `make create_machine_specifics`.

```toml
conda_path = ''
# Optional; leave empty if you do not run detectors that need Hugging Face gated weights.
hugging_face_token = ''
```
- `conda_path` contains the absolute path to the conda installation on the machine (str).
- `hugging_face_token` is an optional secret for Hugging Face Hub (e.g. **sam_3d_body**, **WhisperX** pyannote diarization). Hub-based detectors read only this field (not shell environment variables).


## Project config

The `./nice_project.toml` file configures paths specific to a project (datasets, outputs, configs). It is gitignored and generated with `make create_project`. The `<project_folder_path>` placeholder is automatically set to the absolute path of the folder passed via `--project_folder_path`.

```toml
configs_folder_path = '<project_folder_path>/configs'
datasets_folder_path = '../datasets'
output_folder_path = '../outputs'
```
- `configs_folder_path` is the path to the project's config folder (str). Used as the default location for all config files.
- `datasets_folder_path` is the path to the directory in which all datasets are stored (str).
- `output_folder_path` is the path to the directory in which all toolbox output is saved (str).

## Run file

The run file `./configs/detectors_run_file.toml` defines the experiments to run. This config consists of four parts, that detail general properties, the chosen detectors, the dataset(s), and the experiment's output files. The config supports placeholders as described [here](#placeholders). Each part is described in the following.


### General properties

Properties that apply to all experiments.

```toml
visualize = true
save_csv = true

error_level = "DETECTOR"
log_level = "INFO"

check_missing_detectors_dependencies = true
skip_gated_models_errors = true
```
- `visualize` enables saving of intermediate results per detector (bool). Disable for a faster run time, enable for test runs of smaller data subsets and debugging.
- `save_csv` enables saving all results to 2d tables in csv-files (bool).
- `error_level` controls how strictly errors abort processing (str). Options: `"DETECTOR"` (skip failed detectors), `"SEQUENCE"` (skip failed sequences), `"STRICT"` (halt on any error).
- `log_level` sets the logging verbosity (str). Options: `"CRITICAL"`, `"ERROR"`, `"WARNING"`, `"INFO"`, `"DEBUG"`.
- `check_missing_detectors_dependencies` when true, verifies that all selected detectors have their input dependencies enabled before running (bool).
- `skip_gated_models_errors` when true, silently drops any algorithm that relies on a gated Hugging Face model (e.g. `sam_3d_body`, `whisperx` diarization) if no `hugging_face_token` is configured in `machine_specific_paths.toml` (bool). When false, the run aborts with an error instead. Set to true for smoke runs on machines without HF access; set to false in production to catch missing tokens early.

### Choosing algorithms to run

The `algorithms` field lists which algorithm instances (defined in `detectors_config.toml`) to run. All listed algorithms are applied to every dataset in the `[run]` section.

```toml
algorithms = ["hrnetw48", "vitpose_huge", "eth_xgaze", "gaze_fusion", "gaze_distance"]
```

### Defining the experiments

NICE Toolbox supports running multiple datasets sequentially from a single program call.

`[run]` defines which data to process. Each key is a dataset name; the value specifies which sequences to run on:

```toml
[run.dataset_name]
sequences = [
    {sequence_id = "sequence_xyz", video_start = 0, video_length = 100},
    {sequence_id = "S1_*",         video_start = 0, video_length = -1},   # wildcard: all sequences starting with "S1_"
    ...
]
```
- `sequences` defines which sequences of the chosen dataset to run on (list of dict). Each dict selects one sequence (or a group of them via wildcard). Multiple entries run sequentially.

    - `sequence_id` selects one of the sequences declared in the [dataset's properties](#dataset-properties) (str). Supports glob wildcards (`*`, `?`, `[...]`) — `"*"` matches all sequences, `"S1_*"` matches every id starting with `S1_`. A non-wildcard id must match exactly.
    - `video_start` - starting point of the video (int or timestamp).
    - `video_length` - duration of the video segment (int or timestamp). Use `-1` for the full video.
    
    Both `video_start` and `video_length` accept either frame numbers or timestamps:

    - **Frame-based format:**
      - `video_start`: Frame number (0 for the beginning).
      - `video_length`: Number of frames.
      - `video_start` + `video_length` must not exceed the total frame count.
      - `video_length` can be set to `-1`, which means all frames until the end of the video.

    - **Timestamp format:**
      - Accepts: `HH-MM-SS`, `HH-MM-SS.mmm`, `HH:MM:SS` or `HH:MM:SS.mmm`
      - Examples:
        - `00:01:12.100` - starts 1 minute, 12 seconds, and 100 milliseconds from the beginning
        - `00-05-30` - starts 5 minutes and 30 seconds from the beginning
      - Constraint: `video_start` + `video_length` must not exceed the video duration

```{note}
Every sequence declared in `dataset_properties.toml` has a unique `sequence_id`; that id addresses one video (or bundle of video/audio tracks) end-to-end. `video_start` and `video_length` slice into that video.
```



### Input and output files

The last part of the run file specifies where inputs can be found and any output of the NICE Toolbox gets saved.

```toml
[io]
experiment_name = "<yyyymmdd>"
out_folder = "<output_folder_path>/experiments/<experiment_name>"
out_sub_folder_name = "<cur_dataset_name>_<cur_sequence_id>_s<cur_video_start>_l<cur_video_length>"
out_sub_folder = "<out_folder>/<out_sub_folder_name>"
csv_out_folder = "<out_sub_folder>/_csv_files"
nicetoolbox_input_folder = "<output_folder_path>/nicetoolbox_input/<cur_dataset_name>_<cur_sequence_id>"
code_folder = "<pwd>"
assets = "<code_folder>/nicetoolbox/detectors/assets"
asset_manifest = "<code_folder>/configs/asset_manifest.toml"
dataset_properties = "<configs_folder_path>/dataset_properties.toml"
detectors_config = "<configs_folder_path>/detectors_config.toml"
predictions_mapping = "<configs_folder_path>/predictions_mapping.toml"
detector_folder = "<out_sub_folder>/<cur_component_name>/<cur_algorithm_name>"
detector_out_folder = "<detector_folder>/detector_output"
detector_visualization_folder = "<detector_folder>/visualization"
detector_additional_output_folder = "<detector_folder>/additional_output"
detector_run_config_path = "<detector_folder>"
detector_final_result_folder = "<out_sub_folder>/<cur_component_name>"
```
- `experiment_name` defines the name under which all experiments are run (str), defaults to today's date (in format YYYYMMDD).
- `out_folder` is the top level directory where the results of all experiments are saved (str).
- `out_sub_folder_name` is the name template for a single experiment run (str).
- `out_sub_folder` is the full output directory path for a single experiment run (str).
- `csv_out_folder` is where CSV result files are saved (str).
- `nicetoolbox_input_folder` is the path to the directory in which pre-processed input data gets stored during run time (str). As different algorithms require different file formats and folder structures as input, the NICE Toolbox prepares the given data accordingly. This pre-processed data is cached for faster run times when repeating runs over the same data.
- `code_folder` names the machine's folder path to the nicetoolbox repo (str), filled automatically.
- `assets` stores the folder path of additional assets, like model checkpoints and weights (str).
- `asset_manifest` is the path to the asset manifest config that tracks downloadable model weights (str).
- `dataset_properties`, `detectors_config`, and `predictions_mapping` store where to find those config files (str). All default to the project's `<configs_folder_path>`.
- `detector_folder`, `detector_out_folder`, `detector_visualization_folder`, `detector_additional_output_folder`, `detector_run_config_path`, and `detector_final_result_folder` define where each detector stores intermediate and final outputs (str). The final results of all components and algorithms per detector are saved under `detector_final_result_folder`.

## Dataset properties

Properties that are specific per dataset are collected in `./configs/dataset_properties.toml`. Each dataset is a **flat list of sequences**. Every sequence is self-describing — it owns its own camera set, audio tracks, subject list, and annotation paths — so sequences within the same dataset can differ in shape (a lost camera, a missing audio track, etc.).

To avoid repeating identical fields on every sequence, each dataset can declare a **template** whose values are merged into every sequence; per-sequence keys override the template on a per-key basis. For large datasets, a **discovery pattern** can enumerate sequences from the filesystem instead of hand-listing them.

### Minimal shape

```toml
[dataset_name]
sequences = [{sequence_id = "sequence_xyz"}]

[dataset_name.template]
dataset_root      = "<datasets_folder_path>/dataset_name"
data_input_folder = "<dataset_root>/<sequence_id>"
path_to_calibrations = "<dataset_root>/calibrations.npz"
subjects_descr    = ["person_left", "person_right"]

[dataset_name.template.video.cameras]
view_center = {path = "<data_input_folder>/view_center.mp4", sees_subjects = [0, 1]}
view_left   = {path = "<data_input_folder>/view_left.mp4",   sees_subjects = [0]}
```

**Top-level keys per dataset**
- `sequences` — list of sequence entries (list of dict). Each entry needs at least `sequence_id`; any other schema field can be set here to override the template.
- `template` — optional block whose fields are merged into every sequence entry before validation (dict, optional). Keys can be schema fields (`subjects_descr`, `video.cameras`, ...) *or* user-defined placeholder variables (`dataset_root`, `data_input_folder`, ...) referenced from other strings.
- `discover_sequences` — optional filesystem pattern that expands to sequence entries at load time (str, optional). See [Filesystem discovery](#filesystem-discovery) below.

**Per-sequence fields (from schema)**
- `sequence_id` — unique identifier within the dataset (str, required).
- `subjects_descr` — identifiers for subjects in the recording, ordered left to right (list of str). Length must match the number of people visible.
- `path_to_calibrations` — path to the camera calibration `.npz` file (str, optional). Leave empty if you have no calibration.
- `video.cameras` — named camera tracks (dict of `camera_name → {path, sees_subjects}`). `sees_subjects` is a list of subject indices into `subjects_descr` (0-based). `path` may contain a single `*` wildcard, expected to resolve to exactly one file.
- `audio.tracks` — named audio tracks (dict of `track_name → {...}`). Each track is either embedded (`camera = "<camera_name>"`) or standalone (`path = "..."`). Optional `stream` and `channel` select a specific audio stream or channel in the source file. `hears_subjects` is required.
- `annotation.components` — per-component ground-truth files used by the evaluation pipeline (dict of `component_name → {path}`).

**Placeholder variables inside the template**

Any key you add to the template that isn't a schema field (e.g. `dataset_root`, `data_input_folder`, `annotations_folder`) becomes a placeholder you can reference from other strings in that dataset. They are resolved via the sequence's own sibling fields — so `<sequence_id>` inside a template string composes to *that* sequence's id, no runtime placeholder needed.

**Example.** Given this config:

```toml
[my_dataset]
sequences = [
    {sequence_id = "S1"},
    {sequence_id = "S2"},
]

[my_dataset.template]
dataset_root = "<datasets_folder_path>/my_dataset"    # user-defined variable
data_input_folder = "<dataset_root>/<sequence_id>"    # references dataset_root + sequence_id
path_to_calibrations = "<dataset_root>/calibrations.npz"

[my_dataset.template.video.cameras]
view_1 = {path = "<data_input_folder>/view_1.mp4", sees_subjects = [0, 1]}
```

The loader merges the template into each sequence dict, then resolves placeholders using each sequence's own siblings. After loading, sequence `S1` has:

```
dataset_root         = "/abs/path/to/datasets/my_dataset"
data_input_folder    = "/abs/path/to/datasets/my_dataset/S1"
path_to_calibrations = "/abs/path/to/datasets/my_dataset/calibrations.npz"
video.cameras.view_1.path = "/abs/path/to/datasets/my_dataset/S1/view_1.mp4"
```

And sequence `S2` gets the same values with `S2` substituted in place of `S1`. Note the resolution order inside each sequence: `<datasets_folder_path>` comes from the project config; `<dataset_root>` and `<data_input_folder>` are user-defined siblings resolved in the same dict; `<sequence_id>` is the sequence's own `sequence_id` field. No runtime step is involved — everything above is settled at config-load time.

The user-defined variables (`dataset_root`, `data_input_folder`) are used only for placeholder composition — they're not part of the sequence schema and are dropped after resolution.

### Filesystem discovery

Setting `discover_sequences` enumerates sequence directories or files at config-load time. Named wildcards `[name]` capture path segments as placeholder variables; ordinary `<placeholders>` are resolved first.

```toml
[dataset_name]
discover_sequences = "<datasets_folder_path>/dataset_name/[sequence_id]"

[dataset_name.template]
dataset_root      = "<datasets_folder_path>/dataset_name"
data_input_folder = "<dataset_root>/<sequence_id>"
...
```

**Example.** Suppose the filesystem contains:

```
/abs/path/to/datasets/dataset_name/
├── S1/
├── S2/
└── S3/
```

The discovery step turns the pattern `<datasets_folder_path>/dataset_name/[sequence_id]` into three sequence entries — one per matching folder — with the captured segment assigned to `sequence_id`:

```
sequences = [
    {sequence_id = "S1"},
    {sequence_id = "S2"},
    {sequence_id = "S3"},
]
```

Each entry then goes through the same template merge described above, so `data_input_folder` for `S1` resolves to `/abs/path/to/datasets/dataset_name/S1`, and so on.

**Multiple captures.** If the pattern uses more than one `[name]` segment, e.g. `.../[capture_id]/[take]`:

```
/abs/path/to/datasets/dataset_name/
├── day1/
│   ├── take_A/
│   └── take_B/
└── day2/
    └── take_A/
```

discovery emits three entries — one per matched leaf path — each carrying both captures:

```
sequences = [
    {capture_id = "day1", take = "take_A"},
    {capture_id = "day1", take = "take_B"},
    {capture_id = "day2", take = "take_A"},
]
```

Since none of these have a `sequence_id` yet, compose one in the template:

```toml
[dataset_name.template]
sequence_id = "<capture_id>_<take>"    # → "day1_take_A", "day1_take_B", "day2_take_A"
data_input_folder = "<dataset_root>/<capture_id>/<take>"
```

### Combining discovery with explicit overrides

`discover_sequences` and an explicit `sequences = [...]` list can be used together in the same dataset. This is the common case where most sequences follow the template exactly, but one or two need a tweak (an extra annotation file, a different camera set, a missing audio track).

**How the merge works.** At load time the sequence list is built in this order:

1. Run `discover_sequences` to produce one dict per matched path.
2. For each explicit entry in `sequences = [...]`:
   - If its `sequence_id` matches one of the discovered entries, its fields are merged **on top of** that discovered entry — key by key, so only the fields you set on the explicit entry override the template. All other fields keep whatever the template provides.
   - If the `sequence_id` does not match any discovered entry, the explicit entry is **appended** to the list.
3. The template is then merged into every resulting entry (per-key override, as described in [Template merge rules](#template-merge-rules)).

**Example.** Discovery finds `S1`, `S2`, `S3` on disk, and we want to give `S2` a different left camera and add annotations only for `S3`:

```toml
[dataset_name]
discover_sequences = "<datasets_folder_path>/dataset_name/[sequence_id]"
sequences = [
    # S2 exists on disk — this entry overrides its `video.cameras` block
    {sequence_id = "S2", video = {cameras = {view_left = {path = "<data_input_folder>/alt_left.mp4", sees_subjects = [0]}}}},
    # S3 also exists — attach an annotation component to just this one
    {sequence_id = "S3", annotation = {components = {gaze = {path = "<dataset_root>/annotations/S3_gaze.npz"}}}},
    # this id does NOT match any discovered folder — it's appended as a new sequence
    {sequence_id = "S4_manual"},
]

[dataset_name.template]
# ... shared template as before ...
```

The resulting sequence list after resolution is `[S1, S2 (with alt left camera), S3 (with gaze annotation), S4_manual]`. `S1` is untouched — it takes everything from the template. `S2` and `S3` still get every other template field (subjects_descr, path_to_calibrations, other cameras, ...); only the fields you explicitly set on the override are replaced.

**When to use this.** Reach for override-by-id whenever the shape of most sequences is uniform and you want to describe only the exceptions. It scales better than dropping discovery and hand-listing everything, and keeps the exceptions visible next to the rule they break.

## Detectors config

Algorithm instances are configured in `./configs/detectors_config.toml`. Each `[algorithms.<name>]` block declares one named instance that can be referenced from the run file's `algorithms` list.

```toml
[algorithms.vitpose_huge]
algorithm_type = "mmpose_2d"
components = ["body_joints"]
camera_names = ["view_top", "view_center"]   # names must match cameras declared in dataset_properties
env_name = "conda:openmmlab"
device = "cuda:0"
pose_config = "td-hm_ViTPose-huge_8xb64-210e_coco-256x192"
...

[algorithms.eth_xgaze]
algorithm_type = "eth_xgaze"
camera_names = ["view_left", "view_right"]   # or use `"*"` to bind to every camera the sequence declares
env_name = "venv:eth_xgaze"
...

[algorithms.gaze_fusion]
algorithm_type = "gaze_fusion"
input_detector_names = [["gaze_individual", "eth_xgaze"]]
...
```

`algorithm_type` selects the detector class to use (str). Multiple instances can share the same `algorithm_type` with different parameters.

### Templates

Templates define shared field sets that algorithm instances can inherit from, avoiding repetition. They are declared under `[templates.<name>]` and referenced via `template = "<name>"`.

```toml
[templates.mmpose_2d_template]
algorithm_type = "mmpose_2d"
camera_names = ["view_top", "view_center"]   # names must match cameras declared in dataset_properties
env_name = "conda:openmmlab"
device = "cuda:0"
...

[algorithms.hrnetw48]
template = "mmpose_2d_template"
components = ["body_joints", "hand_joints", "face_landmarks"]
pose_config = "td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288"
```

Algorithm fields always override template fields. For example, `device = "cpu"` below overrides the `device = "cuda:0"` from the template:

```toml
[templates.mmpose_2d_template]
algorithm_type = "mmpose_2d"
device = "cuda:0"
...

[algorithms.hrnetw48_cpu]
template = "mmpose_2d_template"
device = "cpu"           # overrides template's cuda:0
components = ["body_joints"]
...
```

Templates can themselves inherit from other templates:

```toml
[templates.mmpose_2d_template]
algorithm_type = "mmpose_2d"
camera_names = ["view_top", "view_center"]   # names must match cameras declared in dataset_properties
env_name = "conda:openmmlab"
device = "cuda:0"
...

[templates.hrnetw48_template]
template = "mmpose_2d_template"
components = ["body_joints", "hand_joints", "face_landmarks"]
pose_config = "td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288"

[algorithms.hrnetw48_low_conf]
template = "hrnetw48_template"
min_detection_confidence = 0.3

[algorithms.hrnetw48_high_conf]
template = "hrnetw48_template"
min_detection_confidence = 0.8
```


## Predictions mapping

The config file `./configs/predictions_mapping.toml` contains information about mappings between different data identifiers. These are, for example, different conventions for selecting and naming human body joints, also called keypoints. The mappings are primarily used for internal purposes.


## Visualizer Config

Defined in `./configs/visualizer_config.toml`.

Visualizer Config consists of three main part io, media, and component specifications.

```toml
spawn_viewer = true                                                        # if true, will spawn window with GUI

[io]
dataset_folder = "<datasets_folder_path>"                                 # main dataset folder
dataset_name = 'communication_multiview'                                  # dataset of the video
video_name = 'communication_multiview__sequence_xyz_s0_l-1'              # name of video result folder
nice_tool_input_folder = "<output_folder_path>/nicetoolbox_input/<cur_dataset_name>_<cur_sequence_id>" # pre-processed input data
nice_tool_output_folder = "<output_folder_path>/experiments"              # NICE Toolbox experiment output
experiment_folder = "<output_folder_path>/experiments/<yyyymmdd>"         # select single NICE Toolbox experiment output folder
experiment_video_folder = "<experiment_folder>/<video_name>"              # NICE Toolbox output folder for the specific video
experiment_video_component = "<experiment_video_folder>/<cur_component_name>" # NICE Toolbox output folder for the specific component

[media]                                # each Media session shows one video results.
multi_view = true                      # true if you have multiple cameras, otherwise set it to false
[media.visualize]                      # specify what will be visualized
components = [..]                     # list of components
camera_position = true                 # true if you want to visualize camera position -- requires extrinsic information of the camera
start_frame = 0                        # starting frame for the visualization
end_frame = -1                         # end frame for the visualization, -1 means process until the end of the video
visualize_interval = 1                 # 1 means visualize every frame; change the parameter accordingly if you want to visualize every x frames
```
### Configuring Component Data Display in Rerun Windows
You can control which data will be shown in specific rerun windows by adjusting the `media.component.canvas` items
The keys (like `3d` or `2d_interpolated`) represent different type of data provided by that component.
The value lists define which canvases (rerun windows) will show the data
1. `3D_Canvas`: This shows data in the 3D canvas. It is only for multi-view datasets. (Do not change the canvas name).
2. Cameras: Data will be visualized on that specific camera image. The camera name must match the camera placeholder names in dataset_properties.toml
3. Metrics - The displays the data as plots.
4. Empty list: If you don't want the data to be visualized, leave the list empty.

**Configuring Algorithm Display**
Under `media.component`, the algorithms parameter let you choose which algorithms to display.
For example, if you have multiple algorithms (e.g., hrnetw48 and vitpose in the body_joints component),
you can specify which algorithm’s results to show.
If you want to see results from both algorithms, list both names.

**Configuring Appearance**
Under `media.component.appearance`, you can configure the color and radii (the size of the dots and lines).

```toml
# Component: gaze individual - An example for 3D_Canvas and Camera Canvases
[media.gaze_individual]
algorithms = ['multiview_eth_xgaze']  # list of algorithms
[media.gaze_individual.canvas]
3d_filtered = ["3D_Canvas", "view_left", "view_right", "view_top", "view_center"] ## key options 3d, 3d_filtered ## value options: [3D_Canvas], [3D_Canvas, camera names], [camera names], []
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

This Blueprint can be saved using the `Save blueprint...` menu option and reopened later using the
`Open` option. Once you configure the Rerun viewer, it will use the same blueprint for future sessions.
You can reset the layout by clicking `Reset Blueprint.`

![../graphics/rerun_viewer.png](../graphics/rerun_viewer.png)

If your new video does not have certain windows that the old dataset had, unused empty windows may appear.
To get a fresh heuristic layout, reset the blueprint.
