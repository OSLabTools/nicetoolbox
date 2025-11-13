# NICE Toolbox Evaluation Overview

This tutorial guides you through the evaluation pipeline of the NICE Toolbox.

## Table of Contents

- [NICE Toolbox Evaluation Overview](#nice-toolbox-evaluation-overview)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Configuration Setup](#configuration-setup)
    - [Machine Specific Paths](#machine-specific-paths)
    - [Evaluation Config](#evaluation-config)
      - [(1) Global Settings:](#1-global-settings)
      - [(2) Evaluation IO](#2-evaluation-io)
      - [(3) Metric selection and configuration](#3-metric-selection-and-configuration)
      - [(4) Metric aggregation summaries](#4-metric-aggregation-summaries)
  - [Running the Evaluation](#running-the-evaluation)
    - [Step 1: Input selection](#step-1-input-selection)
    - [Step 2: Configure Metrics](#step-2-configure-metrics)
    - [Step 3: Run Evaluation](#step-3-run-evaluation)
    - [Step 4: Monitor Progress and check the results](#step-4-monitor-progress-and-check-the-results)
  - [Summary Generation](#summary-generation)
    - [Customizing Summaries](#customizing-summaries)
    - [Regenerating Summaries](#regenerating-summaries)
  - [Understanding Results](#understanding-results)
    - [Output Structure](#output-structure)
    - [What is inside a single .npz file?](#what-is-inside-a-single-npz-file)
    - [Result Files](#result-files)
  - [Evaluation Results Wrapper (python)](#evaluation-results-wrapper-python)

---

## Overview

The NICE Toolbox evaluation pipeline allows for comprehensive assessment of NICE toolbox detectors and algorithms using a variety of metrics. It supports evaluations with and without ground truth annotations, making it versatile for different experimental setups.

**Key Features:**
- Quality metrics for evaluation without ground truth annotations. When no labels are available or labeling is expensive.
- Classic evaluation of detector prediction and ground truth annotation pairs for selected datasets. 
- Detailed per-frame scores and aggregated summary statistics.
- API for advanced analysis of raw evaluation results.

> [!NOTE]
> Currently we have only implemented a limit set of datasets with GT annotations. We are working on simplifying the process of adding user datasets with their own label set. 


## Configuration Setup

### Machine Specific Paths

Ensure `./machine_specific_paths.toml` contains:

```toml
datasets_folder_path = "/absolute/path/to/datasets"
output_folder_path = "/absolute/path/to/outputs"
```
This is set up by default during the installation.

### Evaluation Config

The main configuration file is located at `./configs/evaluation_config.toml`. 

#### (1) Global Settings:

- **`git_hash`**: Automatically filled with current git commit hash
- **`device`**: Computing device (`"cpu"` or `"cuda:0"`)
- **`batchsize`**: Number of frames processed per batch (higher = faster but more memory)
- **`verbose`**: Enable detailed logging and CSV exports including automatic summaries
- **`skip_evaluation`**: Skip main loop and only regenerate summaries from existing results

Here is an example:
```toml
# Global Settings
git_hash = "<git_hash>"
device = "cuda:0"
batchsize = 10000
verbose = true
skip_evaluation = false
```

#### (2) Evaluation IO
- **`experiment_name`**: The evaluation pipeline needs access to the results of the NICE toolbox detectors. Using the first 2 lines of the IO config, you specify the path to the folder of the detector results via the experiment_name and experiment_folder field.
- **`output_folder`**: The outputs and results of the evaluation will be exported to this folder.
 
```toml
[io]
# Evaluation input folders (NICE toolbox detector output folders)
experiment_name = "<yyyymmdd>"  # Default placeholder for experiments run on the same day.
experiment_folder = "<output_folder_path>/experiments/<experiment_name>"  # Default folder for all experiments
# Evaluation output folders
output_folder = "<experiment_folder>_eval"  # Output folder of evaluation 
eval_visualization_folder = "<output_folder>/visualization"
```

#### (3) Metric selection and configuration

Metrics are grouped under categories called `metric_type` that are compatible with specific data types. Each `metric_type` has its own config and selection. For a detailed explanation and an overview of all available metrics, please refer to the [evaluation metrics wiki page](../wikis/wiki_evaluation_metrics.md).

In this section, please select the metrics to be run. For each metric type, you can specify the list of metric names to be computed with the `metric_names` field.

```toml
[metrics.point_cloud_metrics]  # Metric type (here: point cloud metrics)
metric_names = ["jpe"]         # List of metric names to compute
gt_required = true

[metrics.keypoint_metrics]
metric_names = ["jump_detection", "bone_length"]
gt_required = false
gt_components = ["body_joints", "hand_joints", "face_landmarks"]
keypoint_mapping_file = "configs/predictions_mapping.toml"

[metrics.categorical_metrics]
metric_names = ["accuracy", "precision", "recall", "f1_score"]
gt_required = true
```

#### (4) Metric aggregation summaries
Here you can define multiple summaries with different aggregation settings that are automatically computed after evaluation when `verbose` is set to true. Please refer to the section [below](#summary-generation) for more details.
```toml
[summaries.bone_length_report]                              # Name of the summary                                          
metric_names = ["bone_length"]                              # List of metric names to include in the summary
aggr_functions = ["mean", "std", "min", "max"]              # List of aggregation functions to apply
filter = {dataset = "communication_multiview"}              # Filters to apply before aggregation
aggregate_dims = ["sequence", "person", "camera", "frame"]  # Dimensions to aggregate over
```

## Running the Evaluation

### Step 1: Input selection

Ensure your experiment folder contains detector outputs (`.npz` files) and select it inside the `evaluation_config` under `[IO]`. See the evaluation config descriptions [above](#2-evaluation-io).

Ground truth annotations (if required by selected metrics) need to be processed and stored inside the dataset folder. A tutorial on how to add custom datasets with annotations will be available soon. Please contact us for more information or future collaborations.


### Step 2: Configure Metrics

Edit `./configs/evaluation_config.toml` to select desired metrics:

```toml
[metrics.point_cloud_metrics]
metric_names = ["jpe"]

[metrics.keypoint_metrics]
metric_names = ["bone_length"]
```

Optionally, edit the [summary reports](#summary-generation) list as well.

### Step 3: Run Evaluation

```bash
cd /path/to/nicetoolbox/
envs\nicetoolbox\Scripts\activate # Windows
source ./envs/nicetoolbox/bin/activate # Linux
run_evaluation
```

### Step 4: Monitor Progress and check the results

Check the log file at `/path/to/output_folder/evaluation.log` and verify the success of the pipeline by looking at the [results](#understanding-results). 


## Summary Generation

After evaluation completes, summaries are automatically generated in `<output_folder>/csv_files/summaries/`. For this to happen, you need to set `verbose=true` inside the evaluation config under the global settings up top. In addition, you need to configure summaries in the 4th part of the evaluation config:

```toml
# === (4) Metric aggregation summaries ===

# Here you can define multiple summaries with different aggregation settings
# that are automatically computed after evaluation when `verbose` is set to true. 
```

Based on the example dataset, there are already a few summaries provided that showcase the flexibility of these automatic reports.

### Customizing Summaries

Edit summaries in the evaluation config. Here you can customize the following per summary:
- The name of the summary report. It will be used for exporting the results.

  ```toml
  # Name of the summary (Only change the part after "summaries.")
  [summaries.bone_length_report]
  ```
- The metrics to include in the summary.

  ```toml
  metric_names = ["bone_length"]  # List of metric names to include in the summary
  ```
- The aggregation functions to apply (e.g., mean, std, min, max).

  ```toml
  aggr_functions = ["mean", "std", "min", "max"]  # List of aggregation functions to apply
  ```
- Filters to apply before aggregation. Here you can flexibly filter along all your data dimensions. These include: metric_name, dataset, sequence, component, algorithm, metric_type, person, camera, label. Add these keys to the `filter` dictionary and add the values that you want to query in your data.
  ```toml
  filter = {metric_name=["jump_detection"], label=["left_wrist", "right_wrist"]}
  #  Using this filter, any aggregation selected will only be applied and exported to the jump_detection metric with further specification of the keypoints of interest (only the two wrists joints).
  ```

- The dimensions to aggregate over. Please select a subset of sequence, person, camera, frame and label.

  ```toml
  aggregate_dims = ["person", "camera", "frame", "label"]  # Dimensions to aggregate over
  ```


### Regenerating Summaries

To regenerate summaries without re-running evaluation, set

```toml
skip_evaluation = true
```

in the first part of the `evaluation_config` and then rerun the nicetoolbox evaluation with:

```bash
run_evaluation
```

---

## Understanding Results

### Output Structure

```bash
<output_folder>/
├── evaluation.log
├── config_<time>.log
├── <dataset_name>__<sequence_ID>/
│   ├── <component_name>/
│   │   ├── <algorithm_name>__<metric_type>.npz
│   │   └── ...  # One npz file for each metric_type
│   └── csv_files/
│       ├── summaries/
│       │   ├── <summary_name>.csv
│       │   └── ...  # One csv file for each configured summary report
│       ├── <dataset_name>__<sequence_ID>_<component>_<algorithm>__<metric_type>_<metric_name>.csv
│       └── ...  # One csv file for each metric
└── visualization/
    └── Coming Soon
```

### What is inside a single .npz file?

Lets look at the body_joints (component) results for vitpose (algorithm):

```bash
├── vitpose__keypoint_metrics.npz
│   ├── bone_length.npy              # numpy array with shape ( # persons, # cameras, # frames, # bone lengths (# bone names) )
│   ├── jump_detection.npy           # numpy array with shape ( # persons, # cameras, # frames, # jump metric scores (# body joints) )
│   └── data_description.npy         # dictionary of dicts with Dict["axis0"=List[persons], "axis1"=List[cameras], "axis2"=List[frames], "axis3"=List[labels]]
├── vitpose__point_cloud_metrics.npz
│   ├── jpe.npy                      # numpy array with shape ( # persons, # cameras, # frames, # jpe metric scores (# body joints) )
│   └── data_description.npy         # dictionary of dicts with Dict["axis0"=List[persons], "axis1"=List[cameras], "axis2"=List[frames], "axis3"=List[labels]]
```

### Result Files

Each `.npz` file contains:
- **Metric arrays**: Multi-dimensional arrays indexed by `(person, camera, frame, label)`
- **Description dictionaries**: Metadata called `data_description` for each metric array. The data_description is a dictionary that describes each dimension of the given output npy arrays

  For example:
  ```python
  data_description = {
    "bone_length": {
      axis0=["person_left", "person_right"],  # A list of persons in the video
      axis1=["camera_front", "camera_top"],  # A list of camera perspectives in multi-camera setups
      axis2=[frames]  # The frames that were processed
      axis3=["left_lower_leg", "right_lower_leg", ..., "right_upper_arm"]  # The labels (Bone names -> lengths)
    },
    "jump_detection": {
      ...  # Axis 0 to 2 would be equal
      axis3=["nose", "left_wrist", "right_wrist", ..., "right_toe"]  # The labels (Keypoints/Joints -> Jumps)
    }
  }

---

## Evaluation Results Wrapper (python)

We have created a simple python API to allow for more complex analysis of the raw evaluation results with the following features:
- Fine grained querying/filtering of high dimensional data
- Flexible aggregations
- Converting to pandas DataFrame
- Exporting to .csv files

Please refer to the [tutorial](tutorial6_results_wrapper.md) to get started on using the pandas based API.