# Evaluate Detector Outputs

This tutorial shows how to run the evaluation pipeline on your detector outputs. The evaluation module computes configurable metrics over your experiment's NPZ files and produces CSV summaries and plots.

For a full reference of all available metrics and their parameters, see the [Evaluation Metrics wiki](../wikis/wiki_evaluation_metrics.md).

```{contents} Contents
:depth: 3
```


## 1. Prerequisites

Before running the evaluation, you need:

- A completed detector run with NPZ output files in your experiment folder (see [Tutorial 1](tutorial1_dataset_single_view.md)).
- *(For accuracy metrics)* Ground-truth annotation NPZ files stored in a folder that follows the same component structure as your experiment output.

No additional installation is needed — the evaluation pipeline is included in the standard NICE Toolbox environment.


## 2. Configure the Evaluation

The evaluation is driven by `./configs/evaluation_config.toml`. A ready-to-use template with all supported metrics is provided in the repository.

### Top-level settings

The most important top-level fields are:

```toml
# default experiment path — points to the detector run you want to evaluate
default_experiment_name = "<yyyymmdd>"
default_experiment = "<output_folder_path>/experiments/<default_experiment_name>"
output_folder = "<default_experiment>_eval"   # where evaluation results are saved

# path to bone and joint definitions (needed for bone_length metric)
predictions_mapping = "<configs_folder_path>/predictions_mapping.toml"

# run all metrics defined below, or list specific names to run a subset
run_metrics = "*"
```

`output_folder` defaults to your experiment folder with an `_eval` suffix, so the results always sit next to the detector outputs they came from.

### Defining metric instances

Each metric instance is a TOML table under `[metrics.<name>]`. The `metric_type` field selects which metric to run; the name you give the table (`<name>`) is used to label the output folder. You can define multiple instances of the same metric type with different parameters.

```toml
[metrics.my_metric]
metric_type = "bone_length"   # selects the metric
# ... metric-specific parameters
```

### Example: no ground truth

The `missing_points` and `bone_length` metrics work directly on your detector outputs without any annotations. The example below computes a single detection-rate number per algorithm across all data:

```toml
[metrics.missing_points_3d]
metric_type = "missing_points"
predictions = { component = "body_joints", npz_key = "3d" }
missing_points_summary_group_by = []        # pool everything — one row per algorithm
missing_points_summary_aggr = [
    { name = "total_count",    fn = "count" },
    { name = "miss_count",     fn = "sum" },
    { name = "detection_rate", fn = "one_minus_mean" }
]
```

To get a breakdown per joint, change `group_by` to `["label"]`. To get the most detailed breakdown possible, use `"*"`.

### Example: with ground truth annotations

Accuracy metrics require a ground-truth NPZ file. The `source = "annotation"` field tells the pipeline to look for the ground truth inside the `dataset_properties.toml` config.

```toml
# dataset_properties.toml
[communication_multiview]
sequences = [{sequence_id = "sequence_xyz"}]
# ... template with dataset_root, video.cameras, etc. (see wiki) ...

# ======== Evaluation configuration ========
[communication_multiview.template.annotation]
annotations_folder = "<dataset_root>/annotations"
# labels for different components — <sequence_id> resolves per-sequence
[communication_multiview.template.annotation.components]
gaze_interaction = {path = "<annotations_folder>/<sequence_id>_gaze.npz"}
body_joints      = {path = "<annotations_folder>/<sequence_id>_body_joints.npz"}
```

Now we can specify PCK metric to use the `body_joints` annotation as the ground truth for the algorithm:

```toml
[metrics."body_pck@10px"]
metric_type = "pck"
threshold = 10
predictions = { component = "body_joints", npz_key = "2d_interpolated" }
ground_truth = { source = "annotation", component = "body_joints", npz_key = "2d" }
summary_group_by = ["label"]        # one row per joint name
summary_aggr = [
    { name = "pck",           fn = "mean"  },
    { name = "correct_count", fn = "sum"   },
    { name = "support",       fn = "count" }
]
```

Alternatevily, you can use other detectors predictions as the ground truth. For example, let's say we believe that `vitpose_huge` will be our ground truth and we want to compare it with `hrnetw48`:

```toml
[metrics.body_pck_hrnet_vs_vitpose]
metric_type = "pck"
threshold = 10
predictions = { component = "body_joints", npz_key = "2d_interpolated", algorithm = "hrnetw48" }
ground_truth = { component = "body_joints", npz_key = "2d_interpolated", algorithm = "vitpose_huge" }
summary_group_by = ["label"]        # one row per joint name
summary_aggr = [
    { name = "pck",           fn = "mean"  },
    { name = "correct_count", fn = "sum"   },
    { name = "support",       fn = "count" }
]
```

### Grouping and aggregation

#### Grouping

The `group_by` parameter controls how fine-grained the output rows are. Results are computed separately for each unique combination of the listed dimensions; everything else is pooled together.

| Dimension | Breaks results down by |
| - | - |
| `dataset` | dataset name |
| `session` | recording session |
| `sequence` | video sequence |
| `subject` | individual person |
| `camera` | camera view |
| `label` | joint or keypoint name |

Two special values cover the common extremes:
- `[]` — pool everything, one row per algorithm
- `"*"` — maximum detail, one row per every combination of all available dimensions

#### Aggregation

The `summary_aggr` parameter selects which statistical columns appear in the output CSV. You can list function names directly or provide a custom column name:

```toml
summary_aggr = ["mean", "std"]                              # column names match function names
summary_aggr = [{ name = "detection_rate", fn = "one_minus_mean" }]  # custom name
```

For the full list of available functions, see [Aggregation Functions](../wikis/wiki_evaluation_metrics.md#aggregation-functions) in the wiki.

Keypoint metrics (bone length, missing points, PCK, distance error) use **macro aggregation** — each group is scored independently. Categorical metrics (confusion matrix, ROC AUC, PR curve) use **micro aggregation** — frames are pooled before computing the metric. For a full explanation with examples, see the [Evaluation Metrics wiki](../wikis/wiki_evaluation_metrics.md#micro-and-macro-aggregation).

## 3. Run the Evaluation

Activate your environment and run:

```bash
cd /path/to/nicetoolbox/

# LINUX
source ./envs/nicetoolbox/bin/activate

# WINDOWS
envs\nicetoolbox\Scripts\activate

run_evaluation
```

By default the command reads `./configs/evaluation_config.toml`. You can override this with the `--eval_config` flag if needed.

Progress is printed to the console and also written to the log file defined in `log_file_path` (defaults to `<output_folder>/evaluation.log`).


## 4. Read the Results

The output folder (e.g. `experiments/20240101_eval/`) contains one subfolder per metric instance:

```
<output_folder>/
├── missing_points_3d/
│   ├── summary.csv                  # aggregated scores — start here
│   ├── npz/                         # per-frame raw values
│   └── visualization/               # auto-generated plots (.png)
├── body_pck@10px/
│   ├── summary.csv
│   ├── npz/
│   └── visualization/
└── evaluation.log
```

**`summary.csv`** is the first file to look at. Each row is a unique combination of the grouping dimensions you specified, plus one column per aggregation function. Open it in any spreadsheet tool or load it with `pandas.read_csv()`.

**`visualization/`** contains automatically generated plots — heatmaps, candle plots, and curve visualisations — saved as PNG files.

**`npz/`** contains the raw per-frame arrays in the same structure as detector outputs, which you can use for custom analysis.

For a detailed explanation of each metric's parameters and aggregation options, see the [Evaluation Metrics wiki](../wikis/wiki_evaluation_metrics.md).
