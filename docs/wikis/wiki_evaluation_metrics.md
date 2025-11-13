# Evaluation Metrics

This overview shows the logic of grouping metrics and all the currently available metrics that can be used as part of the evalaution pipeline. We are working on extending the list of support metrics.

## Grouping of Metrics - Metric Types

Metrics are grouped by their input requirements and evaluation semantics. We assume that any metric from a specific metric type can be computed on the supported prediction data types.

## Available Metric Types

- **Point Cloud Metrics**: Any metrics that compare prediction and ground truth (GT) pairs in the context of point clouds. Requires 3D ground truth coordinates.
- **Keypoint Metrics**: Operate on keypoint predictions and do not require GT
- **Categorical Metrics**: Operate on discrete labels or classifications.

### Point Cloud Metrics

Metrics that require ground truth for keypoint coordinates or any other point cloud data.

**Available Metrics:**
- **`jpe`** (Joint Position Error): Euclidean distance between predicted and ground truth.

**Configuration:**
```toml
[metrics.point_cloud_metrics]
metric_names = ["jpe"]
gt_required = true
```

**Use Case:** Evaluating pose estimation accuracy in calibrated multi-view setups.

---

### Keypoint Metrics

Metrics that analyze keypoint predictions without requiring ground truth. Requires 3D keypoint predictions.

**Available Metrics:**
- **`jump_detection`**: Detects temporal discontinuities in keypoint trajectories. Using fixed diameter thresholds for each joint or keypoint, this metric checks each frame and flags a jump when the displacement from the previous prediction exceeds the joint-specific threshold. Only available for 3D keypoint predictions.
- **`bone_length`**: Measures bone length consistency over time. For each defined bone (a set of two adjacent joints), we compute the euclidean distance between the 2 points. Only available for 3D keypoint predictions.

**Configuration:**
```toml
[metrics.keypoint_metrics]
metric_names = ["jump_detection", "bone_length"]
gt_required = false
gt_components = ["body_joints"]
keypoint_mapping_file = "configs/predictions_mapping.toml"
```

**Parameters:**
- **`gt_components`**: Component names used to extract ground truth bone definitions
- **`keypoint_mapping_file`**: Maps algorithm-specific keypoint names to canonical skeleton structure, provides the definitions of bones for bone_length computation and also the joint diameter sizes used to compute the threshold for jump_detection.

**Use Case:** Quality control for pose estimation without ground truth annotations.

---

### Categorical Metrics

Metrics for classification tasks (e.g., emotion recognition or mutual gaze).

**Available Metrics:**
- **`accuracy`, `precision`, `recall` and `f1_score`**: Count-based classification metrics that are tracked across videos. Results will be exported via csv based summary reports.

**Configuration:**
```toml
[metrics.categorical_metrics]
metric_names = ["accuracy", "precision", "recall", "f1_score"]
gt_required = true
```

**Use Case:** Validating classification outputs.