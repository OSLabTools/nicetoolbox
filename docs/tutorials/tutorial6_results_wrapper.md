# Evaluation Results Wrapper Tutorial (python)

The Evaluation Results Wrapper provides a simple, pandas-like interface for querying and analyzing evaluation results stored in `.npz` files.

## Table of Contents

- [Evaluation Results Wrapper Tutorial (python)](#evaluation-results-wrapper-tutorial-python)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Quick Start](#quick-start)
  - [Core Operations](#core-operations)
    - [Loading Results](#loading-results)
    - [Inspecting Data](#inspecting-data)
    - [Reset to Original View:](#reset-to-original-view)
    - [Querying / Filtering](#querying--filtering)
    - [Aggregation](#aggregation)
    - [Exporting Results](#exporting-results)
    - [Chainable operations](#chainable-operations)
  - [Common Use Cases](#common-use-cases)
    - [Use Case 1: Compare Algorithms Across Cameras](#use-case-1-compare-algorithms-across-cameras)
    - [Use Case 2: Multi-Statistic Summary for a Dataset](#use-case-2-multi-statistic-summary-for-a-dataset)
    - [Use Case 3: Export Subset for External Analysis](#use-case-3-export-subset-for-external-analysis)
    - [Use Case 4: Identify Problematic Labels](#use-case-4-identify-problematic-labels)
  - [Best Practices](#best-practices)
    - [Questions or Issues?](#questions-or-issues)


## Overview

The **`EvaluationResults`** class wraps evaluation metrics stored in `.npz` files and provides a pandas DataFrame–backed API for:

- **Loading and indexing** multi-dimensional evaluation results (persons, cameras, frames, labels, etc.)
- **Filtering** by dataset, sequence, algorithm, component, person, camera, metric name, and label
- **Aggregating** metrics with built-in functions (mean, std, min, max, median, etc.)
- **Exporting** results to CSV or DataFrames for downstream analysis

**Key Benefits:**

- No manual `.npz` parsing
- Consistent query and aggregation interface
- Seamless integration with pandas for custom analysis

## Quick Start

```python
from pathlib import Path
from nicetoolbox.evaluation.results_wrapper import EvaluationResults

# Load results from evaluation output folder
root = Path("/path/to/output_folder/20240315_eval")
results = EvaluationResults(root=root, memory_optimized=True)

# Inspect available data
print(results)

# Query specific algorithm and person
results.query(algorithm="hrnetw48", person="p1")

# Aggregate and export
agg_df = results.aggregate(group_by=["algorithm", "metric_name"]).to_dataframe()
print(agg_df)
```


## Core Operations

### Loading Results

Initialize from an evaluation output folder containing `.npz` metric files:

```python
from pathlib import Path
from nicetoolbox.evaluation.results_wrapper import EvaluationResults

root = Path("/outputs/experiments/20251115_eval")
results = EvaluationResults(root=root)
```

---

### Inspecting Data

**Quick Overview:**

```python
print(results)
```

**Preview DataFrame:**

```python
df = results.to_dataframe()
print(df.head(10))
```

---

### Reset to Original View:

```python
results.reset()
```

> [!NOTE]
> Each `query()` or `aggregate()` call updates the internal state of the results. 
> Use `reset()` to restore the original dataset when you want to start a new analysis.

---

### Querying / Filtering

Filter the internal DataFrame using keyword arguments. Supports both single values or lists:

**Single Value Queries:**

```python
# Filter by dataset
results.query(dataset="dataset_A")

# Filter by algorithm
results.query(algorithm="hrnetw48")

# Filter by metric name
results.query(metric_name="jpe")
```

**Multi-Value Queries:**

```python
# Select multiple labels
results.query(label=["nose", "left_knee", "right_knee"])

# Select multiple algorithms
results.query(algorithm=["hrnetw48", "vitpose"])
```

**Easily combine queries in a single call:**

```python
results.query(
  dataset="dataset_A",
  algorithm="vitpose",
  person=["p1", "p2"]
)
```

---

### Aggregation

Compute summary statistics grouped by specified columns:

**Basic Aggregation (Default: Mean):**

```python
agg_df = results.aggregate(group_by=["dataset", "algorithm", "metric_name"]).to_dataframe()
print(agg_df)
```

**Output:**

```
       dataset algorithm metric_name  mean_value
0   dataset_A  hrnetw48          jpe       43.2
1   dataset_A   vitpose          jpe       38.7
2   dataset_B  hrnetw48   bone_length     152.3
...
```

**Multi-Statistic Aggregation:**

```python
agg_funcs = ["mean", "std", "min", "max", "median"]
summary = results.aggregate(
    group_by=["algorithm", "metric_name"],
    agg_funcs=agg_funcs
).to_dataframe()
print(summary)
```

**Output:**

```
  algorithm metric_name       mean       std        min        max           median
0  hrnetw48         jpe       43.2       12.1       15.4       98.3          41.5
1   vitpose         jpe       38.7       10.5       12.1       87.6          37.2
```

**Supported Aggregation Functions:**

- `"mean"`, `"std"`, `"min"`, `"max"`, `"median"`, `"sum"`, `"count"`

> [!NOTE]
> Aggregation creates a new `EvaluationResults` instance with the aggregated DataFrame. Use `.to_dataframe()` to access it. Use `.reset()` to restore the original view to all available results.

---

### Exporting Results

**Export to pandas DataFrame:**

```python
df = results.to_dataframe()
```

**Export to CSV:**

```python
output_path = results.to_csv(
    output_dir=Path("./exports"),
    base_name="my_results"
)
print(f"Exported to: {output_path}")
```

### Chainable operations

The functions `reset()`, `query()` and `aggregate()` return the mutated instance of the `EvaluationResults` class. This allows for chaining multiple function class. Below we have provided some common use cases that make use of this. 


## Common Use Cases

### Use Case 1: Compare Algorithms Across Cameras

**Goal:** In the context of human pose estimation (HPE), compare JPE performance of two algorithms broken down by camera.

```python
from pathlib import Path
from nicetoolbox.evaluation.results_wrapper import EvaluationResults

root = Path("/outputs/experiments/20240315_eval")
results = EvaluationResults(root=root)

camera_comparison = (
    results.query(metric_name="jpe")
           .aggregate(group_by=["dataset", "algorithm", "camera", "metric_name"])
           .to_dataframe()
)
print(camera_comparison)
```

**Output:**

```
       dataset algorithm camera metric_name      mean
0   dataset_A  hrnetw48     c1         jpe       42.3
1   dataset_A  hrnetw48     c2         jpe       45.1
3   dataset_A   vitpose     c1         jpe       38.2
4   dataset_A   vitpose     c2         jpe       39.5
6   dataset_B  hrnetw48     c1         jpe       50.4
7   dataset_B  hrnetw48     c2         jpe       52.6
9   dataset_B   vitpose     c1         jpe       45.7
10  dataset_B   vitpose     c2         jpe       47.3
...
```

**Insight:** Identify which cameras have higher errors and which algorithm performs better per viewpoint.

---

### Use Case 2: Multi-Statistic Summary for a Dataset

**Goal:** Generate a detailed summary table with multiple statistics for one dataset.

```python
results.reset()
summary = (
    results.query(dataset="dataset_A")
           .aggregate(
               group_by=["algorithm", "metric_name"],
               agg_funcs=["mean", "std", "min", "max", "median"]
           )
           .to_dataframe()
)
print(summary)
```

**Output:**

```
  algorithm metric_name       mean       std        min        max           median
0  hrnetw48         jpe       43.2       12.1       15.4       98.3          41.5
1  hrnetw48  bone_length      152.3      8.7        130.2      175.4         151.8
2   vitpose         jpe       38.7       10.5       12.1       87.6          37.2
3   vitpose  bone_length      153.1      7.9        135.6      172.3         152.5
```

**Insight:** Easily include multiple statistics in reports or e.g. compare robustness (std) across algorithms.

---

### Use Case 3: Export Subset for External Analysis

**Goal:** Export specific labels (nose and knees) aggregated by dataset, sequence, and algorithm.

```python
output_folder = Path("./exports")
output_path = (
    results.reset()
           .query(label=["left_knee", "right_knee"])
           .aggregate(group_by=["dataset", "sequence", "algorithm", "label"])
           .to_csv(output_dir=output_folder, base_name="knees_summary")
)
print(f"Exported to: {output_path}")
```

**Output:**

```
Exported to: ./exports/nose_and_knees_summary.csv
```

**CSV Content:**

```
dataset,sequence,algorithm,label,mean
dataset_A,seq_01,hrnetw48,left_knee,55.2
dataset_A,seq_01,hrnetw48,right_knee,54.8
dataset_A,seq_01,vitpose,left_knee,50.3
dataset_A,seq_01,vitpose,right_knee,51.1
...
```

**Insight:** Share with collaborators or import into Excel/R for further analysis.

---

### Use Case 4: Identify Problematic Labels

**Goal:** Find joints with the highest mean JPE for a specific algorithm. Given that GT annotations are available.

```python
results.reset()
hrnet_jpe = (
    results.query(dataset="your_dataset", algorithm="vitpose", metric_name="jpe")
           .aggregate(group_by=["label"])
           .to_dataframe()
)
worst_joints = hrnet_jpe.sort_values("mean_value", ascending=False).head(5)
print(worst_joints)
```

**Output:**

```
          label  mean_value
17   left_ankle       78.2
18  right_ankle       76.5
12   left_wrist       65.3
13  right_wrist       63.8
9    left_knee        58.1
```


## Best Practices

1. **Chain operations for clarity:** `results.query(...).aggregate(...).to_dataframe()`
2. **Reset between analyses:** Use `reset()` to avoid stale filters


### Questions or Issues?

Refer to the [Evaluation Tutorial](tutorial5_evaluation.md), open an issue on our GitHub or contact us.