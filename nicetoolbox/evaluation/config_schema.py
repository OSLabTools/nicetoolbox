"""
Data models for evaluation configuration files and final evaluation config.
Validated with Pydantics BaseModel.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# -----------------------------------------------------
# From file: nicetoolbox/configs/evaluation_config.toml
class IOConfig(BaseModel):
    experiment_folder: Path
    output_folder: Path
    eval_visualization_folder: Path
    experiment_io: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MetricTypeConfig(BaseModel):
    metric_type: str
    metric_names: List[str]
    gt_required: bool
    keypoint_mapping_file: Optional[Path] = None  # TODO: Move somewhere else! IO?
    gt_components: Optional[List[str]] = Field(default_factory=list)


class GlobalEvalConfig(BaseModel):  # Top-level keys in evaluation_config.toml
    git_hash: Optional[str] = None
    device: str = "cuda:0"
    batchsize: int = 50
    verbose: bool = True
    skip_evaluation: bool = False

    # List of metric type configs
    metric_types: Dict[str, MetricTypeConfig] = Field(default_factory=dict)


class AggregationConfig(BaseModel):
    summary_name: str
    metric_names: List[str]
    aggr_functions: List[str]
    filter: Dict[str, str | List[str]] = Field(default_factory=dict)
    aggregate_dims: List[str] = Field(default_factory=list)

    @field_validator("aggregate_dims", mode="before")
    def _validate_aggregate_dims(cls, values: List[str]) -> List[str]:
        """Validate that aggregate_dims values are a subset of allowed values."""
        allowed = {"sequence", "person", "camera", "label", "frame"}
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(
                f"Invalid aggregate_dims entries: {sorted(invalid)}. "
                f"Allowed values are {sorted(allowed)}."
            )
        return values


# --------------------------------------------------------
# From file: <nicetoolbox_output_folder>/config_**_**.toml
class DatasetPropertiesEvaluation(BaseModel):
    annotation_components: List[str]
    metric_types: List[str]


class DatasetProperties(BaseModel):
    dataset_name: str
    session_IDs: List[str]
    sequence_IDs: List[str]
    cam_front: str
    cam_top: str
    cam_face1: str
    cam_face2: str
    subjects_descr: List[str]
    path_to_calibrations: Path
    data_input_folder: Path
    start_frame_index: int
    fps: int
    path_to_annotations: Path
    evaluation: List[DatasetPropertiesEvaluation]
    synonyms: Optional[dict] = Field(default_factory=dict)
    cam_sees_subjects: Optional[dict] = Field(default_factory=dict)


class RunConfigVideo(BaseModel):
    session_ID: str
    sequence_ID: str
    video_start: int
    video_length: int


class RunConfig(BaseModel):
    dataset_name: str
    components: List[str]
    videos: List[RunConfigVideo]


# ---------------------------------------------
# Final eval config for main loop iterator:
# (run_config, dataset_properties, eval_config)


class EvaluationConfig(BaseModel):
    # Global settings relevant to the task
    device: str
    verbose: bool
    batchsize: int

    # From DatasetPropertiesEvaluation (for this specific task)
    prediction_components: Dict[str, List[str]]
    annotation_components: Dict[str, List[str]]

    metric_types: Dict[str, MetricTypeConfig]
    component_algorithm_mapping: Dict[str, List[str]] = Field(default_factory=dict)
