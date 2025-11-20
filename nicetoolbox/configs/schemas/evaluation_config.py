from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator


class EvaluationIO(BaseModel):
    """
    I/O paths configuration for evaluation.
    """

    experiment_name: str
    experiment_folder: Path
    output_folder: Path
    eval_visualization_folder: Path

    # Runtime fields
    _experiment_io: dict[str, Any] = PrivateAttr(default_factory=dict)


class EvaluationMetricType(BaseModel):
    """
    Configuration for a single evaluation metric type.
    """

    metric_names: List[str]
    gt_required: bool
    gt_components: Optional[List[str]] = None
    keypoint_mapping_file: Optional[Path] = None

    # Runtime fields
    _metric_type: str = PrivateAttr()


class AggregationConfig(BaseModel):
    """
    Configuration for aggregated metrics summaries.
    """

    metric_names: List[str]
    aggr_functions: List[str]
    filter: dict[str, str | List[str]] = Field(default_factory=dict)
    aggregate_dims: List[str] = Field(default_factory=list)

    # Runtime fields
    _summary_name: str = PrivateAttr()

    @field_validator("aggregate_dims", mode="before")
    def _validate_aggregate_dims(cls, values: List[str]) -> List[str]:
        allowed = {"sequence", "person", "camera", "label", "frame"}
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(
                f"Invalid aggregate_dims entries: {sorted(invalid)}. "
                f"Allowed values are {sorted(allowed)}."
            )
        return values


# Top-level keys in evaluation_config.toml
class EvaluationConfig(BaseModel):
    """
    Configuration schema for evaluation settings.
    Contains I/O paths and metric type definitions.
    """

    git_hash: str
    device: str
    batchsize: int
    verbose: bool
    skip_evaluation: bool

    io: EvaluationIO
    metrics: dict[str, EvaluationMetricType] = Field(default_factory=dict)
    summaries: dict[str, AggregationConfig] = Field(default_factory=dict)

    def model_post_init(self, _):
        # injecting key into each metrics
        for key, value in self.metrics.items():
            value._metric_type = key
        # injecting key into each summary
        for key, value in self.summaries.items():
            value._summary_name = key
