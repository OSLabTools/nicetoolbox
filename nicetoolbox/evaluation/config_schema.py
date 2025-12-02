"""
Data models for evaluation configuration files and final evaluation config.
Validated with Pydantics BaseModel.
"""

from typing import Dict, List

from pydantic import BaseModel, Field

from ..configs.schemas.evaluation_config import EvaluationMetricType

# ---------------------------------------------
# Final eval config for main loop iterator:
# (run_config, dataset_properties, eval_config)


class FinalEvaluationConfig(BaseModel):
    # Global settings relevant to the task
    device: str
    verbose: bool
    batchsize: int

    # From DatasetPropertiesEvaluation (for this specific task)
    prediction_components: Dict[str, List[str]]
    annotation_components: Dict[str, List[str]]

    metric_types: Dict[str, EvaluationMetricType] = Field(default_factory=dict)
    component_algorithm_mapping: Dict[str, List[str]] = Field(default_factory=dict)
