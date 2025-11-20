"""
This file contains helper classes to build mock data for testing the data pipeline.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from nicetoolbox.evaluation.config_schema import (
    DatasetProperties,
    EvaluationConfig,
    MetricTypeConfig,
    RunConfig,
    RunConfigVideo,
)


class DiscoveryMockDataBuilder:
    """Builds a consistent set of mock config objects for discovery tests."""

    def __init__(self, dataset_name="mock_dataset"):
        self.dataset_name = dataset_name

    def build_run_config(self, overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
        data = {
            "dataset_name": self.dataset_name,
            "components": ["body_joints"],
            "videos": [
                RunConfigVideo(
                    session_ID="S1", sequence_ID="Seq1", video_start=0, video_length=100
                )
            ],
        }
        if overrides:
            data.update(overrides)
        return RunConfig(**data)

    def build_dataset_properties(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> DatasetProperties:
        data = {
            "dataset_name": self.dataset_name,
            "path_to_annotations": Path(f"/{self.dataset_name}/annotations.npz"),
            "session_IDs": ["S1"],
            "sequence_IDs": ["Seq1"],
            "cam_front": "c1",
            "cam_top": "c2",
            "cam_face1": "c3",
            "cam_face2": "c4",
            "subjects_descr": ["p1"],
            "path_to_calibrations": Path("/calib.npz"),
            "data_input_folder": Path("/data"),
            "start_frame_index": 0,
            "fps": 30,
            "evaluation": [],
            "synonyms": {},
        }
        if overrides:
            data.update(overrides)
        return DatasetProperties.model_validate(data)

    def build_evaluation_config(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> EvaluationConfig:
        data = {
            "device": "cpu",
            "verbose": False,
            "batchsize": 50,
            "prediction_components": {"point_cloud_metrics": ["body_joints"]},
            "annotation_components": {},
            "metric_types": {
                "point_cloud_metrics": MetricTypeConfig(
                    metric_type="point_cloud_metrics",
                    metric_names=["jpe"],
                    gt_required=True,
                )
            },
            "component_algorithm_mapping": {"body_joints": ["vitpose"]},
        }
        if overrides:
            data.update(overrides)
        return EvaluationConfig(**data)
