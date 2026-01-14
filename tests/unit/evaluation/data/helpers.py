"""
This file contains helper classes to build mock data for testing the data pipeline.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from nicetoolbox.configs.schemas.dataset_properties import DatasetConfig
from nicetoolbox.configs.schemas.detectors_run_file import DetectorsRunConfig, RunConfigVideo
from nicetoolbox.configs.schemas.evaluation_config import EvaluationMetricType
from nicetoolbox.evaluation.config_schema import FinalEvaluationConfig


class DiscoveryMockDataBuilder:
    """Builds a consistent set of mock config objects for discovery tests."""

    def __init__(self, dataset_name="mock_dataset"):
        self.dataset_name = dataset_name

    def build_run_config(self, overrides: Optional[Dict[str, Any]] = None) -> DetectorsRunConfig:
        data = {
            "components": ["body_joints"],
            "videos": [RunConfigVideo(session_ID="S1", sequence_ID="Seq1", video_start=0, video_length=100)],
        }
        if overrides:
            data.update(overrides)
        config = DetectorsRunConfig(**data)
        config._dataset_name = self.dataset_name
        return config

    def build_dataset_config(self, overrides: Optional[Dict[str, Any]] = None) -> DatasetConfig:
        data = {
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
        config = DatasetConfig.model_validate(data)
        config._dataset_name = self.dataset_name
        return config

    def build_evaluation_config(self, overrides: Optional[Dict[str, Any]] = None) -> FinalEvaluationConfig:
        data = {
            "git_hash": "ffffffff",
            "skip_evaluation": True,
            "device": "cpu",
            "verbose": False,
            "batchsize": 50,
            "prediction_components": {"point_cloud_metrics": ["body_joints"]},
            "annotation_components": {},
            "metric_types": {
                "point_cloud_metrics": EvaluationMetricType(
                    metric_type="point_cloud_metrics",
                    metric_names=["jpe"],
                    gt_required=True,
                )
            },
            "io": {
                "experiment_name": "<yyyymmdd>",
                "experiment_folder": "<output_folder_path>/experiments",
                "output_folder": "<experiment_folder>_eval",
                "eval_visualization_folder": "<output_folder>/visualization",
            },
            "component_algorithm_mapping": {"body_joints": ["vitpose"]},
        }
        if overrides:
            data.update(overrides)
        return FinalEvaluationConfig(**data)
