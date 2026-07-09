from enum import Enum
from pathlib import Path
from typing import List

from pydantic import BaseModel, NonNegativeInt, PositiveInt, PrivateAttr

from nicetoolbox_core.errors import ErrorLevel

from ..models.video_timestamp import VideoTimestamp


# Enum of python logging levels
class LoggingLevelEnum(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class DetectorsRunIO(BaseModel):
    """
    I/O paths configuration in detectors pipeline.
    """

    experiment_name: str
    out_folder: Path
    csv_out_folder: Path

    out_sub_folder_name: str
    out_sub_folder: Path

    nicetoolbox_input_folder: Path

    code_folder: Path
    dataset_properties: Path
    detectors_config: Path
    predictions_mapping: Path
    assets: Path
    asset_manifest: Path

    detector_out_folder: Path
    detector_visualization_folder: Path
    detector_additional_output_folder: Path
    detector_run_config_path: Path
    detector_final_result_folder: Path


class SubsequenceConfig(BaseModel):
    """Sequence selection entry in the detectors pipeline."""

    sequence_id: str
    video_start: NonNegativeInt | VideoTimestamp  # can be frame index or timestamp
    video_length: int | VideoTimestamp  # frame index, timestamp or -1 for full length


class ResolvedSubsequenceConfig(BaseModel):
    """
    Resolved subsequence configuration, containing actual video fps, start and length frame indexes.
    Used downstream for visualizer and evaluation.
    """

    sequence_id: str
    video_start: NonNegativeInt  # resolved to frame index
    video_length: PositiveInt  # resolved to frame count
    fps: PositiveInt


class DetectorsRunConfig(BaseModel):
    """
    Single dataset config in detectors pipeline.
    Contains settings for sequences to run.
    """

    sequences: List[SubsequenceConfig]

    # Runtime fields
    _dataset_name: str = PrivateAttr()


# Top-level config in detectors_run_file.toml
class DetectorsRunFile(BaseModel):
    """
    Configuration for single detector run pipeline.
    Contains settings for datasets processing, components and I/O paths.
    """

    visualize: bool
    save_csv: bool

    error_level: ErrorLevel
    log_level: LoggingLevelEnum

    check_missing_detectors_dependencies: bool
    skip_gated_models_errors: bool

    algorithms: list[str]
    run: dict[str, DetectorsRunConfig]
    io: DetectorsRunIO

    def model_post_init(self, _):
        # injecting key into each DetectorsRunConfig
        for name, ds in self.run.items():
            ds._dataset_name = name
