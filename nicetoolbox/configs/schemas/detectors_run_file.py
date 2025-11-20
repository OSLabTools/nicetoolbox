from pathlib import Path
from typing import List

from pydantic import BaseModel, NonNegativeInt, PositiveInt, PrivateAttr


class DetectorsRunIO(BaseModel):
    """
    I/O paths configuration in detectors pipeline.
    """

    experiment_name: str
    out_folder: Path
    out_sub_folder: Path
    dataset_properties: Path
    detectors_config: Path
    assets: Path

    process_data_to: Path
    data_folder: Path
    tmp_folder: Path
    detector_out_folder: Path
    detector_visualization_folder: Path
    detector_additional_output_folder: Path
    detector_tmp_folder: Path
    detector_run_config_path: Path
    detector_final_result_folder: Path
    csv_out_folder: Path
    code_folder: Path
    conda_path: Path


class RunConfigVideo(BaseModel):
    """
    Video configuration in detectors pipeline.
    """

    session_ID: str
    sequence_ID: str
    video_start: NonNegativeInt
    video_length: PositiveInt


class DetectorsRunConfig(BaseModel):
    """
    Single dataset config in detectors pipeline.
    Contains settings for videos and components to run.
    """

    components: List[str]
    videos: List[RunConfigVideo]

    # Runtime fields
    _dataset_name: str = PrivateAttr()


# Top-level config in detectors_run_file.toml
class DetectorsRunFile(BaseModel):
    """
    Configuration for single detector run pipeline.
    Contains settings for datasets processing, components and I/O paths.
    """

    git_hash: str
    visualize: bool
    save_csv: bool

    component_algorithm_mapping: dict[str, list[str]]
    run: dict[str, DetectorsRunConfig]
    io: DetectorsRunIO

    def model_post_init(self, _):
        # injecting key into each DetectorsRunConfig
        for name, ds in self.run.items():
            ds._dataset_name = name
