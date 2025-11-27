from pathlib import Path

from pydantic import BaseModel

from .dataset_properties import DatasetProperties
from .detectors_config import DetectorsConfig
from .detectors_run_file import DetectorsRunFile
from .machine_specific_paths import MachineSpecificConfig


class CodeConfig(BaseModel):
    """
    Auto-placeholders saved during detectors run.
    """

    user: str
    git_hash: str
    commit_message: str
    date: str
    time: str
    pwd: Path


class DetectorsExperimentConfig(BaseModel):
    """
    Combined schema for detectors experiment output.
    Describe all relevant configuration state during detectors run.
    Used as an input for visualizer and evaluation.
    """

    run_config: DetectorsRunFile
    machine_specific_config: MachineSpecificConfig
    code_config: CodeConfig
    dataset_config: DatasetProperties
    detector_config: DetectorsConfig
