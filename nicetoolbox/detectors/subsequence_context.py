from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from ..configs.models.video_timestamp import VideoTimestamp
from ..configs.placeholders import resolve_placeholders
from ..configs.schemas.dataset_properties import SequenceConfig
from ..configs.schemas.detectors_config import DetectorsConfig
from ..configs.schemas.detectors_run_file import DetectorsRunFile, DetectorsRunIO, LoggingLevelEnum, SubsequenceConfig
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..configs.schemas.predictions_mapping import PredictionsMappingConfig


class SubsequenceContext(BaseModel):
    """
    Immutable context for processing a single subsequence.

    Created by Configuration.iter_sequence_contexts(), holds all resolved
    configuration needed for one subsequence (sequence with start/stop timestamps).
    Discarded after processing.

    All placeholders (except <cur_component_name> and <cur_algorithm_name>
    in IO paths) are fully resolved at construction time.

    Note: frozen=True only prevents reassignment of top-level attributes.
    Nested models are mutable but should be treated as immutable by convention.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # subsequence specific info
    dataset_name: str
    run_sequence: SubsequenceConfig
    sequence_properties: SequenceConfig

    # configs resolved for this specific subsequence
    machine: MachineSpecificConfig
    run: DetectorsRunFile
    detectors_config: DetectorsConfig
    predictions_mapping: PredictionsMappingConfig

    # injected from config handler
    log_file: Path

    # -------------------------------------------------------------------------
    # Convenience Properties
    # -------------------------------------------------------------------------
    @property
    def all_camera_names(self) -> List[str]:
        return list(self.sequence_properties.video.cameras.keys())

    @property
    def log_level(self) -> LoggingLevelEnum:
        return self.run.log_level

    @property
    def algorithms(self) -> List[str]:
        return self.run.algorithms

    @property
    def io(self) -> DetectorsRunIO:
        return self.run.io

    @property
    def sequence_id(self) -> str:
        return self.run_sequence.sequence_id

    @property
    def video_start(self) -> int | VideoTimestamp:
        return self.run_sequence.video_start

    @property
    def video_length(self) -> int | VideoTimestamp:
        return self.run_sequence.video_length

    @property
    def subjects_descr(self) -> List[str]:
        return self.sequence_properties.subjects_descr

    @property
    def calibration_path(self) -> Optional[Path]:
        path = self.sequence_properties.path_to_calibrations
        return Path(path) if path else None

    # -------------------------------------------------------------------------
    # Config Access (no resolution needed - already resolved)
    # -------------------------------------------------------------------------

    def get_detector_config(self, algorithm_name: str) -> BaseModel:
        """
        Get the pre-resolved configuration for a detector.

        Args:
            algorithm_name: Name of the algorithm (e.g., 'hrnetw48', 'velocity_body')

        Returns:
            Resolved configuration dict ready for detector initialization.
            Includes the injected 'visualize' flag.

        Raises:
            KeyError: If algorithm was not in the selected algorithms for this video
        """
        if algorithm_name not in self.detectors_config.algorithms:
            raise KeyError(
                f"Algorithm '{algorithm_name}' not found. "
                f"Available: {list(self.detectors_config.algorithms.keys())}"
            )
        return self.detectors_config.algorithms[algorithm_name]

    # -------------------------------------------------------------------------
    # IO Path Helpers (handle remaining component/algorithm placeholders)
    # -------------------------------------------------------------------------

    def get_detector_folder(self, component: str, algorithm: str, folder_type: str) -> Path:
        """
        Get resolved detector-specific folder path.

        The IO paths contain <cur_component_name> and <cur_algorithm_name>
        placeholders that are resolved here based on the specific detector.

        Args:
            component: Component name (e.g., 'body_joints')
            algorithm: Algorithm name (e.g., 'hrnetw48')
            folder_type: One of 'output', 'visualization', 'additional', 'run_config', 'result'

        Returns:
            Resolved Path to the requested folder
        """
        template_map = {
            "output": self.io.detector_out_folder,
            "visualization": self.io.detector_visualization_folder,
            "additional": self.io.detector_additional_output_folder,
            "run_config": self.io.detector_run_config_path,
            "result": self.io.detector_final_result_folder,
        }

        if folder_type not in template_map:
            raise ValueError(f"Unknown folder_type '{folder_type}'. Valid: {list(template_map.keys())}")

        path = template_map[folder_type]
        resolved = resolve_placeholders(path, {"cur_component_name": component, "cur_algorithm_name": algorithm})
        return resolved
