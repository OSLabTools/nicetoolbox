"""
IO module for the NICE toolbox.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from ..utils import check_and_exception as exc
from .subsequence_context import SubsequenceContext


class SequenceIO:
    """
    IO for per-video operations.

    Handles:
    - Video-specific folder creation
    - Detector output paths
    - Data source paths
    - Calibration file access
    """

    # All attributes declared for type checking
    out_folder: Path
    out_sub_folder: Path
    csv_folder: Path
    code_folder: Path
    nice_input_folder: Path
    camera_input_folders: Dict[str, Path]
    calibration_file: Optional[Path]
    _subsequence_context: SubsequenceContext

    def __init__(
        self,
        subsequence_context: SubsequenceContext,
    ):
        """
        Initialize for video processing.

        Args:
            sequence_context: Frozen video runtime configuration
        """
        self._subsequence_context = subsequence_context

        # All paths from resolved IO config
        io = subsequence_context.io
        self.out_folder = io.out_folder
        self.out_sub_folder = io.out_sub_folder
        self.csv_folder = io.csv_out_folder
        self.code_folder = io.code_folder
        self.nice_input_folder = io.nicetoolbox_input_folder

        # Per-camera input folders, created lazily on frames extraction
        self.camera_input_folders = subsequence_context.get_camera_input_folders()

        # Dataset properties
        self.calibration_file = subsequence_context.calibration_path

        # Create folders
        self._create_folders()

    def _create_folders(self) -> None:
        """Create necessary output and data folders."""
        self.out_sub_folder.mkdir(parents=True, exist_ok=True)
        self.csv_folder.mkdir(parents=True, exist_ok=True)
        self.nice_input_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Path Getters
    # -------------------------------------------------------------------------

    def get_calibration_file(self):
        """
        Returns the calibration file path.

        Returns:
            str: The path of the calibration file.
        """
        return self.calibration_file

    def get_inference_path(self, component_name, detector_name):
        """
        Get the file path for the inference script of a given detector.

        Args:
            detector_name (str): The name of the detector.

        Returns:
            str: The file path for the inference script.

        Raises:
            FileNotFoundError: If the inference script file does not exist.
        """
        filepath = os.path.join(
            self.code_folder,
            "nicetoolbox",
            "detectors",
            "method_detectors",
            component_name,
            f"{detector_name}_inference.py",
        )
        try:
            exc.file_exists(filepath)
        except FileNotFoundError:
            logging.exception(f"Detector inference file {filepath} does not exist!")
            raise
        return filepath

    def get_output_folder(self, token: str) -> Path:
        """
        Get output folder by token.

        Args:
            token: One of 'output', 'main', 'csv'
        """
        if token == "output":
            return self.out_sub_folder
        if token == "main":
            return self.out_folder
        if token == "csv":
            os.makedirs(self.csv_folder, exist_ok=True)
            return self.csv_folder
        raise NotImplementedError(f"Unknown token '{token}'")

    def get_detector_output_folder(self, component: str, algorithm: str, token: str) -> Path:
        """
        Get detector-specific output folder.

        Args:
            component: Component name (e.g., 'body_joints')
            algorithm: Algorithm name (e.g., 'hrnetw48')
            token: Folder type - 'output', 'visualization', 'native_visualization', 'run_config', 'result'
        """
        path = self._subsequence_context.get_detector_folder(component, algorithm, token)
        os.makedirs(path, exist_ok=True)
        return path

    def get_algorithm_output_folder(self, algorithm: str, token: str) -> Path:
        """
        Get algorithm-centric output folder (e.g., under _native_outputs/<cur_algorithm_name>/).

        Args:
            algorithm: Algorithm instance name (e.g., 'motionbert')
            token: Folder type - 'detector_native_folder', 'output', 'native_visualization', 'run_config'
        """
        path = self._subsequence_context.get_detector_folder("", algorithm, token)
        os.makedirs(path, exist_ok=True)
        return path
