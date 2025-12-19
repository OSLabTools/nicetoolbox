"""
Input and Output management for the evaluation module.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..configs.config_loader import ConfigLoader
from ..configs.schemas.dataset_properties import DatasetConfig
from ..configs.schemas.detectors_run_file import DetectorsRunIO
from ..configs.schemas.evaluation_config import EvaluationIO


class IO:
    """
    Manages folder structure and reading/writing NPZ results.
    """

    # initially available
    _cfg_loader: ConfigLoader
    output_folder: Path
    eval_visualization_folder: Path
    experiment_folder: Path
    _experiment_folder_placeholders: Path

    # when dataset set
    dataset_name: str
    experiment_results_folder: str
    path_to_annotations: Path
    path_to_calibrations: Path

    def __init__(
        self,
        io_config: EvaluationIO,
        experiment_io: DetectorsRunIO,
        cfg_loader: ConfigLoader,
    ):
        """
        Initialize the IO manager with the provided IO configuration.

        Args:
            io_config (IOConfig): Configuration object containing paths for
                experiment, output, and evaluation visualization folders.
        """
        self._cfg_loader = cfg_loader

        self.output_folder = io_config.output_folder
        self.eval_visualization_folder = io_config.eval_visualization_folder

        self.eval_visualization_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.experiment_folder = io_config.experiment_folder
        self._experiment_folder_placeholders = (
            experiment_io.detector_final_result_folder
        )  # noqa: E501

    # TODO: IO mutates itself, when we call init_dataset
    # it should be immutable and only return mutated data
    def init_dataset(self, dataset_properties: DatasetConfig):
        """
        Dataset-specific IO initialization inside the main dataset loop.

        Args:
            dataset_properties (DatasetProperties): Properties for the specific dataset.
        """
        self.dataset_name = dataset_properties._dataset_name
        self.path_to_annotations = dataset_properties.path_to_annotations
        self.path_to_calibrations = dataset_properties.path_to_calibrations

    def get_detector_results_file(self, video_config, component, algorithm):
        """
        Returns the path to the detector results file based on the video configuration,
        component, and algorithm.

        Args:
            video_config (RunConfigVideo): Configuration for the video run.
            component (str): Name of the component.
            algorithm (str): Name of the algorithm.
        Returns:
            Path: The path to the detector results file.
        """
        # resolve folder path in case there are placeholders
        runtime_ctx = {
            "cur_dataset_name": self.dataset_name,
            "cur_component_name": component,
            "cur_algorithm_name": algorithm,
            "cur_session_ID": video_config.session_ID,
            "cur_sequence_ID": video_config.sequence_ID,
            "cur_video_start": video_config.video_start,
            "cur_video_length": video_config.video_length,
        }
        resolved_folder_path = self._cfg_loader.resolve(
            str(self._experiment_folder_placeholders), runtime_ctx
        )
        return Path(resolved_folder_path) / f"{algorithm}.npz"

    def get_out_folder(self, selection="", component=""):
        """
        Returns the output folder path based on the provided parameters.

        Args:
            selection (str): Selection criteria for the output.
            component (str): Component name.

        Returns:
            Path: The output folder path.
        """
        if not (selection and component):
            return self.output_folder

        out = self.output_folder / f"{self.dataset_name}__{selection}" / component
        out.mkdir(parents=True, exist_ok=True)
        return out

    def get_csv_folder(self):
        """
        Returns the path to the CSV folder within the output directory.
        """
        csv_folder = self.get_out_folder() / "csv_files"
        csv_folder.mkdir(parents=True, exist_ok=True)
        return csv_folder

    def save_npz(self, path: Path, **arrays):
        """Save arrays to a compressed NPZ file."""
        np.savez_compressed(path, **arrays)
        logging.info(f"Saved results to {path}")

    def save_summaries_to_csv(self, path: Path, data: Dict[str, Any]):
        """Save a summary dict to a CSV file."""
        pd.DataFrame(data).to_csv(path, index=False)
        logging.info(f"Saved results to {path}")
