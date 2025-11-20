"""
Input and Output management for the evaluation module.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..configs.schemas.dataset_properties import DatasetConfig
from ..configs.schemas.evaluation_config import EvaluationIO


class IO:
    """
    Manages folder structure and reading/writing NPZ results.
    """

    def __init__(self, io_config: EvaluationIO):
        """
        Initialize the IO manager with the provided IO configuration.

        Args:
            io_config (IOConfig): Configuration object containing paths for
                experiment, output, and evaluation visualization folders.
        """
        self.io_config = io_config
        self.experiment_io = io_config._experiment_io

        self.output_folder = io_config.output_folder
        self.eval_visualization_folder = io_config.eval_visualization_folder

        self.eval_visualization_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.experiment_folder = io_config.experiment_folder
        self.experiment_folder_placeholders = io_config._experiment_io[
            "detector_final_result_folder"
        ]

    def init_dataset(self, dataset_properties: DatasetConfig):
        """
        Dataset-specific IO initialization inside the main dataset loop.

        Args:
            dataset_properties (DatasetProperties): Properties for the specific dataset.
        """
        # Experiment details:
        self.dataset_name = dataset_properties._dataset_name
        self.experiment_results_folder = self.experiment_folder_placeholders.replace(
            "<dataset_name>", self.dataset_name
        )
        # Dataset Properties
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
        folder_path = copy.deepcopy(self.experiment_results_folder)
        folder_path = folder_path.replace("<component_name>", component)
        folder_path = folder_path.replace("<algorithm_name>", algorithm)
        folder_path = folder_path.replace("<session_ID>", video_config.session_ID)
        folder_path = folder_path.replace("<sequence_ID>", video_config.sequence_ID)
        folder_path = folder_path.replace(
            "<video_start>", str(video_config.video_start)
        )
        folder_path = folder_path.replace(
            "<video_length>", str(video_config.video_length)
        )
        return Path(folder_path) / f"{algorithm}.npz"

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
