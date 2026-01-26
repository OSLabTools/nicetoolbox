"""
Data module handling the data loading and processing of the give datasets.
"""

import glob
import json
import logging
import os
from pathlib import Path

import numpy as np

from ..utils import check_and_exception as exc
from ..utils import video as vid
from .in_out import IO


class Data:
    """
    A data class for NICE toolbox input data validation and creation.
    """

    def __init__(
        self,
        config: dict,  # TODO: pydantic model?
        io: IO,
        data_formats: list[str],
        all_camera_names: list[str],
        all_dataset_names: list[str],
    ) -> None:
        """
        Initialize the Data class.

        Args:
            config (dict): The configuration dictionary.
            io (IO): The IO object for file and folder operations.
            data_formats (list): The list of data formats required.
            all_camera_names (list): The list of all camera names.

        Returns:
            None
        """
        logging.info("Start DATA PREPARATION.")

        self.io: IO = io
        self.data_formats = data_formats  # currently ['frames'] for all detectors
        self.all_camera_names = all_camera_names
        self.all_dataset_names = all_dataset_names

        # --- Path Definitions ---
        self.input_folder = io.get_nice_input_folder()  # nicetoolbox_input (frames)
        self.source_folder = io.get_data_source_folder()  # original folder (vids)

        # --- Config Parameters ---
        self.dataset_name = config["dataset_name"]
        self.session_ID = config["session_ID"]
        self.sequence_ID = config["sequence_ID"]

        self.start_frame_index: int = config["start_frame_index"]
        self.video_start = config["video_start"]
        self.video_length_config = config["video_length"]  # Can be -1 for full length

        self.video_skip_frames = None  # Hardcoded - No access via config yet
        self.annotation_interval = 2.0  # Keep? Hardcoded - No access via config yet
        self.subjects_descr = config["subjects_descr"]
        self.camera_mapping = dict((key, config[key]) for key in config if "cam_" in key)

        self.filename_template = config.get("filename_template", "{idx:09d}.png")

        # --- Data Preparation Steps ---

        # 1. Detect input format and search for an example video file
        self.input_format: str = self._get_input_format()
        self.video_sample_path: Path = self._get_example_video_path()

        # 2. Validate fps and video length given an example video file
        self.fps: int = self._get_fps_validated(config["fps"])
        self.video_length = self._resolve_video_length()

        # 3. Check and create input data if necessary
        self._input_data_creation()

        # 4. Load camera calibration if available
        self.calibration: dict | None = self._load_calibration()

        logging.info("DATA PREPARATION finished.\n\n")

    def get_input_recipe(self) -> dict:
        """
        Generates the Recipe config to be injected into the subprocess TOML.
        """
        if self.input_format in [".avi", ".mp4"]:
            root = self.input_folder  # Extracted data lives in nicetoolbox_input
            template = "{camera}/frames/" + self.filename_template  # Structure: {cam}/frames/{idx:09d}.png
        else:
            raise NotImplementedError(f"Input recipe generation for '{self.input_format}' not implemented.")

        # TODO: Pydantic model for recipe?
        return {
            "input_recipe": {
                "type": "video_sequence",  # For future extensions like frame datasets?
                "root_path": str(root),
                "camera_names": sorted(list(self.all_camera_names)),
                "filename_template": template,
                "range_start": self.video_start,
                "range_end": self.video_start + self.video_length,
                "step": 1 if self.video_skip_frames is None else self.video_skip_frames,
            }
        }

    def _get_input_format(self) -> str:
        """
        Get the input format for the given camera names.

        Returns:
            str: The input format for the given camera names.

        Raises:
            ValueError: If multiple or no valid input format is found in the data
                input folder.
        """
        possible_formats = [".mp4", ".avi", ".png", ".jpg", ".jpeg"]

        example_input_folder = str(self.source_folder).replace("<camera_name>", self.all_camera_names[0])
        found_formats = [name in "_".join(sorted(os.listdir(example_input_folder))) for name in possible_formats]
        if sum(found_formats) != 1:
            exc.error_log_and_raise(
                ValueError,
                "Reading input data",
                f"Multiple/no valid input format found in '{self.source_folder}'. "
                f"Found '{found_formats}', valid formats are ['mp4', 'avi'].",
            )
        return possible_formats[found_formats.index(True)]

    def _get_example_video_path(self) -> Path:
        """
        Finds a video file for metadata extraction Reused by FPS check and Length resolution.

        Returns:
            Path: The path to an example video file.
        """
        cam_0 = self.all_camera_names[0]
        src_pattern = str(self.source_folder).replace("<camera_name>", cam_0)
        search_path = Path(src_pattern)

        files = sorted(search_path.glob(f"*{self.input_format}"))

        if not files:
            raise FileNotFoundError(
                f"No video files ({self.input_format}) found in {search_path} " f"for camera {cam_0}."
            )
        return files[0]

    def _get_fps_validated(self, target_fps: int) -> int:
        """
        Validates the frames per second (fps) of the input video files against the
        target fps specified in the configuration.

        Args:
            target_fps (int): The desired fps specified in the configuration.

        Returns:
            int: The fps of the input video files. If the input formats are not 'mp4'
                or 'avi', the target_fps value specified in the config is returned.
        """
        if self.input_format in [".mp4", ".avi"]:
            fps = vid.get_fps(str(self.video_sample_path))
            if fps != target_fps:
                logging.warning(f"Detected fps = {fps} does not match fps given in the " f"config = {target_fps}!")
            return fps

        raise NotImplementedError(f"FPS validation for input format '{self.input_format}' is not implemented.")

    def _resolve_video_length(self) -> int:
        """
        Resolves the video length in frames. If the video_length is specified in the
        configuration, it is returned directly. If it is (-1), the length is determined
        from the example video file.

        Returns:
            int: The resolved video length in frames.
        """
        if self.video_length_config > 0:
            return self.video_length_config

        # If length is -1, we need to figure out the length from the video file

        if self.input_format in [".mp4", ".avi"]:
            total_frames = vid.get_number_of_frames(str(self.video_sample_path))
            available_length = total_frames - self.video_start

            if available_length <= 0:
                raise ValueError(
                    f"video_start ({self.video_start}) is beyond the end of the video "
                    f"({total_frames} frames) in {self.video_sample_path.name}"
                )

            logging.info(
                f"Auto-detected length: {available_length} frames "
                f"(Total: {total_frames}, Start: {self.video_start})"
            )
            return available_length

        raise NotImplementedError(f"Video length resolution for '{self.input_format}' not implemented.")

    def _input_data_creation(self) -> None:
        """
        Initializes the data required for running NICE toolbox.
        """
        if self.input_format in [".avi", ".mp4"]:
            if self._check_frames_exist():
                logging.info("Frames FOUND in nicetoolbox input folder")
            else:
                logging.info("EXTRACTING frames from video...")
                self._extract_frames_from_video()
            self.filename_template = "{idx:09d}.png"

        elif self.input_format in [".png", ".jpg", ".jpeg"]:
            if not self._check_frames_exist(is_source=True):
                raise FileNotFoundError(f"Required source frames missing for dataset {self.dataset_name}.")
        else:
            raise NotImplementedError(f"Input format '{self.input_format}' not supported for data creation.")
        logging.info("DATA CREATION completed.")

    def _check_frames_exist(self, is_source: bool = False) -> bool:
        """
        Check if frames exist in the nicetoolbox input folder ("Source of truth").

        Args:
            is_source (bool): Whether to check in the source input folder or the
                nicetoolbox input folder. Defaults to False.

        Returns:
            bool: True if frames exist for all cameras, False otherwise.
        """
        root = self.source_folder if is_source else self.input_folder
        template = self.filename_template

        start_idx = self.video_start
        end_idx = self.video_start + self.video_length - 1

        for cam in self.all_camera_names:
            if is_source:
                raise NotImplementedError("Checking source frames is not implemented yet.")
                # TODO: implement source frame checking
                # This requires knowing the original folder structure and
                # filename template from config, including session_ID, sequence_ID, etc.
                # cam_folder = root / self.session_ID / self.sequence_ID / cam / ?

            cam_folder = root / cam / "frames"

            start_name = template.format(idx=start_idx)
            end_name = template.format(idx=end_idx)

            start_path = cam_folder / start_name
            end_path = cam_folder / end_name

            # Check existence
            if not (start_path.exists() and end_path.exists()):
                logging.info(f"No input frames found for camera '{cam}': " f"Files will be created in '{cam_folder}'.")
                return False

        return True

    def _extract_frames_from_video(self):
        """
        Create input frames from video files inside the nicetoolbox_input folder.

        This method detects video input files, splits them into frames, and organizes
        the frames into different data formats which are frames, segments, and snippets.

        Raises:
            AssertionError: If the length of the frame indices list does not match
                the specified video length.
            AssertionError: If the frame indices of different cameras do not match.
        """
        # detect all video input files
        # Build glob pattern for all camera input folders and list video files
        data_input_pattern = str(self.source_folder).replace("<camera_name>", "*")
        pattern = Path(data_input_pattern) / "*"
        video_files = sorted(glob.glob(str(pattern)))

        for video_file in video_files:
            camera_name_indices = [name.lower() in video_file.lower() for name in list(self.all_camera_names)]
            if not any(camera_name_indices):
                continue
            camera_name = list(self.all_camera_names)[camera_name_indices.index(True)]

            logging.info("Extracting Video Specifications...")
            raw_video_info = vid.probe_video(video_file)
            video_info_path = os.path.join(self.input_folder, camera_name + "_meta.json")
            with open(video_info_path, "w") as f:
                json.dump(raw_video_info, f, indent=4)

            logging.info("Parsing Video Specifications...")
            video_info = vid.json_to_video_info(raw_video_info)

            # split video into frames
            input_folder = os.path.join(self.input_folder, camera_name)
            os.makedirs(input_folder, exist_ok=True)

            if "frames" in self.data_formats:
                os.makedirs(os.path.join(input_folder, "frames"), exist_ok=True)
                vid.split_into_frames(
                    video_file,
                    os.path.join(input_folder, "frames/"),
                    video_info.frames,
                    start_frame=self.start_frame_index,
                    keep_indices=True,
                )

    def _load_calibration(self) -> dict | None:
        """
        Load camera calibration from a file for a specific dataset.

        Currently implemented for the datasets 'dyadic_communication' and
        'mpi_inf_3dhp'.

        Args:
            calibration_file (str): The path to the calibration file.
            dataset_name (str): The name of the dataset.

        Returns:
            dict: A dictionary containing the loaded camera calibration.

        Raises:
            NotImplementedError: If loading camera calibration for the specified
            dataset is not implemented.
        """
        # 1. Get calibration file path
        calib_path = self.io.get_calibration_file()
        if not calib_path or not os.path.isfile(calib_path):
            logging.warning("Calibration file not found, skipping calibration.")
            return None

        # 2. Load calibration file
        calib_details = "__".join([word for word in [self.session_ID, self.sequence_ID] if word])
        try:
            loaded_calib = np.load(calib_path, allow_pickle=True)[calib_details].item()
        except KeyError as err:
            logging.exception(
                f"Calibration for session '{self.session_ID}' and sequence "
                f"'{self.sequence_ID}' not found for calibration file at "
                f"'{calib_path}'."
            )
            raise err
        try:
            calib = dict((key, value) for key, value in loaded_calib.items() if key in self.all_camera_names)
        except Exception as err:
            logging.exception(f"An error occurred while creating calibration dictionary: {err}")
            raise err

        return calib
