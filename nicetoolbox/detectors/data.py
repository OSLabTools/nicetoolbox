"""
Data module handling the data loading and processing of the give datasets.
"""

import glob
import logging
import os
from pathlib import Path

import numpy as np

from ..utils import check_and_exception as exc
from ..utils import system as oslab_sys
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

        # Collect all required file/folder paths
        # Note: In the future we expect io to return Path objects directly!
        self.code_folder = Path(config["code_folder"])
        self.input_folder = Path(io.get_data_folder())  # nicetoolbox_input (frames)
        self.source_folder = Path(io.get_input_folder())  # original folder (vids)
        # RENAME THESE TWO FUNCTIONS IN IO WTF

        # Collect data details from config (We only look at a single video here)
        # Note: In the future we expect config to be a pydantic model!
        self.dataset_name = config["dataset_name"]
        self.session_ID = config["session_ID"]
        self.sequence_ID = config["sequence_ID"]
        self.video_length = config["video_length"]
        self.video_start = config["video_start"]
        self.video_skip_frames = None  # Hardcoded - No access via config yet
        self.annotation_interval = 2.0
        self.subjects_descr = config["subjects_descr"]
        self.start_frame_index = config["start_frame_index"]
        self.camera_mapping = dict(
            (key, config[key]) for key in config if "cam_" in key
        )
        self.filename_template = config.get("filename_template", "{idx:09d}.png")

        # For BACKWARD COMPATIBILITY with detectors using frames_list
        self.frames_list = None
        self.frame_indices_list: None | list[int] = None

        # 1. Detect Input Type
        self.input_format: str = self._get_input_format()

        # 2. Validate fps given an example video file
        self.fps: int = self._get_fps_validated(config["fps"])

        # 3. Check and create input data if necessary
        self._input_data_creation()

        # 4. Load camera calibration if available
        self.calibration: dict | None = self._load_calibration()

        logging.info("DATA PREPARATION finished.\n\n")

    # TODO move to IO class
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

    # TODO move to IO class
    def get_venv_path(self, detector_name, env_name):
        """
        Get the file path of the virtual environment for the given detector and
        environment name.

        Args:
            detector_name (str): The name of the detector.
            env_name (str): The name of the environment.

        Returns:
            str: The file path of the virtual environment.

        Raises:
            FileNotFoundError: If the virtual environment does not exist.
        """
        os_type = oslab_sys.detect_os_type()
        if os_type == "linux":
            filepath = os.path.join(self.code_folder, "envs", env_name, "bin/activate")
        elif os_type == "windows":
            filepath = os.path.join(
                self.code_folder, "envs", env_name, "Scripts", "activate"
            )
        try:
            exc.file_exists(filepath)
        except FileNotFoundError:
            logging.exception(
                f"Virtual environment file {filepath} for detector = "
                f"'{detector_name}' does not exist!"
            )
            raise
        return filepath

    def get_input_recipe(self) -> dict:
        """
        Generates the Recipe config to be injected into the subprocess TOML.
        """
        if self.input_format in [".avi", ".mp4"]:
            # Extracted data lives in nicetoolbox_input
            root = self.input_folder
            # Structure: {cam}/frames/{idx:09d}.png
            template = "{camera}/frames/" + self.filename_template
        else:
            pass
            # TODO: implement source folder recipe generation
            # This requires knowing the original folder structure and
            # filename template from config, including session_ID, sequence_ID, etc.
            # cam_folder = root / self.session_ID / self.sequence_ID / cam / ?
            raise NotImplementedError(
                f"Input recipe generation for '{self.input_format}' not implemented."
            )

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

        example_input_folder = str(self.source_folder).replace(
            "<camera_name>", self.all_camera_names[0]
        )
        found_formats = [
            name in "_".join(sorted(os.listdir(example_input_folder)))
            for name in possible_formats
        ]
        if sum(found_formats) != 1:
            exc.error_log_and_raise(
                ValueError,
                "Reading input data",
                f"Multiple/no valid input format found in '{self.source_folder}'. "
                f"Found '{found_formats}', valid formats are ['mp4', 'avi'].",
            )
        return possible_formats[found_formats.index(True)]

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
            example_camera_name = self.all_camera_names[0]
            example_input_folder = str(self.source_folder).replace(
                "<camera_name>", example_camera_name
            )
            video_files = sorted(glob.glob(os.path.join(example_input_folder, "*")))
            fps = vid.get_fps(video_files[0])
            if fps != target_fps:
                logging.warning(
                    f"Detected fps = {fps} does not match fps given in the "
                    f"config = {target_fps}!"
                )
            return fps
        return target_fps

    def _input_data_creation(self) -> None:
        """
        Initializes the data required for running NICE toolbox.
        """
        if self.input_format in [".avi", ".mp4"]:
            if self._check_frames_exist():
                logging.info("Frames FOUND in nicetoolbox input folder")
                # === START BACKWARD COMPATIBILITY ===
                logging.info("LOADING frames_list for BACKWARD COMPATIBILITY...")
                data_list = self.get_inputs_list(
                    self.data_formats[0], self.all_camera_names
                )
                self.frames_list = []
                frames_list = []
                frame_indices_list = set()
                for filename in data_list:
                    if "frames" in filename:
                        frames_list.append(filename)
                        frame_indices_list.add(int(os.path.basename(filename)[:-4]))

                self.frame_indices_list = list(frame_indices_list)
                for camera_name in sorted(self.all_camera_names):
                    cam_frames = sorted(
                        [file for file in frames_list if camera_name in file]
                    )
                    self.frames_list.append(cam_frames)
                self.frames_list = [
                    frame.tolist() for frame in np.array(self.frames_list).T
                ]
                # === END BACKWARD COMPATIBILITY ===
            else:
                logging.info("EXTRACTING frames from video...")
                self._extract_frames_from_video()
            self.filename_template = "{idx:09d}.png"

        elif self.input_format in [".png", ".jpg", ".jpeg"]:
            if not self._check_frames_exist(is_source=True):
                raise FileNotFoundError(
                    f"Required source frames missing for dataset {self.dataset_name}."
                )
        else:
            raise NotImplementedError(
                f"Input format '{self.input_format}' not supported for data creation."
            )
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
        # todo check naming of input folders (see init...)
        root = self.source_folder if is_source else self.input_folder
        template = self.filename_template

        start_idx = self.video_start
        end_idx = self.video_start + self.video_length - 1

        for cam in self.all_camera_names:
            if is_source:
                raise NotImplementedError(
                    "Checking source frames is not implemented yet."
                )
                # TODO: implement source frame checking
                # This requires knowing the original folder structure and
                # filename template from config, including session_ID, sequence_ID, etc.
                # cam_folder = root / self.session_ID / self.sequence_ID / cam / ?
            else:  # noqa: RET506
                cam_folder = root / cam / "frames"

                start_name = template.format(idx=start_idx)  # Loop over entire range?
                end_name = template.format(idx=end_idx)

                start_path = cam_folder / start_name
                end_path = cam_folder / end_name

            # Check existence
            if not (start_path.exists() and end_path.exists()):
                logging.info(
                    f"No input frames found for camera '{cam}': "
                    f"Files will be created in '{cam_folder}'."
                )
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
            camera_name_indices = [
                name.lower() in video_file.lower()
                for name in list(self.all_camera_names)
            ]
            if not any(camera_name_indices):
                continue
            camera_name = list(self.all_camera_names)[camera_name_indices.index(True)]

            # split video into frames
            input_folder = os.path.join(self.input_folder, camera_name)
            os.makedirs(input_folder, exist_ok=True)

            if "frames" in self.data_formats:
                os.makedirs(os.path.join(input_folder, "frames"), exist_ok=True)
                frames_list, frame_indices_list = vid.split_into_frames(
                    video_file,
                    os.path.join(input_folder, "frames/"),
                    start_frame=self.start_frame_index,
                    keep_indices=True,
                )
                # We now have the frames list for the entire video
                # But the detectors only need the specified range
                # Below: BACKWARD COMPATIBILITY for detectors using frames_list
                if self.frame_indices_list is None:
                    slice_end = self.video_start + self.video_length

                    self.frame_indices_list = [
                        idx
                        for idx in frame_indices_list
                        if self.video_start <= idx < slice_end
                    ]
                    slice_frames = frames_list[self.video_start : slice_end]
                    self.frames_list = [[f] for f in slice_frames]
                else:
                    slice_end = self.video_start + self.video_length
                    slice_frames = frames_list[self.video_start : slice_end]

                    for n, f in enumerate(slice_frames):
                        if n < len(self.frames_list):
                            self.frames_list[n].append(f)

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

        if not calib_path or not os.path.exists(calib_path):
            logging.warning("Calibration file not found, skipping calibration.")
            return None

        # 2. Load calibration file
        calib_details = "__".join(
            [word for word in [self.session_ID, self.sequence_ID] if word]
        )
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
            calib = dict(
                (key, value)
                for key, value in loaded_calib.items()
                if key in self.all_camera_names
            )
        except Exception as err:
            logging.exception(
                f"An error occurred while creating calibration dictionary: {err}"
            )
            raise err

        return calib

    # BACKWARD COMPATIBILITY functions
    def get_inputs_list(self, data_format, camera_names):
        """
        Returns a list of input file paths based on the specified input format,
        data format, and camera names.
        """
        start = self.video_start
        end = self.video_start + self.video_length

        inputs_list = []
        if data_format == "snippets":
            raise NotImplementedError(
                "Data format 'snippets' not implemented in get_inputs_list."
            )
        if data_format == "segments":
            raise NotImplementedError(
                "Data format 'segments' not implemented in get_inputs_list."
            )
        if data_format == "frames":
            skip = 1 if not self.video_skip_frames else self.video_skip_frames
            file_names = [f"{x:05d}.png" for x in range(start, end, skip)]
            for camera_name in camera_names:
                inputs_list += [
                    os.path.join(self.input_folder, camera_name, "frames", n)
                    for n in file_names
                ]
        return inputs_list
