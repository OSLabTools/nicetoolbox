"""
Video/frame data handler for the NICE Toolbox.

Handles frame extraction from video files and preparation of image sequences.
Also owns camera calibration loading since calibration is video-specific.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from nicetoolbox_core.input_recipes import VideoInputRecipe

from ...configs.models.video_timestamp import timestamp_to_frame_index
from ...utils import video as vid
from ...utils.filehandling import resolve_single_file
from ...utils.logging_utils import log_with_underscore
from ..in_out import SequenceIO
from ..subsequence_context import SubsequenceContext
from .handler import BaseModalityHandler

FRAME_LABEL_TEMPLATE = "{idx:09d}"
FILENAME_TEMPLATE = f"{FRAME_LABEL_TEMPLATE}.png"


class VideoDataHandler(BaseModalityHandler):
    """
    Handles video/frame data preparation.

    Responsibilities:
    - Resolve each camera's configured video path (single-file, glob-single-match)
    - Validate that all cameras share the same FPS and frame count
    - Extract frames from video files (mp4, mov)
    - Validate existing frame sequences
    - Generate input recipes for frame loaders
    - Load camera calibration data
    """

    def __init__(self, io: SequenceIO, subsequence_context: SubsequenceContext):
        # Shared fields
        super().__init__(io, subsequence_context)

        # Resolved during prepare()
        self.camera_video_paths: Optional[Dict[str, Path]] = None
        self.calibration: Optional[Dict[str, Any]] = None

    @property
    def modality_name(self) -> str:
        return "video"

    def prepare(self) -> None:
        log_with_underscore("Preparing Video Modality...")

        if not self.all_camera_names:
            raise ValueError("No camera names provided.")

        # Resolve one video file per camera from its configured path.
        self.camera_video_paths = self._resolve_video_paths()

        # Probe all videos, check cross-camera consistency, then validate against config
        self.fps, self.length_frames = self._resolve_fps_and_length()
        self.start_frame = timestamp_to_frame_index(self.subsequence_context.video_start, self.fps)

        # Save all frames str names for data serialization
        self.frame_labels = self._frame_labels(self.start_frame, self.length_frames)

        # Check and create input data if necessary
        self._input_data_creation()

        # Load camera calibration if available
        self.calibration = self._load_calibration()

        self._available = True
        logging.info("Video DATA CREATION completed.")

    def get_recipe(self) -> VideoInputRecipe:
        """
        Generates the Recipe config to be injected into the subprocess TOML.
        """
        return VideoInputRecipe(
            root_path=str(self.nice_input_folder),
            camera_names=sorted(list(self.all_camera_names)),
            filename_template="{camera}/frames/" + FILENAME_TEMPLATE,
            range_start=self.start_frame,
            range_end=self.start_frame + self.length_frames,
            step=1,
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _frame_labels(self, start_frame, length_frames) -> list[str]:
        """Generate a list of all frame names."""
        start = start_frame
        end = start + length_frames
        return [FRAME_LABEL_TEMPLATE.format(idx=idx) for idx in range(start, end)]

    def _resolve_video_paths(self) -> Dict[str, Path]:
        """
        Resolve each active camera's configured path to exactly one video file.

        Paths may contain a `*` wildcard; zero or multiple matches raise.
        """
        result: Dict[str, Path] = {}
        cameras = self.sequence_properties.video.cameras
        for cam in self.all_camera_names:
            track = cameras[cam]
            result[cam] = resolve_single_file(Path(track.path), label=f"Video track '{cam}'")
        return result

    def _resolve_fps_and_length(self) -> tuple[int, int]:
        """
        Probe all camera videos, assert cross-camera consistency of FPS and
        frame count, then validate FPS against config.

        Returns:
            (fps, length_frames) — fps detected from videos, length in frames
            from config or auto-detected.

        Raises:
            ValueError: If cameras disagree on FPS or frame count, or if
                video_start is beyond the end of the video.
        """
        infos = {}
        for cam, path in self.camera_video_paths.items():
            raw = vid.probe_video(str(path))
            infos[cam] = vid.json_to_video_info(raw)

        # Cross-camera consistency: every probed camera must agree on fps and frame count.
        fps_values = {cam: int(info.fps) for cam, info in infos.items() if info.fps is not None}
        frame_values = {cam: info.frames for cam, info in infos.items() if info.frames is not None}

        if len(fps_values) != len(infos):
            missing = sorted(set(infos) - set(fps_values))
            raise ValueError(f"Could not determine FPS from cameras: {missing}.")
        if len(set(fps_values.values())) > 1:
            raise ValueError(f"Cameras have inconsistent FPS: {fps_values}")
        if len(set(frame_values.values())) > 1:
            raise ValueError(f"Cameras have inconsistent frame counts: {frame_values}")

        fps = next(iter(fps_values.values()))
        logging.info(f"Auto-detected FPS: {fps}")

        # Resolve length
        video_length_frame = timestamp_to_frame_index(self.subsequence_context.video_length, fps)
        if video_length_frame > 0:
            return fps, video_length_frame

        # Auto-detect from frame count
        total_frames = next(iter(frame_values.values())) if frame_values else None
        if total_frames is None:
            raise ValueError("Could not determine frame count from any camera video.")

        start_frame = timestamp_to_frame_index(self.subsequence_context.video_start, fps)
        available = total_frames - start_frame
        if available <= 0:
            raise ValueError(f"video_start ({start_frame}) is beyond the end of the video " f"({total_frames} frames).")

        logging.info(f"Auto-detected length: {available} frames " f"(Total: {total_frames}, Start: {start_frame})")
        return fps, available

    def _input_data_creation(self) -> None:
        """
        Initializes the data required for running NICE toolbox.
        """
        if self._check_frames_exist():
            logging.info("Frames FOUND in nicetoolbox input folder")
        else:
            logging.info("EXTRACTING frames from video...")
            self._extract_frames_from_video()

    def _check_frames_exist(self) -> bool:
        """
        Check if frames exist in the nicetoolbox input folder ("Source of truth").

        Returns:
            bool: True if frames exist for all cameras, False otherwise.
        """
        start_idx = self.start_frame
        end_idx = self.start_frame + self.length_frames - 1

        for cam in self.all_camera_names:
            cam_folder = self.nice_input_folder / cam / "frames"

            start_name = FILENAME_TEMPLATE.format(idx=start_idx)
            end_name = FILENAME_TEMPLATE.format(idx=end_idx)

            if not ((cam_folder / start_name).exists() and (cam_folder / end_name).exists()):
                logging.info(f"No input frames found for camera '{cam}': " f"Files will be created in '{cam_folder}'.")
                return False

        return True

    def _extract_frames_from_video(self) -> None:
        """
        Extract frames from each camera's resolved video file into the
        nicetoolbox_input folder.
        """
        for cam, video_path in self.camera_video_paths.items():
            logging.info(f"Extracting frames for camera '{cam}' from '{video_path}'...")

            raw_video_info = vid.probe_video(str(video_path))
            video_info_path = self.nice_input_folder / f"{cam}_meta.json"
            with open(video_info_path, "w") as f:
                json.dump(raw_video_info, f, indent=4)

            video_info = vid.json_to_video_info(raw_video_info)

            cam_folder = self.nice_input_folder / cam
            frames_folder = cam_folder / "frames"
            frames_folder.mkdir(parents=True, exist_ok=True)

            vid.split_into_frames(
                str(video_path),
                str(frames_folder) + "/",
                video_info.frames,
                keep_indices=True,
            )

    def _load_calibration(self) -> dict | None:
        """
        Load camera calibration from a file for a specific dataset.

        Returns:
            dict: A dictionary containing the loaded camera calibration.

        Raises:
            KeyError: If loading camera calibration for the specified
            dataset is not implemented.
        """
        calib_path = self.io.get_calibration_file()
        if not calib_path or not os.path.isfile(calib_path):
            logging.warning("Calibration file not found, skipping calibration.")
            return None

        calib_details = self.sequence_id
        try:
            loaded_calib = np.load(calib_path, allow_pickle=True)[calib_details].item()
        except KeyError as err:
            logging.exception(
                f"Calibration for sequence '{self.sequence_id}' not found for calibration file at '{calib_path}'."
            )
            raise err
        try:
            calib = {key: value for key, value in loaded_calib.items() if key in self.all_camera_names}
        except Exception as err:
            logging.exception(f"An error occurred while creating calibration dictionary: {err}")
            raise err

        missing = set(self.all_camera_names) - set(calib.keys())
        if missing:
            raise KeyError(
                f"Calibration file '{calib_path}' (sequence '{self.sequence_id}') is missing "
                f"entries for cameras {sorted(missing)}. Available calibration keys: "
                f"{sorted(loaded_calib.keys())}. Configured camera names: "
                f"{sorted(self.all_camera_names)}. Rename the keys in the calibration file "
                f"(or the camera names in the dataset config) so they match."
            )

        return calib
