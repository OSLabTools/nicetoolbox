"""
Video/frame data handler for the NICE Toolbox.

Handles frame extraction from video files and preparation of image sequences.
Also owns camera calibration loading since calibration is video-specific.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from pydantic import ValidationError

from nicetoolbox_core.input_recipes import VideoInputRecipe

from ...configs.models.video_timestamp import timestamp_to_frame_index
from ...configs.schemas.detectors_run_file import (
    FRAME_EXTRACTION_FILENAME,
    FrameExtractionConfig,
    load_frame_extraction,
    save_frame_extraction,
)
from ...utils import video as vid
from ...utils.filehandling import resolve_single_file
from ...utils.logging_utils import log_with_underscore
from ..in_out import SequenceIO
from ..subsequence_context import SubsequenceContext
from .handler import BaseModalityHandler

FRAME_LABEL_TEMPLATE = "{idx:09d}"


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
        super().__init__(io, subsequence_context)

        # Resolved during prepare()
        self.camera_video_paths: Optional[Dict[str, Path]] = None
        self.frames_folders: Optional[Dict[str, Path]] = None
        self.calibration: Optional[Dict[str, Any]] = None

        self.frame_extraction: FrameExtractionConfig = subsequence_context.io.frame_extraction
        self.filename_template: str = FRAME_LABEL_TEMPLATE + self.frame_extraction.frame_extension

    @property
    def modality_name(self) -> str:
        return "video"

    def prepare(self) -> None:
        log_with_underscore("Preparing Video Modality...")

        if not self.all_camera_names:
            raise ValueError("No camera names provided.")

        # Resolve one video file per camera
        self.camera_video_paths = self._resolve_video_paths()
        # Resolve one frames folder per camera where to extract frames
        self.frames_folders = self._resolve_frames_folders()

        # Probe each camera video once and expose per-camera resolution.
        infos = self._resolve_video_meta_info()
        self.resolutions = {cam: [info.width, info.height] for cam, info in infos.items()}

        # Check cross-camera consistency and validate against config
        self.fps, self.length_frames = self._resolve_fps_and_length(infos)
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
        # need to convert keys to str for serialization
        frames_folders = {cam: str(folder) for cam, folder in self.frames_folders.items()}

        return VideoInputRecipe(
            frames_folders=frames_folders,
            filename_template=self.filename_template,
            range_start=self.start_frame,
            range_end=self.start_frame + self.length_frames,
            step=1,
            fps=self.fps,
            resolutions=self.resolutions,
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

    def _resolve_frames_folders(self) -> Dict[str, Path]:
        """
        Resolve the folder holding extracted frames for each active camera.

        Cameras are sorted to keep the same camera order in the input recipe.
        """
        result: Dict[str, Path] = {}
        for cam in sorted(self.all_camera_names):
            result[cam] = self.io.camera_input_folders[cam] / "frames"
        return result

    def _resolve_video_meta_info(self) -> Dict[str, vid.VideoInfo]:
        """
        Probe each resolved camera video once via ffprobe.
        """
        infos: Dict[str, vid.VideoInfo] = {}
        for cam, path in self.camera_video_paths.items():
            raw = vid.probe_video(str(path))
            infos[cam] = vid.json_to_video_info(raw)
        return infos

    def _resolve_fps_and_length(self, infos: Dict[str, vid.VideoInfo]) -> tuple[int, int]:
        """
        Assert cross-camera consistency of FPS and frame count on the probed
        video infos, then validate FPS against config.

        Returns:
            (fps, length_frames) — fps detected from videos, length in frames
            from config or auto-detected.

        Raises:
            ValueError: If cameras disagree on FPS or frame count, or if
                video_start is beyond the end of the video.
        """
        # TODO: warn user that we are rounding fps. Or better, support float fps.
        # Cross-camera consistency: every probed camera must agree on fps and frame count.
        fps_values = {cam: round(info.fps) for cam, info in infos.items() if info.fps is not None}
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
        # Every camera must report the same number
        if len(frame_values) != len(infos):
            # One or more cameras missing frame count meta
            # That means we can't reliably know the end of the video
            # TODO: we can count frames after extraction and see if they agree
            missing = sorted(set(infos) - set(frame_values))
            raise ValueError(
                f"Could not determine frame count from cameras: {missing}. The container does not "
                "store one (common for webm/mkv and streamed mp4). Set an explicit video_length "
                "for this sequence in detector_run_config, or re-encode this video."
            )

        total_frames = next(iter(frame_values.values()))
        start_frame = timestamp_to_frame_index(self.subsequence_context.video_start, fps)
        available = total_frames - start_frame
        if available <= 0:
            raise ValueError(f"video_start ({start_frame}) is beyond the end of the video " f"({total_frames} frames).")

        logging.info(f"Auto-detected length: {available} frames " f"(Total: {total_frames}, Start: {start_frame})")
        return fps, available

    def _input_data_creation(self) -> None:
        """
        Initializes the data required for running NICE toolbox.
        Extracts frames only for cameras whose cached frames are missing or outdated.
        """
        stale_cameras = [cam for cam in self.all_camera_names if not self._frames_valid(cam)]
        if not stale_cameras:
            logging.info("Frames FOUND in nicetoolbox input folder")
            return

        logging.info(f"EXTRACTING frames from video for cameras {stale_cameras}...")
        self._extract_frames_from_video(stale_cameras)

    def _frames_valid(self, cam: str) -> bool:
        """
        Check if the cached frames of a camera ("Source of truth") cover the subsequence
        and were extracted with the configured frame extraction settings.
        """
        cam_folder = self.io.camera_input_folders[cam]
        frames_folder = self.frames_folders[cam]
        start_name = self.filename_template.format(idx=self.start_frame)
        end_name = self.filename_template.format(idx=self.start_frame + self.length_frames - 1)
        if not ((frames_folder / start_name).exists() and (frames_folder / end_name).exists()):
            logging.info(f"No input frames found for camera '{cam}': Files will be created in '{frames_folder}'.")
            return False

        try:
            cached = load_frame_extraction(cam_folder)
        except (OSError, ValidationError) as e:
            logging.warning(f"Unknown frame extraction settings for camera '{cam}' ({e}). Re-extracting frames.")
            return False

        if cached != self.frame_extraction:
            logging.warning(
                f"Frame extraction settings for camera '{cam}' differ from detectors_run_file.toml:\n"
                f"  cached: {cached!r}\n"
                f"  config: {self.frame_extraction!r}"
            )
            return False

        return True

    def _extract_frames_from_video(self, cameras: list[str]) -> None:
        """
        Extract frames from the resolved video files of the given cameras into their
        camera input folders, replacing any outdated frames.
        """
        for cam in cameras:
            video_path = self.camera_video_paths[cam]
            logging.info(f"Extracting frames for camera '{cam}' from '{video_path}'...")

            cam_folder = self.io.camera_input_folders[cam]
            frames_folder = self.frames_folders[cam]

            # Outdated frames may use another extension, so they would not be overwritten.
            # Settings are removed first, so an interrupted extraction is never trusted.
            (cam_folder / FRAME_EXTRACTION_FILENAME).unlink(missing_ok=True)
            shutil.rmtree(frames_folder, ignore_errors=True)
            frames_folder.mkdir(parents=True)

            raw_video_info = vid.probe_video(str(video_path))
            video_info_path = cam_folder / "meta.json"
            with open(video_info_path, "w") as f:
                json.dump(raw_video_info, f, indent=4)

            video_info = vid.json_to_video_info(raw_video_info)

            vid.split_into_frames(
                str(video_path),
                str(frames_folder) + "/",
                video_info.frames,
                frame_extraction=self.frame_extraction,
                keep_indices=True,
            )

            save_frame_extraction(self.frame_extraction, cam_folder)

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
