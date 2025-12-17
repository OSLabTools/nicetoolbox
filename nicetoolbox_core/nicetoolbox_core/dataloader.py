"""
Data loaders that generate input file paths based on a recipe pattern.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Type definitions for the different return formats

# 1. (frame_idx, {camera_name: file_path_str})
IMAGE_PATHS_BY_FRAMES_INDEX = Tuple[int, Dict[str, str]]

# 2. ([frame_indices], {camera_name: [file_path_strs]})
IMAGE_PATHS_BATCHED_BY_FRAMES_INDICES = Tuple[List[int], Dict[str, List[str]]]

# 3. (camera_name, [all_file_paths_for_sequence])
IMAGE_PATHS_BY_CAMERAS = Tuple[str, List[str]]


class InputDataLoader(ABC):
    """
    Abstract Base Class for loading input data based on a recipe.

    Handles configuration parsing, validation, and path generation logic.
    Subclasses must implement __iter__ to define how data is yielded.
    """

    def __init__(self, config: dict, expected_cameras: List[str]):
        """
        Initializes the data loader base.

        Args:
            config (dict): Configuration dictionary containing 'input_recipe'.
            expected_cameras (List[str]): Cameras the detector expects to process.

        Raises:
            ValueError: If expected cameras do not match available cameras in recipe.
        """
        # (1) Extract recipe from config
        recipe = config["input_recipe"]

        # (2) Root path and filename pattern for parsing files
        self.root_path = Path(recipe["root_path"])
        self.pattern = recipe["filename_template"]

        # (3) Range logic
        self.start = recipe["range_start"]
        self.end = recipe["range_end"]
        self.step = recipe.get("step", 1)

        # (4) Camera Validation
        # The recipe tells us what exists on disk
        available_cameras = recipe.get("camera_names", [])

        # Filter to only use the cameras the detector actually wants
        self.cameras = [cam for cam in expected_cameras if cam in available_cameras]

        if not self.cameras or len(self.cameras) != len(expected_cameras):
            logging.error(
                "Mismatch between expected cameras and available cameras in recipe."
                f" Expected: {expected_cameras}, Available: {available_cameras}"
            )
            raise ValueError("Camera mismatch between detector and input recipe.")

    def __len__(self) -> int:
        """Returns total number of frames to be processed per camera."""
        return len(range(self.start, self.end, self.step))

    def get_frames_range(self) -> Tuple[int, int]:
        """Returns the start and end frame indices."""
        return self.start, self.end

    def _generate_path(self, cam: str, idx: int) -> Optional[str]:
        """
        Generates and validates a single file path.
        Returns the path string if it exists, otherwise logs error and returns None.
        """
        rel_path = self.pattern.format(camera=cam, idx=idx)
        full_path = self.root_path / rel_path

        if not full_path.exists():
            logging.error(f"Expected input file does not exist: {full_path}")
            return None

        return str(full_path)

    @abstractmethod
    def __iter__(self):
        """Yields data according to the specific loader implementation."""
        pass


class ImagePathsByFrameIndexLoader(InputDataLoader):
    """
    Iterates over the sequence frame by frame.

    Yields:
        tuple: (frame_idx, {camera_name: file_path_str})

    Usage:
        Ideal for detectors that process one time-step across all views simultaneously
        (e.g., SPIGA, Multiview-XGaze).
    """

    def __iter__(self) -> Iterator[IMAGE_PATHS_BY_FRAMES_INDEX]:
        """Yields (frame_idx, {camera_name: file_path_str})."""
        for idx in range(self.start, self.end, self.step):
            frame_files = {}
            for cam in self.cameras:
                path = self._generate_path(cam, idx)
                if path:
                    frame_files[cam] = path

            yield (idx, frame_files)


class ImagePathsBatchByFrameIndexLoader(InputDataLoader):
    """
    Iterates over the sequence in batches of frames.

    Yields:
        tuple: ([frame_indices], {camera_name: [file_path_strs]})

    Usage:
        Ideal for detectors that support batch processing for efficiency
        (e.g., Py-Feat).
    """

    def __init__(self, config: dict, expected_cameras: List[str], batch_size: int):
        """
        Initializes the batch data loader.

        Args:
            config (dict): Configuration dictionary containing 'input_recipe'.
            expected_cameras (List[str]): Cameras the detector expects to process.
            batch_size (int): Number of frames to include in each batch.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(config, expected_cameras)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"Invalid batch_size: {batch_size}. Must be positive integer."
            )
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[IMAGE_PATHS_BATCHED_BY_FRAMES_INDICES]:
        """Yields ([frame_indices], {camera_name: [file_path_strs]})."""
        batch_indices = range(self.start, self.end, self.step)

        for i in range(0, len(batch_indices), self.batch_size):
            current_indices = list(batch_indices[i : i + self.batch_size])

            batch_files: Dict[str, List[str]] = {cam: [] for cam in self.cameras}

            for idx in current_indices:
                for cam in self.cameras:
                    path = self._generate_path(cam, idx)
                    if path:
                        batch_files[cam].append(path)

            yield (current_indices, batch_files)


class ImagePathsByCameraLoader(InputDataLoader):
    """
    Iterates camera by camera, yielding the full sequence for one camera at a time.

    Yields:
        tuple: (camera_name, [all_file_paths_for_sequence])

    Usage:
        Ideal for detectors that process a full video/view as a single unit or
        handle their own internal batching (e.g., MMPose).
    """

    def __iter__(self) -> Iterator[IMAGE_PATHS_BY_CAMERAS]:
        """Yields (camera_name, [all_file_paths_for_sequence])."""
        for cam in self.cameras:
            video_files = []
            for idx in range(self.start, self.end, self.step):
                path = self._generate_path(cam, idx)
                if path:
                    video_files.append(path)

            yield (cam, video_files)


class MP4VideoPathsByCameraLoader(InputDataLoader):
    """
    Iterates camera by camera, yielding the full video file for one camera at a time.

    Yields:
        tuple: (camera_name, video_file_path)

    Usage:
        Ideal for detectors that process video files directly.
    """

    pass
