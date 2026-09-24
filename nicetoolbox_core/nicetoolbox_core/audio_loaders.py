"""
Audio data loaders that provide audio file paths and timing information
based on an AudioInputRecipe.

Audio loaders do NOT load audio into memory — they provide paths and
timing so that inference scripts can load with their preferred library
(e.g., librosa, torchaudio).

Mirrors the pattern of video_loaders.py: accepts either an InputRecipes
model (in-process) or a raw dict (subprocess config).
"""

import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from tqdm import tqdm

from .input_recipes import AudioInputRecipe, AudioStreamRecipe, InputRecipes

# Type definitions for the different return formats

# 1. (track_name, file_path, offset_seconds, duration_seconds)
AUDIO_STREAM_WITH_TIMING = Tuple[str, str, float, float]

# 2. (track_name, file_path, chunk_offset_seconds, chunk_duration_seconds)
AUDIO_CHUNK_WITH_TIMING = Tuple[str, str, float, float]


def _resolve_audio_recipe(config: Union[InputRecipes, dict]) -> AudioInputRecipe:
    """
    Extract AudioInputRecipe from either an InputRecipes object or a raw dict.
    """
    if isinstance(config, InputRecipes):
        if config.audio_input_recipe is None:
            raise KeyError("InputRecipes contains no audio_input_recipe.")
        return config.audio_input_recipe

    recipes = InputRecipes.from_dict(config)
    if recipes.audio_input_recipe is None:
        raise KeyError("No audio_input_recipe found in config dictionary")
    return recipes.audio_input_recipe


def get_console_stream():
    """Returns a stream writing directly to the active terminal screen."""
    try:
        console_path = "CONOUT$" if os.name == "nt" else "/dev/tty"
        return open(console_path, "w", encoding="utf-8")
    except Exception:
        return sys.stdout


class AudioDataLoader(ABC):
    """
    Abstract Base Class for loading audio data based on a recipe.

    Handles configuration parsing, track filtering, and timing logic.
    Subclasses must implement __iter__ to define how data is yielded, and
    should override `progress_unit` with the name of the unit yielded (used
    as the default progress-bar unit in `with_progress`).
    """

    progress_unit: str = "it"

    def __init__(
        self,
        config: Union[InputRecipes, dict],
        expected_tracks: Optional[List[str]] = None,
    ):
        """
        Initializes the audio data loader base.

        Args:
            config: Either an InputRecipes model or a dict containing recipes.
                Supports subprocess config format (nested under 'input_recipes')
                and flat format (direct keys).
            expected_tracks: If set, only yield these tracks. If None, yield all.

        Raises:
            KeyError: If audio_input_recipe is missing from config.
            ValueError: If none of the expected tracks are found in the recipe.
        """
        # (1) Extract recipe from config
        recipe = _resolve_audio_recipe(config)

        # (2) Store recipe fields
        self.start_time_ms: float = recipe.start_time_ms
        self.end_time_ms: float = recipe.end_time_ms
        self.streams: Dict[str, AudioStreamRecipe] = recipe.streams

        # (3) Track filtering / validation
        available_tracks = list(self.streams.keys())

        if expected_tracks is not None:
            self.tracks = [t for t in expected_tracks if t in available_tracks]
            missing = set(expected_tracks) - set(self.tracks)
            if missing:
                raise ValueError(
                    f"None/some of the expected tracks {expected_tracks} found. " f"Available: {available_tracks}"
                )
        else:
            self.tracks = available_tracks

    def with_progress(self, desc: Optional[str] = None):
        """
        Enable CLI progress bar for this audio data loader using tqdm.

        Args:
            desc (str, optional): Description displayed in progress bar.

        Returns:
            tqdm: A tqdm iterator wrapping this audio data loader.
        """
        desc = desc or self.__class__.__name__
        return tqdm(
            self,
            desc=desc,
            unit=self.progress_unit,
            file=get_console_stream(),
            bar_format=(
                "[PROGRESS] {desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
            dynamic_ncols=True,
        )

    @property
    def offset_seconds(self) -> float:
        """Start time offset in seconds for time-range loading."""
        return self.start_time_ms / 1000.0

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds for time-range loading."""
        return (self.end_time_ms - self.start_time_ms) / 1000.0

    def get_stream_info(self, track_name: str) -> Dict[str, Any]:
        """Get full stream info dict for a specific track."""
        if track_name not in self.streams:
            raise KeyError(f"Track '{track_name}' not found. Available: {list(self.streams.keys())}")
        return self.streams[track_name].model_dump()

    def get_sample_rate(self, track_name: str) -> int:
        """Get sample rate for a specific track."""
        if track_name not in self.streams:
            raise KeyError(f"Track '{track_name}' not found. Available: {list(self.streams.keys())}")
        return self.streams[track_name].sample_rate

    def get_channels(self, track_name: str) -> int:
        """Get the channels (int) for a specific track"""
        if track_name not in self.streams:
            raise KeyError(f"Track '{track_name}' not found. Available: {list(self.streams.keys())}")
        return self.streams[track_name].channels

    def get_hears_subjects(self, track_name: str) -> List[int]:
        """Get hears subjects list for a specific track."""
        if track_name not in self.streams:
            raise KeyError(f"Track '{track_name}' not found. Available: {list(self.streams.keys())}")
        return self.streams[track_name].hears_subjects

    def get_tracks_for_subject(self, subject_idx: int) -> List[str]:
        """Get track names that can hear a specific subject."""
        result = []
        for name in self.tracks:
            if subject_idx in self.streams[name].hears_subjects:
                result.append(name)
        return result

    def _validate_path(self, path: str) -> bool:
        """Check if an audio file path exists."""
        if not Path(path).exists():
            logging.warning(f"Audio file not found: {path}")
            return False
        return True

    def __len__(self) -> int:
        """Number of audio tracks that will be yielded."""
        return len(self.tracks)

    @abstractmethod
    def __iter__(self):
        """Yields data according to the specific loader implementation."""
        pass


class AudioStreamLoader(AudioDataLoader):
    """
    Iterates over audio tracks, yielding one entry per track with full timing info.

    Yields:
        tuple: (track_name, file_path, offset_seconds, duration_seconds)

    Usage:
        Ideal for detectors that process each audio track as a whole segment
        (e.g., Whisper transcription on the full time range).

        for track_name, path, offset, duration in loader:
            audio, sr = librosa.load(path, sr=None, offset=offset, duration=duration)
    """

    progress_unit = "tracks"

    def __iter__(self) -> Iterator[AUDIO_STREAM_WITH_TIMING]:
        """Yields (track_name, file_path, offset_seconds, duration_seconds)."""
        for track_name in self.tracks:
            stream = self.streams[track_name]
            if not self._validate_path(stream.path):
                continue
            yield track_name, stream.path, self.offset_seconds, self.duration_seconds


class AudioChunkLoader(AudioDataLoader):
    """
    Iterates over audio tracks in fixed-duration chunks with optional overlap.

    Yields:
        tuple: (track_name, file_path, chunk_offset_seconds, chunk_duration_seconds)

    Usage:
        Ideal for detectors that process fixed-length audio segments, e.g.,
        voice activity detection or models with a maximum context window.

        loader = AudioChunkLoader(config, chunk_duration_seconds=30.0, overlap_seconds=5.0)
        for track_name, path, chunk_offset, chunk_dur in loader:
            audio, sr = librosa.load(path, sr=None, offset=chunk_offset, duration=chunk_dur)
    """

    progress_unit = "chunks"

    def __init__(
        self,
        config: Union[InputRecipes, dict],
        chunk_duration_seconds: float,
        overlap_seconds: float = 0.0,
        expected_tracks: Optional[List[str]] = None,
    ):
        """
        Initializes the chunk audio data loader.

        Args:
            config: Either an InputRecipes model or a dict containing recipes.
            chunk_duration_seconds: Length of each chunk in seconds.
            overlap_seconds: Overlap between consecutive chunks in seconds.
            expected_tracks: If set, only yield these tracks. If None, yield all.

        Raises:
            ValueError: If chunk_duration_seconds is not positive or overlap >= chunk.
        """
        super().__init__(config, expected_tracks)

        if chunk_duration_seconds <= 0:
            raise ValueError(f"Invalid chunk_duration_seconds: {chunk_duration_seconds}. Must be positive.")
        if overlap_seconds < 0:
            raise ValueError(f"Invalid overlap_seconds: {overlap_seconds}. Must be non-negative.")
        if overlap_seconds >= chunk_duration_seconds:
            raise ValueError(
                f"overlap_seconds ({overlap_seconds}) must be less than "
                f"chunk_duration_seconds ({chunk_duration_seconds})."
            )
        self.chunk_duration = chunk_duration_seconds
        self.overlap = overlap_seconds

    def __iter__(self) -> Iterator[AUDIO_CHUNK_WITH_TIMING]:
        """Yields (track_name, file_path, chunk_offset_seconds, chunk_duration_seconds)."""
        step = self.chunk_duration - self.overlap
        total_offset = self.offset_seconds
        total_end = total_offset + self.duration_seconds

        for track_name in self.tracks:
            stream = self.streams[track_name]
            if not self._validate_path(stream.path):
                continue

            current = total_offset
            while current < total_end:
                chunk_dur = min(self.chunk_duration, total_end - current)
                yield track_name, stream.path, current, chunk_dur
                current += step
