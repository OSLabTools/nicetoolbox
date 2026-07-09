"""
Data module handling the data loading and processing of the give datasets.
"""

import logging
from typing import Any, Dict, Optional

from nicetoolbox_core.input_recipes import AudioInputRecipe, InputRecipes, VideoInputRecipe

from ..configs.models.video_timestamp import timestamp_to_ms
from .data_handlers.audio_handler import AudioDataHandler
from .data_handlers.video_handler import VideoDataHandler
from .in_out import SequenceIO
from .subsequence_context import SubsequenceContext


class SequenceData:
    """
    Facade for NICE Toolbox input data preparation.

    Determines which modalities to prepare based on 'input_data_type' fields
    in the selected detector/algorithm configs from detectors_config.toml.

    Supported input_data_type values:
    - "video"  : video files / frame sequences (always prepared by default)
    - "audio"  : audio tracks extracted from video or standalone files
    - "frames" : (future) non-temporal image datasets

    Algorithms may declare a single string or a list for cross-modal use:
        input_data_type = "video"
        input_data_type = ["video", "audio"]

    Feature detectors (those with input_detector_names instead of
    input_data_type) do not trigger raw data preparation directly.
    """

    # Frame-based attributes (set after video handler prepares)
    video_start_frame_index: int
    video_length_frames: int
    fps: int

    def __init__(self, subsequence_context: SubsequenceContext, io: SequenceIO) -> None:
        """
        Initialize data facade and orchestrate data handling.
        """
        logging.info("Start DATA PREPARATION.")

        self.io: SequenceIO = io
        self.sequence_context = subsequence_context

        # --- START: Config Parameters / Meta data used by detectors ---
        self.dataset_name = subsequence_context.dataset_name
        self.subjects_descr = subsequence_context.subjects_descr
        self.all_camera_names = subsequence_context.all_camera_names

        self.sequence_id = subsequence_context.run_sequence.sequence_id

        self.video_skip_frames = None  # Hardcoded - No access via config yet
        self.annotation_interval = 2.0  # Keep? Hardcoded - No access via config yet

        sequence_properties = subsequence_context.sequence_properties
        cameras = sequence_properties.video.cameras
        # TODO: remove cam_sees_subjects?
        self.cam_sees_subjects = {name: t.sees_subjects for name, t in cameras.items()}
        # --- END: Config Parameters / Meta data used by detectors ---
        # (1) Always prepare video data (main source)
        self._video_handler = VideoDataHandler(io, subsequence_context)
        self._video_handler.prepare()

        # Expose resolved values for detectors and audio handler
        # TODO: use video handler or video recipe directly?
        self.fps = self._video_handler.fps
        self.video_start_frame_index = self._video_handler.start_frame
        self.video_length_frames = self._video_handler.length_frames
        self.video_start_ms = timestamp_to_ms(subsequence_context.video_start, self.fps)
        self.video_length_ms = timestamp_to_ms(subsequence_context.video_length, self.fps)

        # if need full video - take resolved video length in frames
        if self.video_length_ms < 0:
            self.video_length_ms = timestamp_to_ms(self.video_length_frames, self.fps)
        assert self.video_start_ms <= self.video_start_ms + self.video_length_ms

        # (2) Prepare audio if available
        self._audio_handler: AudioDataHandler | None = None
        tracks_cfg = sequence_properties.audio.tracks
        if tracks_cfg:
            self._audio_handler = AudioDataHandler(
                io, subsequence_context, self.video_start_ms, self.video_length_ms, tracks_cfg
            )
            self._audio_handler.prepare()
            if not self._audio_handler.is_available:
                logging.warning("Audio preparation produced no usable tracks.")

        logging.info("DATA PREPARATION complete.\n")

    # -------------------------------------------------------------------------
    # Properties (used by detectors)
    # -------------------------------------------------------------------------

    @property
    def calibration(self) -> Optional[Dict[str, Any]]:
        """Camera calibration data (video-specific)."""
        if self._video_handler:
            return self._video_handler.calibration
        return None

    # -------------------------------------------------------------------------
    # Recipe access
    # -------------------------------------------------------------------------

    def get_video_input_recipe(self) -> VideoInputRecipe:
        """Get video/frame input recipe for video data loaders."""
        return self._video_handler.get_recipe()

    def get_audio_input_recipe(self) -> Optional[AudioInputRecipe]:
        """Get audio input recipe for audio data loaders."""
        if self._audio_handler and self._audio_handler.is_available:
            return self._audio_handler.get_recipe()
        return None

    def get_input_recipes(self) -> InputRecipes:
        """
        Get composed InputRecipes containing all available modality recipes.

        This is the primary method used by BaseMethod to build the runtime
        config for subprocesses. Each recipe is validated via Pydantic.

        Returns:
            InputRecipes with video and/or audio recipes set.
        """
        return InputRecipes(
            video_input_recipe=self.get_video_input_recipe(),
            audio_input_recipe=self.get_audio_input_recipe(),
        )

    def has_audio(self) -> bool:
        """Check if audio data is available."""
        return self._audio_handler is not None and self._audio_handler.is_available
