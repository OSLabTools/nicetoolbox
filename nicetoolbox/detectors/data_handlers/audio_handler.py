"""
Audio data extraction and organization handler.

Extracts audio from video files or locates standalone audio files,
organizes them into the nicetoolbox_input/audio/ folder.

DESIGN DECISIONS:
1. Audio is ALWAYS extracted in FULL (no time-range cutting).
   The time range is passed in the recipe for inference scripts to use
   with librosa's offset/duration parameters.
2. No preprocessing (resampling, normalization) — inference scripts own that.
3. Track configuration from dataset_properties.toml drives extraction.
"""

import glob as glob_module
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from nicetoolbox_core.input_recipes import AudioInputRecipe, AudioStreamRecipe

from ...configs.schemas.dataset_properties import AudioTrackConfig
from ...configs.video_runtime_config import SequenceRuntimeConfig
from ...utils.logging_utils import log_with_underscore
from ..in_out import SequenceIO
from .handler import BaseModalityHandler


class AudioDataHandler(BaseModalityHandler):
    """
    Handles audio data extraction and organization.

    Uses the track configuration from dataset_properties to determine
    which audio streams to extract and from where.
    """

    def __init__(
        self,
        # Shared fields (passed to base)
        io: SequenceIO,
        sequence_context: SequenceRuntimeConfig,
        audio_start_ms: float,
        audio_length_ms: float,
        tracks_config: dict[str, AudioTrackConfig],
    ):
        super().__init__(io, sequence_context)
        self.audio_start_ms = audio_start_ms
        self.audio_length_ms = audio_length_ms
        self.audio_end_ms = audio_start_ms + audio_length_ms
        self.tracks_config = tracks_config

        self.audio_output_folder = self.nice_input_folder / "audio"
        self._streams: Dict[str, AudioStreamRecipe] = {}

    @property
    def modality_name(self) -> str:
        return "audio"  # TODO: Do we actually need this property?

    @property
    def streams(self) -> Dict[str, AudioStreamRecipe]:
        return self._streams

    def prepare(self) -> None:
        """
        Prepare audio data based on track configuration.

        Audio is always extracted in FULL. Already-extracted files are reused.
        Sets self._available = True if at least one track was prepared.
        """
        log_with_underscore("Preparing Audio Modality...")
        self.audio_output_folder.mkdir(parents=True, exist_ok=True)

        for track_name, track_cfg in self.tracks_config.items():
            self._prepare_track(track_name, track_cfg)

        self._available = len(self._streams) > 0
        logging.info(f"Audio preparation complete. {len(self._streams)} track(s): {list(self._streams.keys())}")

    def get_recipe(self) -> AudioInputRecipe:
        """Build audio input recipe with full file paths + time range."""
        return AudioInputRecipe(
            streams=self._streams.copy(),
            start_time_ms=self.audio_start_ms,
            end_time_ms=self.audio_end_ms,
        )

    # -------------------------------------------------------------------------
    # Helper methods for track preparation
    # -------------------------------------------------------------------------

    def _prepare_track(self, track_name: str, track_cfg: AudioTrackConfig) -> None:
        """Prepare a single audio track (embedded or standalone)."""
        stream_idx = track_cfg.stream
        channel_idx = track_cfg.channel
        hears = track_cfg.hears_subjects
        logging.info(f"Extracting audio track '{track_name}', stream: {stream_idx}, channel: {channel_idx}")

        camera = None
        # check where to look for a file
        if track_cfg.is_embedded:
            logging.info(f"Audio track is embedded into video, looking for camera {track_cfg.camera}...")
            source_path = self._find_video_for_camera(track_cfg.camera)
            source_type = "embedded"
            camera = track_cfg.camera
        elif track_cfg.is_standalone:
            source_path = Path(track_cfg.path)
            source_type = "standalone"
        else:
            raise ValueError("Audio track should be embedded or standalone")

        if not source_path.exists():
            raise FileNotFoundError(f"Audio track '{track_name}': file not found: '{source_path}'")
        logging.info(f"Audio track file: '{source_path}'")

        # did we already extracted it before?
        output_path = self.audio_output_folder / f"{track_name}.wav"
        if output_path.exists():
            logging.info(f"Audio track '{track_name}' already prepared: '{output_path}'")
            self._register_file(track_name, source_type, hears, output_path, source_path, camera=camera)
            return

        # validate input file with ffprobe before processing
        logging.info("Extracting Audio Specifications...")
        audio_streams = self._probe_file(source_path)
        self._dump_probe_json(output_path, audio_streams)
        logging.info("Validating Audio Specifications...")
        self._validate_stream_and_channel(track_name, source_path, stream_idx, channel_idx, audio_streams)

        # extract with ffmpeg
        logging.info("Extracting Audio Track from file...")
        self._ffmpeg_extract_full(source_path, output_path, stream_idx, channel_idx)
        self._register_file(track_name, source_type, hears, output_path, source_path, camera=camera)
        logging.info(f"Audio Track extracted: {output_path}")

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def _register_file(
        self,
        track_name: str,
        source_type: str,
        hears_subjects: List[int],
        output_path: Path,
        source_path: Optional[Path] = None,
        camera: Optional[str] = None,
    ) -> None:
        """Probe and register an audio file as a stream."""
        streams = self._probe_file(output_path)
        if len(streams) != 1:
            raise RuntimeError(f"Expected exactly 1 audio stream in '{output_path}', got {len(streams)}")
        stream = streams[0]

        # Write ffprobe json data to file inside nicetoolbox_input folder
        # This is currently used only for logging and debugging
        self._dump_probe_json(output_path, streams, suffix="_wav_meta")

        # Validate that we have relevant fields
        # TODO: pydantic and type validation?
        sample_rate = stream.get("sample_rate")
        channels = stream.get("channels")
        if sample_rate is None or channels is None:
            raise RuntimeError(f"ffprobe missing sample_rate/channels for '{output_path}': {stream}")

        self._streams[track_name] = AudioStreamRecipe(
            path=str(output_path),
            source_path=str(source_path),
            sample_rate=int(sample_rate),
            channels=int(channels),
            source_type=source_type,
            camera=camera,
            hears_subjects=hears_subjects,
        )

    # -------------------------------------------------------------------------
    # File discovery (video for embedded tracks)
    # -------------------------------------------------------------------------

    def _find_video_for_camera(self, camera_name: str) -> Optional[Path]:
        """Find the video file for a specific camera using IO."""
        VIDEO_EXTS = [".mp4", ".avi"]  # TODO: Add more extensions
        src = self.io.get_data_source_folder(camera_name)

        # TODO: STRONG assumptions here...
        # yeah, definitely asking for troubles
        for ext in VIDEO_EXTS:
            video_files = glob_module.glob(str(src / f"*{ext}"))
            matches = [path for path in video_files if camera_name in path]
            if matches:
                return Path(matches[0])

        raise ValueError(f"No video file found for camera '{camera_name}' in {src}")

    # -------------------------------------------------------------------------
    # FFmpeg / FFprobe helpers
    # -------------------------------------------------------------------------

    def _dump_probe_json(self, path: Path, streams: List[Dict[str, Any]], suffix: str = "_audio_meta") -> None:
        """Dump ffprobe stream data to a JSON file in the audio output folder for debugging."""
        dump_path = self.audio_output_folder / f"{path.stem}{suffix}.json"
        with open(dump_path, "w") as f:
            json.dump(streams, f, indent=4)

    def _validate_stream_and_channel(
        self,
        track_name: str,
        path: Path,
        stream_idx: int,
        channel_idx: Optional[int],
        audio_streams: List[Dict[str, Any]],
    ) -> None:
        """Validate that stream_idx and channel_idx exist in audio_streams. Raises on failure."""
        if not audio_streams:
            raise RuntimeError(f"Track '{track_name}': no audio streams found in {path}")
        if stream_idx >= len(audio_streams):
            raise RuntimeError(
                f"Track '{track_name}': stream {stream_idx} not found in "
                f"{path} ({len(audio_streams)} audio stream(s))"
            )
        if channel_idx is not None and channel_idx >= audio_streams[stream_idx]["channels"]:
            raise RuntimeError(
                f"Track '{track_name}': channel {channel_idx} not found in "
                f"stream {stream_idx} of {path} ({audio_streams[stream_idx]['channels']} channel(s))"
            )

    def _probe_file(self, path: Path) -> List[Dict[str, Any]]:
        """Probe a file and return all audio streams as a list of dicts. Raises on failure."""
        # fmt: off
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a",
            str(path),
        ]
        # fmt: on

        # call ffprobe and try get meta information as a json
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed for {path}: {e.stderr}") from None

        # try to parse it
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"ffprobe returned invalid JSON for {path}: {e}") from None

        # let's validate that we have relevant fields
        streams = data.get("streams", [])
        return streams

    def _ffmpeg_extract_full(
        self, input_path: Path, output_path: Path, stream_index: int, channel_index: int | None = None
    ) -> bool:
        """Extract full-length audio stream to WAV"""
        # fmt: off
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", str(input_path),
            "-map", f"0:a:{stream_index}",  # audio stream specific index
            "-c:a", "pcm_s16le",
            "-f", "wav",
        ]
        # fmt: on

        # check if need to extract a specific audio channel
        if channel_index is not None:
            # 'pan=mono|c0=cX' forces the output to be mono and maps input channel X to output channel 0
            cmd.extend(["-af", f"pan=mono|c0=c{channel_index}"])
        # finally add output path
        cmd.append(str(output_path))

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed for {input_path}: {e.stderr}") from None

        logging.info(f"Extracted audio to '{output_path}'")
