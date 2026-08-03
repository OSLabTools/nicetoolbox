"""
CrisperWhisper method detector class.
"""

import json
import os

from nicetoolbox_core.audio_loaders import AudioStreamLoader
from nicetoolbox_core.data.json_schema import AudioTranscription

from ....configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ....utils.srt import SrtWriter
from ....utils.video import render_subtitled_track_video
from ...detector_outputs import DetectorOutput, JsonDetectorOutput
from ..base_method import BaseMethod
from .crisper_whisper_utils import to_audio_transcription

# Raw pack written by crisper_whisper_inference.py (kept in sync manually; importing the inference
# module here would pull its crisper_whisper-venv-only deps into the main toolbox env).
RAW_TRANSCRIPTION_JSON_NAME = "crisper_whisper_transcription_raw.json"


class CrisperWhisper(BaseMethod):
    algorithm_type = "crisper_whisper"
    components = ["audio_transcription"]

    outputs = [JsonDetectorOutput("audio_transcription", schema=AudioTranscription)]

    def _initialize_detector(self) -> MethodDetectorRuntime:
        if not self.data.has_audio():
            raise RuntimeError("CrisperWhisper requires audio data but no audio was prepared.")

        self.audio_loader = AudioStreamLoader(
            config=self.data.get_input_recipes(),
            expected_tracks=self.detector_config.track_names,
        )
        return super()._initialize_detector()

    def post_inference(self) -> DetectorOutput:
        """Convert the raw inference pack into this detector's component output.

        CrisperWhisper produces only word-level chunks, so the segments of the standardized
        transcription are synthesized here (see crisper_whisper_utils). The result shares the
        AudioTranscription schema with WhisperX, so the two backbones stay interchangeable.
        """
        # Timestamps are relative to the subsequence, since inference is handed sliced audio.
        # Recording where it starts lets a consumer place them back on the source timeline.
        raw_path = os.path.join(self.out_folders["audio_transcription"], RAW_TRANSCRIPTION_JSON_NAME)
        with open(raw_path) as f:
            raw_transcription = json.load(f)

        transcription = to_audio_transcription(
            raw_transcription,
            subsequence_start=self.audio_loader.offset_seconds,
            subsequence_length=self.audio_loader.duration_seconds,
            algorithm=self.algorithm_instance,
        )

        # SRTs feed the subtitled-video visualization.
        SrtWriter().write_tracks(transcription.tracks, self.out_folders["audio_transcription"])

        out = DetectorOutput()
        out.add_json("audio_transcription", transcription)
        return out

    def visualization(self, _) -> None:
        """
        Visualizes the transcription results by overlaying subtitles on the video frames.
        Uses the generated SRT files from the extra outputs generated via post-inference.
        """
        if not self.visualize:
            return

        video_recipe = self.data.get_input_recipes().video_input_recipe
        for track_name in self.audio_loader.tracks:
            info = self.audio_loader.get_stream_info(track_name)

            render_subtitled_track_video(
                srt_path=os.path.join(self.out_folders["audio_transcription"], f"{track_name}.srt"),
                audio_path=info["source_path"],
                output_path=os.path.join(self.viz_folders["audio_transcription"], f"{track_name}.mp4"),
                fps=self.data.fps,
                default_start_frame=self.data.video_start_frame_index,
                video_recipe=video_recipe,
                camera=info.get("camera"),
                fallback_camera=self.detector_config.fallback_camera,
            )
