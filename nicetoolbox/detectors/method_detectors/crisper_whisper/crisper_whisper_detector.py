"""
CrisperWhisper method detector class.
"""

import json
import os

from nicetoolbox_core.audio_loaders import AudioStreamLoader
from nicetoolbox_core.data.json_schema import AudioDiarization, AudioTranscription, SpeakerAlignedTranscription
from nicetoolbox_core.entrypoint import SubprocessResult

from ....configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ....utils import video as vd
from ....utils.srt import SrtWriter
from ...detector_inputs import JsonDetectorInput
from ...detector_outputs import DetectorOutput, JsonDetectorOutput
from ...subjects import to_speaker_aligned_transcription
from ..base_method import BaseMethod
from .crisper_whisper_utils import to_audio_transcription


class CrisperWhisper(BaseMethod):
    algorithm_type = "crisper_whisper"
    components = ["audio_transcription", "speaker_aligned_transcription"]

    # inputs and outputs
    inputs = [
        JsonDetectorInput("audio_diarization", "audio_diarization", schema=AudioDiarization, optional=True),
    ]

    def resolve_outputs(self) -> list[JsonDetectorOutput]:
        outputs = [JsonDetectorOutput("audio_transcription", schema=AudioTranscription)]
        if "audio_diarization" in self.loaded_inputs:  # optional, if have diarization
            outputs.append(JsonDetectorOutput("speaker_aligned_transcription", schema=SpeakerAlignedTranscription))
        return outputs

    def _initialize_detector(self) -> MethodDetectorRuntime:
        if not self.data.has_audio():
            raise RuntimeError("CrisperWhisper requires audio data but no audio was prepared.")

        self.audio_loader = AudioStreamLoader(
            config=self.data.get_input_recipes(),
            expected_tracks=self.detector_config.track_names,
        )
        return super()._initialize_detector()

    def post_inference(self, result_manifest: SubprocessResult) -> DetectorOutput:
        """Convert the raw inference pack into this detector's component output.

        CrisperWhisper produces only word-level chunks, so the segments of the standardized
        transcription are synthesized here (see crisper_whisper_utils). The result shares the
        AudioTranscription schema with WhisperX, so the two backbones stay interchangeable.
        """
        # Timestamps are relative to the subsequence, since inference is handed sliced audio.
        # Recording where it starts lets a consumer place them back on the source timeline.
        raw_path = result_manifest.get_raw_output()

        with open(raw_path) as f:
            raw_transcription = json.load(f)

        # parse raw transcription
        out = DetectorOutput()
        transcription = to_audio_transcription(
            raw_transcription,
            subsequence_start=self.audio_loader.offset_seconds,
            subsequence_length=self.audio_loader.duration_seconds,
            algorithm=self.algorithm_instance,
        )
        out.add_json("audio_transcription", transcription)

        # save it to srt
        srt_writer = SrtWriter()
        srt_folder = self.additional_results_folders["audio_transcription"]
        srt_writer.write_tracks(transcription.tracks, srt_folder)

        # try to assign words/segments to speakers
        if "audio_diarization" in self.loaded_inputs:
            speaker_aligned = to_speaker_aligned_transcription(
                transcription,
                self.loaded_inputs["audio_diarization"].json,
                algorithm=self.algorithm_instance,
            )
            out.add_json("speaker_aligned_transcription", speaker_aligned)
            # save aligned version to srt
            path = self.additional_results_folders["speaker_aligned_transcription"]
            srt_writer.write_tracks(speaker_aligned.tracks, path)

        return out

    def visualization(self, _) -> None:
        # for raw and (if produced) aligned transcription
        for component in [output.component for output in self.declared_outputs]:
            video_recipe = self.data.get_input_recipes().video_input_recipe
            # for each audio track
            for track_name in self.audio_loader.tracks:
                info = self.audio_loader.get_stream_info(track_name)
                vd.render_subtitled_track(
                    srt_path=os.path.join(self.additional_results_folders[component], f"{track_name}.srt"),
                    audio_path=info["source_path"],
                    output_path=os.path.join(self.viz_folders[component], f"{track_name}.mp4"),
                    fps=self.data.fps,
                    default_start_frame=self.data.video_start_frame_index,
                    video_recipe=video_recipe,
                    camera=info.get("camera"),
                    fallback_camera=self.detector_config.visualize_fallback_camera,
                )
