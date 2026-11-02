import json

from nicetoolbox_core.audio_loaders import AudioStreamLoader
from nicetoolbox_core.data.json_schema import AudioDiarization
from nicetoolbox_core.entrypoint import SubprocessResult

from ....configs.schemas.detectors_instances_configs import PyannoteConfig
from ...detector_outputs import DetectorOutput, JsonDetectorOutput
from ...utils.draw_timeline import draw_diarization_timeline, render_diarization_video
from ..base_method import BaseMethod
from .whisperx_utils import to_audio_diarization


class Pyannote(BaseMethod):
    algorithm_type = "pyannote"
    components = ["audio_diarization"]

    outputs = [
        JsonDetectorOutput("audio_diarization", schema=AudioDiarization),
    ]

    def _initialize_detector(self) -> PyannoteConfig.RuntimeConfig:
        if not self.data.has_audio():
            raise RuntimeError(f"Pyannote '{self.algorithm_instance}' requires audio data but no audio was prepared.")

        # initialize audio loader
        self.audio_loader = AudioStreamLoader(
            config=self.data.get_input_recipes(),
            expected_tracks=self.detector_config.track_names,
        )
        # send expected subjects to inference script
        # used only for counting how many clusters to expect
        expected_subjects = self.subsequence_context.track_expected_subjects
        base_runtime = super()._initialize_detector()
        return PyannoteConfig.RuntimeConfig(
            **base_runtime.model_dump(),
            track_expected_subjects=expected_subjects,
        )

    def post_inference(self, result_manifest: SubprocessResult) -> DetectorOutput:
        with open(result_manifest.get_raw_output()) as f:
            raw_diarization = json.load(f)

        diarization = to_audio_diarization(
            raw_diarization,
            subsequence_start=self.audio_loader.offset_seconds,
            subsequence_length=self.audio_loader.duration_seconds,
            algorithm=self.algorithm_instance,
        )

        out = DetectorOutput()
        out.add_json("audio_diarization", diarization)
        return out

    def visualization(self, out: DetectorOutput) -> None:
        audio_diarization = out.get_json("audio_diarization")
        viz_folder = self.viz_folders["audio_diarization"]
        # whole track timeline with segments
        draw_diarization_timeline(audio_diarization, viz_folder)
        # frame by frame video with a scrolling timeline and the track's audio
        render_diarization_video(
            audio_diarization,
            viz_folder,
            self.data.get_input_recipes(),
            fallback_camera=self.detector_config.visualize_fallback_camera,
        )
