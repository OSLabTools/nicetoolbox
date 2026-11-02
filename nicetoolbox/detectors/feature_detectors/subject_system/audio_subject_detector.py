import logging

from nicetoolbox_core.data.json_schema import AudioDiarization, JsonMeta, TrackDiarization

from ....configs.schemas.subject_rules import BaseRuleConfig
from ...detector_inputs import BaseDetectorInput
from ...detector_outputs import DetectorOutput, JsonDetectorOutput
from ...utils.draw_timeline import draw_diarization_timeline, render_diarization_video
from ..base_feature import BaseFeature
from .audio_rules import AssignAudioRule, AudioRangeRule, BaseAudioRule, NoneAudioRule, PathAudioRule

# same dict lookup like in detectors
# put your subject rule handler here to register it
HANDLER_CLASSES = {
    NoneAudioRule,
    AssignAudioRule,
    PathAudioRule,
    AudioRangeRule,
}
SUBJECT_RULE_HANDLERS: dict[str, type[BaseAudioRule]] = {cls.rule_type: cls for cls in HANDLER_CLASSES}


def get_cls(rule_config: BaseRuleConfig) -> type[BaseAudioRule]:
    handler_cls = SUBJECT_RULE_HANDLERS.get(rule_config.rule)
    if handler_cls is None:
        raise ValueError(f"No handler registered for audio subject rule: '{rule_config.rule}'")
    return handler_cls


class AudioSubjectDetector(BaseFeature):
    """Compute per-subject time segments per audio track from configured rules."""

    algorithm_type = "audio_subject_detector"
    components = ["audio_diarization"]

    outputs = [
        JsonDetectorOutput("audio_diarization", schema=AudioDiarization),
    ]

    def resolve_inputs(self) -> list[BaseDetectorInput]:
        # go through all rules and see what they want
        inputs = set()
        for rule_config in self._track_rules().values():
            for spec in get_cls(rule_config).required_inputs(rule_config):
                inputs.add(spec)
        return list(inputs)

    def _initialize_detector(self) -> None:
        # Same subsequence time range the audio handler hands to the audio detectors
        self.subsequence_start = self.data.video_start_ms / 1000.0
        self.subsequence_length = self.data.video_length_ms / 1000.0

    def compute(self) -> DetectorOutput:
        tracks = {}

        # Apply each track's rule
        for track_name, rule_config in self._track_rules().items():
            segments = get_cls(rule_config)(self, track_name, rule_config).compute()
            tracks[track_name] = TrackDiarization(segments=segments)

        logging.info(f"Computed audio_diarization for tracks {list(tracks)}.")

        meta = JsonMeta(
            component="audio_diarization",
            algorithm=self.algorithm_instance,
            subsequence_start=self.subsequence_start,
            subsequence_length=self.subsequence_length,
            tables={"tracks": ["segments"]},
        )
        out = DetectorOutput()
        out.add_json("audio_diarization", AudioDiarization(meta=meta, tracks=tracks))
        return out

    def visualization(self, out: DetectorOutput) -> None:
        audio_diarization = out.get_json("audio_diarization")
        viz_folder = self.viz_folders["audio_diarization"]
        # whole track timeline, one row per subject
        draw_diarization_timeline(audio_diarization, viz_folder, speakers=self.subjects_descr)
        # frame by frame video with a scrolling timeline and the track's audio (needs prepared audio)
        if self.data.has_audio():
            render_diarization_video(
                audio_diarization,
                viz_folder,
                self.data.get_input_recipes(),
                speakers=self.subjects_descr,
                fallback_camera=self.detector_config.visualize_fallback_camera,
            )

    def _track_rules(self) -> dict[str, BaseRuleConfig]:
        """Subject rule of every audio track of the sequence."""
        tracks_cfg = self.subsequence_context.sequence_properties.audio.tracks or {}
        return {track_name: track_cfg.subjects for track_name, track_cfg in tracks_cfg.items()}
