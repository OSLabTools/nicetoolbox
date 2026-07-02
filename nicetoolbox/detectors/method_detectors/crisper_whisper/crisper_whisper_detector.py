"""
CrisperWhisper method detector class.
"""

import json
import logging
import os

from nicetoolbox_core.audio_loaders import AudioStreamLoader

from ....configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ....utils.srt import SrtWriter
from ....utils.video import render_subtitled_track_video
from ..base_method import BaseMethod

SENTENCE_END = frozenset({".", "?", "!"})


def _build_segments_from_chunks(chunks: list) -> tuple[list, list]:
    """Convert CrisperWhisper chunks to (segments, word_segments) in WhisperX-compatible format."""
    word_segments = [
        {"word": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1], "score": None} for c in chunks
    ]

    segments = []
    current: list[dict] = []
    for word in word_segments:
        current.append(word)
        if word["word"].rstrip()[-1:] in SENTENCE_END:
            segments.append(_make_segment(current))
            current = []
    if current:
        segments.append(_make_segment(current))

    return segments, word_segments


def _make_segment(words: list) -> dict:
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["word"] for w in words),
        "words": list(words),
        "avg_logprob": None,
    }


class CrisperWhisper(BaseMethod):
    algorithm_type = "crisper_whisper"
    components = ["audio_transcription"]

    def _initialize_detector(self) -> MethodDetectorRuntime:
        if not self.data.has_audio():
            raise RuntimeError("CrisperWhisper requires audio data but no audio was prepared.")

        self.audio_loader = AudioStreamLoader(
            config=self.data.get_input_recipes(), expected_tracks=self.detector_config.track_names
        )
        return super()._initialize_detector()

    def post_inference(self) -> None:
        """
        Process individual raw crisper whisper json outputs into our final audio_transcription
        component results. We apply the same structure as the default outputs from WhisperX to allow
        for interchangability across the transcription backbones.

        In addition, we generate .srt files for simple sub-title visualizations.
        """
        out_dict = {}
        for track_name in self.audio_loader.tracks:
            raw_path = os.path.join(self.out_folders["audio_transcription"], f"{track_name}.json")
            if not os.path.exists(raw_path):
                logging.warning(f"CrisperWhisper: no raw output found for track '{track_name}', skipping.")
                continue

            with open(raw_path) as f:
                raw = json.load(f)

            segments, word_segments = _build_segments_from_chunks(raw.get("chunks", []))
            out_dict[track_name] = {"segments": segments, "word_segments": word_segments}

        result_dir = self.result_folders["audio_transcription"]
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, f"{self.algorithm_instance}.json"), "w") as f:
            json.dump(out_dict, f, indent=4)

        # Generate SRT files from our unified component output (not the raw chunk output)
        SrtWriter().write_tracks(out_dict, self.out_folders["audio_transcription"])

        logging.info("CrisperWhisper post-inference: standardized transcription and SRT files saved.")

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
