"""
WhisperX method detector class (mock/debug implementation).
"""

import json
import logging
import os

from nicetoolbox_core.audio_loaders import AudioStreamLoader

from ....configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ....utils.srt import SrtWriter
from ....utils.video import render_subtitled_track_video
from ..base_method import BaseMethod


class WhisperX(BaseMethod):
    algorithm_type = "whisperx"
    components = ["audio_transcription", "audio_diarization", "speaker_aligned_transcription"]

    def _initialize_detector(self) -> MethodDetectorRuntime:
        """
        Initializes the WhisperX detector.
        """
        if not self.data.has_audio():
            raise RuntimeError("WhisperX requires audio data but no audio was prepared.")

        # Initialize audio loader for visualization and post-processing
        self.audio_loader = AudioStreamLoader(
            config=self.data.get_input_recipes(), expected_tracks=self.detector_config.track_names
        )

        return super()._initialize_detector()

    def post_inference(self) -> None:
        """
        Process individual speaker aligned transcription json outputs into our final json format.

        Structure:
        {
            "track_name": {
                "total": {
                    "text": "full concatenated transcription text for the track",
                    "start": start_time_of_first_segment,
                    "end": end_time_of_last_segment,
                },
                "segments": [
                    {
                        "start": segment_start_time,
                        "end": segment_end_time,
                        "text": "segment_transcription_text",
                        "avg_logprob": segment_avg_log_probability,
                    },
                    ...
                ],
                "word_segments": [
                    {
                        "word": word_text,
                        "start": word_start_time,
                        "end": word_end_time,
                        "score": word log probability score,
                        "speaker": speaker_label provided by pyannote
                    },
                    ...
                ],
                "language": detected_language
            },
            ...
        }
        """
        folder = self.result_folders["speaker_aligned_transcription"]
        out_dict = {}
        for track_name in self.audio_loader.tracks:
            json_path = os.path.join(self.out_folders["speaker_aligned_transcription"], f"{track_name}.json")
            if not os.path.exists(json_path):
                logging.warning(f"No JSON output found for {track_name} in post_inference, skipping.")
                continue

            with open(json_path) as f:
                track_data = json.load(f)

            segments = track_data["segments"]
            total_text = ""
            total_start = None
            total_end = None

            if segments:
                total_text = " ".join(seg["text"].strip() for seg in segments if seg["text"])
                total_start = segments[0]["start"]
                total_end = segments[-1]["end"]

                # Remove redundant words list from each segment and speaker labels
                for seg in segments:
                    seg.pop("words", None)
                    seg.pop("speaker", None)

            out_dict[track_name] = {
                "total": {"text": total_text, "start": total_start, "end": total_end},
                "segments": segments,
                "word_segments": track_data["word_segments"],
                "language": track_data["language"],
            }

        with open(os.path.join(folder, f"{self.algorithm_instance}.json"), "w") as f:
            json.dump(out_dict, f, indent=4)

        # Generate visualization SRTs from our unified component outputs (consistent across detectors).
        srt_writer = SrtWriter()
        srt_writer.write_tracks(out_dict, self.out_folders["speaker_aligned_transcription"])

        # audio_transcription unified output is written directly by inference; its segments keep
        # their words (no speaker). Load it back to emit matching SRTs for this component too.
        audio_json = os.path.join(self.result_folders["audio_transcription"], f"{self.algorithm_instance}.json")
        if os.path.exists(audio_json):
            with open(audio_json) as f:
                audio_tracks = json.load(f)
            srt_writer.write_tracks(audio_tracks, self.out_folders["audio_transcription"])
        else:
            logging.warning(f"No audio_transcription output found at {audio_json}, skipping its SRT generation.")

        logging.info("WhisperX post-inference processing complete. Speaker aligned transcription results collected.")

    def visualization(self, _) -> None:
        """
        Generates visualizations overlaying SRT subtitles onto video files.

        Uses the SRT files generated by our own SrtWriter in post_inference (one per track per
        component). We render both the audio_transcription and the speaker_aligned_transcription
        components so that any differences introduced by the speaker-alignment phase are visible.

        For each track we create a new video from the video frames (if available) or a black
        background (if not) and overlay the SRT subtitles onto it.
        """
        if not self.visualize:
            return

        for component in ("audio_transcription", "speaker_aligned_transcription"):
            self._visualize_component(component)

    def _visualize_component(self, component: str) -> None:
        """Bake the per-track SRT of a single component into a subtitled video."""
        video_recipe = self.data.get_input_recipes().video_input_recipe
        for track_name in self.audio_loader.tracks:
            info = self.audio_loader.get_stream_info(track_name)

            render_subtitled_track_video(
                srt_path=os.path.join(self.out_folders[component], f"{track_name}.srt"),
                audio_path=info["source_path"],
                output_path=os.path.join(self.viz_folders[component], f"{track_name}.mp4"),
                fps=self.data.fps,
                default_start_frame=self.data.video_start_frame_index,
                video_recipe=video_recipe,
                camera=info.get("camera"),
                fallback_camera=self.data.camera_mapping["cam_front"],
            )
