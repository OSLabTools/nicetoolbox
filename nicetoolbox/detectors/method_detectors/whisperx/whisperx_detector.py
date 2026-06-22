"""
WhisperX method detector class (mock/debug implementation).
"""

import json
import logging
import os

from nicetoolbox_core.audio_loaders import AudioStreamLoader

from ....configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ....utils.video import frames_to_video
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

        logging.info("WhisperX post-inference processing complete. Speaker aligned transcription results collected.")

    def visualization(self, _) -> None:
        """
        Generates visualizations overlaying SRTs subtitles onto video files.

        Uses the generated SRT files from the extra outputs of the audio transcription component. These SRT files
        are raw outputs from WhisperX of the final speaker-aligned transcription segments with speaker labels.

        We create a new video from scratch based on the video frames (if available) or a black background (if no video
        frames are available) and overlay the SRT subtitles onto it.
        """
        if not self.visualize:
            return

        for track_name in self.audio_loader.tracks:
            # Find generated SRT in detector_output directory (extra outputs)
            srt_path = os.path.join(self.out_folders["speaker_aligned_transcription"], f"{track_name}.srt")
            if not os.path.exists(srt_path):
                logging.warning(f"No SRT found for {track_name}, skipping visualization.")
                continue

            if os.path.getsize(srt_path) == 0:
                logging.warning(
                    f"SRT file {srt_path} is empty for {track_name}. This probably means no person was"
                    " detected for this track. Skipping visualization."
                )
                continue

            info = self.audio_loader.get_stream_info(track_name)
            source = info["source_path"]

            viz_dir = self.viz_folders["speaker_aligned_transcription"]
            os.makedirs(viz_dir, exist_ok=True)
            video_out = os.path.join(viz_dir, f"{track_name}.mp4")

            frame_folder = None
            video_recipe = self.data.get_input_recipes().video_input_recipe
            start_frame = self.data.video_start_frame_index
            frame_limit = None
            fps = self.data.fps

            if video_recipe:
                camera_to_use = info.get("camera")
                if not camera_to_use or camera_to_use not in video_recipe.camera_names:
                    camera_to_use = self.data.camera_mapping["cam_front"]

                if camera_to_use and camera_to_use in video_recipe.camera_names:
                    frame_folder = os.path.join(video_recipe.root_path, camera_to_use, "frames")
                start_frame = video_recipe.range_start
                frame_limit = video_recipe.range_end

            logging.info(f"Baking subtitles into {video_out}")
            logging.info(f"Using frame folder: {frame_folder}" if frame_folder else "Using black background fallback.")

            frames_to_video(
                input_folder=frame_folder,
                out_filename=video_out,
                fps=fps,
                start_frame=start_frame,
                audio_path=source,
                srt_path=srt_path,
                frame_limit=frame_limit,
            )
