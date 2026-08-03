"""
WhisperX inference entrypoint.
"""

import copy
import json
import logging
import os
from pathlib import Path

import librosa
import torch
import whisperx

from nicetoolbox_core.audio_loaders import AudioStreamLoader
from nicetoolbox_core.entrypoint import run_inference_entrypoint

# Raw packs read back by whisperx_detector.post_inference
RAW_TRANSCRIPTION_JSON_NAME = "whisperx_transcription_raw.json"
RAW_DIARIZATION_JSON_NAME = "whisperx_diarization_raw.json"
RAW_SPEAKER_ALIGNED_JSON_NAME = "whisperx_speaker_aligned_raw.json"


@run_inference_entrypoint
def whisperx_inference(config: dict) -> None:
    logging.info("Starting WhisperX Inference Pipeline.")

    subjects_description = config["subjects_descr"]
    extra_detector_output_folder = config["out_folders"]["speaker_aligned_transcription"]  # additional detector results

    cache_raw = config["hf_weights_cache_dir"]
    assets_dir = str(Path(cache_raw).expanduser())
    os.makedirs(assets_dir, exist_ok=True)
    os.environ["TORCH_HOME"] = assets_dir

    # to make sure whisperx components don't make a call to hf
    # via snapshot_download and explicitly use downloaded assets
    os.environ["HF_HUB_CACHE"] = assets_dir
    os.environ["HF_HOME"] = assets_dir
    os.environ["HF_HUB_OFFLINE"] = "1"

    model_size = config["model_size"]
    language = config["language"]
    batch_size = config["batch_size"]
    vad_onset = config["vad_onset"]
    vad_offset = config["vad_offset"]
    align_model_name = config["alignment_model_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = config["compute_type"] if device == "cuda" else "int8"

    audio_loader = AudioStreamLoader(config=config, expected_tracks=config["track_names"])

    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        language=language,
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset},
        download_root=assets_dir,
    )
    # makes sure pytorch knows checkpoint subfolder under ../../assets
    torch_hub_checkpoints_dir = os.path.join(assets_dir, "hub", "checkpoints")
    os.makedirs(torch_hub_checkpoints_dir, exist_ok=True)

    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
        model_name=align_model_name,
        model_dir=torch_hub_checkpoints_dir,
    )

    diarize_model = whisperx.diarize.DiarizationPipeline(device=device, cache_dir=assets_dir)

    # Components output storage dicts
    out_transcription = {}
    out_diarization = {}
    out_speaker_aligned = {}

    for track_name, path, offset, duration in audio_loader:
        logging.info(f"Processing track: {track_name}")

        # (1) Load and slice audio track
        sample_rate = 16000  # WhisperX default
        audio, _ = librosa.load(path, sr=sample_rate, offset=offset, duration=duration)
        hears_subjects = audio_loader.get_hears_subjects(track_name)
        logging.info(f"Track '{track_name}' hears subjects: {hears_subjects}")

        # (2) Transcribe
        result = model.transcribe(audio, batch_size=batch_size, language=language)

        # (3) Alignment
        result_aligned = whisperx.align(
            result["segments"], align_model, metadata, audio, device, return_char_alignments=False
        )
        out_transcription[track_name] = result_aligned

        # (4) Diarization
        num_speakers = len(hears_subjects)
        # TODO: do we even need to run diarization on a single speaker?
        diarize_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)
        if num_speakers == 1:
            num_speakers_detected = len(diarize_segments["speaker"].unique())
            if num_speakers_detected != 1:
                logging.warning(
                    f"Expected 1 speaker for track '{track_name}' based on hears_subjects, but pyannote found "
                    f"{num_speakers_detected} unique speakers. Check diarization for this track and tune VAD options."
                )
            else:
                # Assign actual subject name to speaker label since we know there's only 1 speaker for this track
                # based on hears_subjects
                diarize_segments["speaker"] = subjects_description[hears_subjects[0]]

        # Keep only the JSON-serializable turn fields
        out_diarization[track_name] = diarize_segments[["speaker", "start", "end"]].to_dict(orient="records")

        # (5) Assign Speakers
        result_aligned_copy = copy.deepcopy(result_aligned)  # Avoid modifying original aligned results in place
        result_final = whisperx.assign_word_speakers(diarize_segments, result_aligned_copy)
        out_speaker_aligned[track_name] = result_final

        # (6) Save whisperx's built-in SRT as <track>_raw.srt, kept for reference only.
        os.makedirs(extra_detector_output_folder, exist_ok=True)
        writer_options = {
            "highlight_words": True,
            "max_line_width": None,
            "max_line_count": None,
        }
        srt_writer = whisperx.utils.get_writer("srt", extra_detector_output_folder)
        srt_writer({**result_final, "language": language}, f"{track_name}_raw.wav", writer_options)

    # Every component is written as a raw intermediate: post_inference converts them to their
    # schema models and the framework saves each component's final json.
    for component, raw_name, out_dict in [
        ("audio_transcription", RAW_TRANSCRIPTION_JSON_NAME, out_transcription),
        ("audio_diarization", RAW_DIARIZATION_JSON_NAME, out_diarization),
        ("speaker_aligned_transcription", RAW_SPEAKER_ALIGNED_JSON_NAME, out_speaker_aligned),
    ]:
        out_dir = config["out_folders"][component]
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, raw_name), "w") as f:
            json.dump(out_dict, f, indent=4)

    logging.info("WhisperX processing complete.")
