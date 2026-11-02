"""
WhisperX inference entrypoint.
"""

import json
import logging
import os
from pathlib import Path

import librosa
import whisperx

from nicetoolbox_core.audio_loaders import AudioStreamLoader
from nicetoolbox_core.entrypoint import SubprocessResult, run_inference_entrypoint

# Raw pack read back by whisperx_detector.post_inference
RAW_TRANSCRIPTION_JSON_NAME = "whisperx_transcription_raw.json"


@run_inference_entrypoint
def whisperx_inference(config: dict) -> SubprocessResult:
    logging.info("Starting WhisperX Inference Pipeline.")

    out_folder = config["algorithm_out_folder"]  # additional detector results
    cache_raw = config["hf_weights_cache_dir"]
    assets_dir = str(Path(cache_raw).expanduser())

    # to make sure whisperx components don't make a call to hf or torch
    # via snapshot_download and explicitly use downloaded assets
    os.environ["TORCH_HOME"] = assets_dir
    os.environ["HF_HUB_CACHE"] = assets_dir
    os.environ["HF_HOME"] = assets_dir
    os.environ["HF_HUB_OFFLINE"] = "1"

    model_size = config["model_size"]
    language = config["language"]
    batch_size = config["batch_size"]
    vad_onset = config["vad_onset"]
    vad_offset = config["vad_offset"]
    align_model_name = config["alignment_model_name"]
    compute_type = config["compute_type"]
    device = "cuda"

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

    out_transcription = {}

    for track_name, path, offset, duration in audio_loader.with_progress(desc="WhisperX"):
        logging.info(f"Processing track: {track_name}")

        # (1) Load and slice audio track
        sample_rate = 16000  # WhisperX default
        audio, _ = librosa.load(path, sr=sample_rate, offset=offset, duration=duration)

        # (2) Transcribe
        result = model.transcribe(audio, batch_size=batch_size, language=language)

        # (3) Alignment
        result_aligned = whisperx.align(
            result["segments"], align_model, metadata, audio, device, return_char_alignments=False
        )
        out_transcription[track_name] = result_aligned

        # (4) Save whisperx's built-in SRT as <track>_raw.srt, kept for reference only.
        os.makedirs(out_folder, exist_ok=True)
        writer_options = {
            "highlight_words": True,
            "max_line_width": None,
            "max_line_count": None,
        }
        srt_writer = whisperx.utils.get_writer("srt", out_folder)
        srt_writer({**result_aligned, "language": language}, f"{track_name}_raw.wav", writer_options)

    save_file = os.path.join(out_folder, RAW_TRANSCRIPTION_JSON_NAME)
    with open(save_file, "w") as f:
        json.dump(out_transcription, f, indent=4)

    logging.info("WhisperX processing complete.")
    return SubprocessResult.from_artifacts(audio_transcription_json=save_file)
