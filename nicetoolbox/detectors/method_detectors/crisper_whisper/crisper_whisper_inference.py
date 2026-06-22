"""
CrisperWhisper inference entrypoint.
"""

import json
import logging
import os
from pathlib import Path

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from nicetoolbox_core.audio_loaders import AudioStreamLoader
from nicetoolbox_core.entrypoint import run_inference_entrypoint


def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.

    This function is directly ported from the CrisperWhisper repository:
    https://github.com/nyrahealth/CrisperWhisper
    """

    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output


@run_inference_entrypoint
def crisper_whisper_inference(config: dict) -> None:
    """
    Executes CrisperWhisper transcription pipeline natively isolated by the environment.
    """
    logging.info("Starting CrisperWhisper Inference Pipeline.")

    # (1) Unpack config fields
    extra_detector_output_folder = config["out_folder"]  # additional detector results (raw per-track JSONs)

    batch_size = config["batch_size"]
    chunk_length_s = config["chunk_length_s"]
    stride_length_s = config["stride_length_s"]
    _ = config["vad_onset"]  # TODO: Does it exist in CrisperWhisper?
    _ = config["vad_offset"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    compute_type = torch.float32 if device == "cuda:0" else torch.int8

    cache_raw = config.get("hf_weights_cache_dir")
    if not cache_raw:
        raise ValueError("CrisperWhisper config must set hf_weights_cache_dir (see CrisperWhisperConfig).")
    assets_dir = str(Path(cache_raw).expanduser())
    os.makedirs(assets_dir, exist_ok=True)

    # Read exclusively from the weights pre-downloaded by the asset manager (declared as
    # required_assets) and prevent any Hugging Face network calls at inference time.
    os.environ["HF_HOME"] = assets_dir
    os.environ["HF_HUB_CACHE"] = assets_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # (2) Setup audio loader with expected tracks
    audio_loader = AudioStreamLoader(config=config, expected_tracks=config["track_names"])

    # (3) Load transformers model, processor, and pipeline
    logging.info("1. Loading transformers model...")
    model_id = "nyrahealth/CrisperWhisper"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=compute_type,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=assets_dir,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=assets_dir)

    logging.info("2. Configuring transformers pipeline...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        batch_size=batch_size,
        return_timestamps="word",
        torch_dtype=compute_type,
        device=device,
    )

    # (4) Process each track
    for track_name, path, offset, duration in audio_loader:
        logging.info(f"Processing track: {track_name}")

        # (4.1) Load and slice audio track
        sample_rate = 16000
        audio, _ = librosa.load(path, sr=sample_rate, offset=offset, duration=duration)

        # (4.2) Transcribe
        pipeline_results = pipe(audio)

        # (4.3) Adjust crisper whisper timesteps
        results = adjust_pauses_for_hf_pipeline_output(pipeline_results)

        # (4.4) Save raw per-track output; post_inference() transforms this to the standardized format
        os.makedirs(extra_detector_output_folder, exist_ok=True)
        with open(os.path.join(extra_detector_output_folder, f"{track_name}.json"), "w") as f:
            json.dump(results, f, indent=4)

    logging.info("CrisperWhisper processing complete.")
