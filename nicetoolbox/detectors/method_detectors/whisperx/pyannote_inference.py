import json
import logging
import os

import librosa
import torch
from matplotlib.figure import Figure
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.core.notebook import notebook

from nicetoolbox_core.audio_loaders import AudioStreamLoader
from nicetoolbox_core.entrypoint import SubprocessResult, run_inference_entrypoint

# Raw pack read back by pyannote_detector.post_inference
RAW_DIARIZATION_JSON_NAME = "pyannote_diarization_raw.json"
SAMPLE_RATE = 16000  # pyannote expects 16 kHz mono audio
MODEL_NAME = "pyannote/speaker-diarization-community-1"


def _load_pipeline(config: dict, assets_dir: str) -> Pipeline:
    """The diarization pipeline on cuda, with the configured speed and quality knobs applied."""
    pipeline = Pipeline.from_pretrained(MODEL_NAME, cache_dir=assets_dir).to(torch.device("cuda"))

    # construction options, read at every call
    pipeline.segmentation_batch_size = config["segmentation_batch_size"]
    pipeline.embedding_batch_size = config["embedding_batch_size"]
    pipeline.embedding_exclude_overlap = config["embedding_exclude_overlap"]

    # tuned hyperparameters: override one, keep the pretrained values of the rest
    params = pipeline.parameters(instantiated=True)
    params["segmentation"]["min_duration_off"] = config["min_duration_off"]
    pipeline.instantiate(params)
    return pipeline


def _plot_native(annotation: Annotation, track_name: str, duration: float, native_viz_folder: str) -> None:
    """Save pyannote's timeline plot of a track's speaker turns as <native_viz_folder>/<track>.png."""
    # the notebook keeps styles and crop between calls: start fresh and span the whole processed track
    notebook.reset()
    notebook.crop = Segment(0.0, duration)

    fig = Figure(figsize=(notebook.width, 2))
    notebook.plot_annotation(annotation, ax=fig.subplots(), time=True, legend=True)
    fig.savefig(os.path.join(native_viz_folder, f"{track_name}.png"), dpi=100, bbox_inches="tight")


@run_inference_entrypoint
def pyannote_inference(config: dict) -> SubprocessResult:
    logging.info("Starting Pyannote Diarization Pipeline.")

    # make sure the pipeline doesn't call hf via snapshot_download and uses the downloaded assets
    assets_dir = config["hf_weights_cache_dir"]
    os.environ["TORCH_HOME"] = assets_dir
    os.environ["HF_HUB_CACHE"] = assets_dir
    os.environ["HF_HOME"] = assets_dir
    os.environ["HF_HUB_OFFLINE"] = "1"

    # subjects each track's rule expects; None if the rule cannot tell
    track_expected_subjects = config["track_expected_subjects"]

    audio_loader = AudioStreamLoader(config=config, expected_tracks=config["track_names"])
    pipeline = _load_pipeline(config, assets_dir)

    out_diarization = {}
    for track_name, path, offset, duration in audio_loader.with_progress(desc="Pyannote"):
        # (1) Load and slice audio track
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, offset=offset, duration=duration)
        waveform = {"waveform": torch.from_numpy(audio[None, :]), "sample_rate": SAMPLE_RATE}

        # (2) Diarize with exactly the expected number of speakers; none or unknown lets pyannote estimate it
        num_speakers = len(track_expected_subjects[track_name] or []) or None
        logging.info(f"Diarizing track '{track_name}' with {num_speakers or 'unknown'} expected speaker(s).")
        output = pipeline(waveform, num_speakers=num_speakers)
        # exclusive: at most one speaker at a time, overlapped speech goes to the dominant one
        annotation = output.exclusive_speaker_diarization if config["exclusive"] else output.speaker_diarization

        found = len(annotation.labels())
        if num_speakers is not None and found != num_speakers:
            logging.warning(f"Track '{track_name}': expected {num_speakers} speaker(s), pyannote found {found}.")

        out_diarization[track_name] = [
            {"speaker": speaker, "start": turn.start, "end": turn.end}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]

        # (3) Pyannote's own plot of the anonymous speaker turns
        if config["visualize_native"]:
            _plot_native(annotation, track_name, duration, config["native_viz_folder"])

    out_dir = config["algorithm_out_folder"]
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, RAW_DIARIZATION_JSON_NAME)
    with open(raw_path, "w") as f:
        json.dump(out_diarization, f, indent=4)

    logging.info("Pyannote diarization complete.")
    return SubprocessResult.from_artifacts(raw_output=raw_path)
