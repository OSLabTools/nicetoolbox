"""
Shared helpers for the audio metric tests.

Imported directly (`from tests.unit.evaluation.metrics.audio.conftest import ...`) in the
style of tests/unit/evaluation/data/conftest.py, except for the autouse figure cleanup which
pytest collects from this file automatically.
"""

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # must precede any pyplot import in the metric under test

from matplotlib import pyplot as plt  # noqa: E402

from nicetoolbox_core.data.json_schema import (  # noqa: E402
    AudioTranscription,
    AudioTranscriptionTotal,
    JsonMeta,
    TrackTranscription,
)

# All five normalization axes are mandatory in the config, so tests need a full baseline to
# override from rather than relying on defaults that deliberately do not exist.
LENIENT_FLAGS = dict(
    remove_filler=True,
    lower_case=True,
    strip_punctuation=True,
    expand_contractions=True,
    normalize_numbers=True,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def write_transcript(
    tmp_path: Path,
    name: str,
    tracks: dict[str, str],
    *,
    component: str = "audio_transcription",
    algorithm: str = "whisperx",
    subsequence_start: float = 0.0,
    subsequence_length: float = 5.0,
) -> Path:
    """Write a valid transcription JSON containing only `total` text per track.

    Built through the pydantic models rather than a literal dict so a schema change breaks
    this one helper instead of every test fixture.
    """
    model = AudioTranscription(
        meta=JsonMeta(
            component=component,
            algorithm=algorithm,
            subsequence_start=subsequence_start,
            subsequence_length=subsequence_length,
        ),
        tracks={
            track: TrackTranscription(
                total=AudioTranscriptionTotal(text=text, start=0.0, end=subsequence_length),
                segments=[],
                words=[],
            )
            for track, text in tracks.items()
        },
    )
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(model.model_dump()))
    return path
