"""
Loading path for transcript (JSON) detector outputs and annotations.

The text-domain counterpart to input_loader.py, sharing nothing with it: transcripts are strings
keyed by track, so none of the axis/frame/alignment logic applies here at the moment.
"""

from nicetoolbox_core.data.json_schema import AudioTranscription, JsonMeta, TrackTranscription

from ...configs.schemas.evaluation_transcript_ref import TranscriptRef
from ...utils.filehandling import load_json_file


def load_transcript(ref: TranscriptRef) -> tuple[JsonMeta, TrackTranscription]:
    """Load one track from a transcription JSON - predictions and ground truth alike.

    Callers consume only `total`; `segments` and `words` are validated but unused, since for now
    only `total` carries annotated content.

    Args:
        ref: Path to a single .json file plus the track to read from it.

    Returns:
        The file's meta block and the requested track.

    Raises:
        FileNotFoundError: The file does not exist.
        ValidationError: The file is not a valid transcription JSON (raised as-is by pydantic,
            so the reported fields point at what actually failed).
        KeyError: The file is valid but has no such track.
    """
    parsed = AudioTranscription.model_validate(load_json_file(ref.path))
    if ref.track not in parsed.tracks:
        raise KeyError(f"Track '{ref.track}' not found in '{ref.path}'. Available: {sorted(parsed.tracks)}.")
    return parsed.meta, parsed.tracks[ref.track]
