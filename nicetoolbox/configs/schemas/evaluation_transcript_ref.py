from pathlib import Path

from pydantic import BaseModel

from ..models.no_wildcard_str import NoWildcardStr


class TranscriptRef(BaseModel):
    """
    Direct reference to one track inside one transcription JSON file.

    The text domain currently skips the discovery machinery the numeric path uses
    (see evaluation_input_block.py).
    """

    path: Path  # single .json file; existence is checked at load time, not here
    track: NoWildcardStr  # key into the file's `tracks` mapping, e.g. "room"
