from pathlib import Path

from pydantic import BaseModel, Field, NonNegativeInt, model_validator

from ...configs.models.video_timestamp import VideoTimestamp


class ElanColumnSpec(BaseModel):
    """Which tab-separated column carries which role in an ELAN tab-delimited export."""

    tier: int
    start: int
    end: int
    annotation: int

    @property
    def width(self) -> int:
        """Number of columns a conforming line must have."""
        return max(self.tier, self.start, self.end, self.annotation) + 1


GAZE_9COL = ElanColumnSpec(tier=0, start=3, end=5, annotation=8)
TRANSCRIPTION_4COL = ElanColumnSpec(tier=0, start=1, end=2, annotation=3)


class ElanSequenceConfig(BaseModel):
    input: Path
    output: Path
    video: Path
    start: NonNegativeInt | VideoTimestamp
    end: int | VideoTimestamp  # -1 means end of video
    reset_frames: bool = False


class ElanImportGazeConfig(BaseModel):
    log_level: str
    log_file_path: Path
    export_csv: bool

    run: dict[str, ElanSequenceConfig]

    subjects: dict[str, str]


class ElanTranscriptionSequence(BaseModel):
    """
    One sequence of a transcription export/import run.

    input/output swap meaning per direction: export reads the transcription JSON and writes
    an ELAN txt; import reads the corrected ELAN txt and writes the transcription JSON.

    The export has no time window: a component output already covers exactly one subsequence, so
    it is exported whole. Windowing lives on the import (see ElanImportTranscriptionSequence).
    """

    input: Path
    output: Path
    tracks: list[str] = Field(min_length=1)


class ElanExportTranscriptionConfig(BaseModel):
    log_level: str
    log_file_path: Path

    export_segments: bool
    export_words: bool

    run: dict[str, ElanTranscriptionSequence]

    @model_validator(mode="after")
    def _at_least_one_family(self) -> "ElanExportTranscriptionConfig":
        if not (self.export_segments or self.export_words):
            raise ValueError(
                "At least one of export_segments, export_words must be true, "
                "otherwise the export writes an empty file."
            )
        return self


class ElanImportTranscriptionConfig(BaseModel):
    log_level: str
    log_file_path: Path

    export_srt: bool
    export_csv: bool

    import_segments: bool
    import_words: bool

    run: dict[str, ElanTranscriptionSequence]

    @model_validator(mode="after")
    def _at_least_one_family(self) -> "ElanImportTranscriptionConfig":
        if not (self.import_segments or self.import_words):
            raise ValueError(
                "At least one of import_segments, import_words must be true, "
                "otherwise the import writes an empty transcript."
            )
        return self
