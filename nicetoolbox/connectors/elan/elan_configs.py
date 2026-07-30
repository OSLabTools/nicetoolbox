from pathlib import Path
from typing import Literal

from pydantic import BaseModel, NonNegativeInt, model_validator

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
    """

    input: Path
    output: Path
    start: NonNegativeInt | VideoTimestamp = 0
    end: int | VideoTimestamp = -1  # -1 means end of recording


class ElanExportTranscriptionConfig(BaseModel):
    log_level: str
    log_file_path: Path

    # Word/speaker tiers are opt-in: segment-level text is the default deliverable.
    include_words: bool = False
    include_speaker: bool = False

    # Per-speaker segment tiers (<track>__<speaker>) for multi-speaker tracks — easier to label.
    include_speaker_segments: bool = False
    # How to place a segment whose words span multiple speakers: "dominant" keeps it whole under the
    # majority speaker; "split" breaks it into separate segments at speaker changes.
    mixed_segment_strategy: Literal["dominant", "split"] = "dominant"

    run: dict[str, ElanTranscriptionSequence]

    @model_validator(mode="after")
    def _speaker_needs_words(self) -> "ElanExportTranscriptionConfig":
        # Speaker labels are stored per word (word_segments[].speaker)
        if self.include_speaker and not self.include_words:
            raise ValueError(
                "include_speaker = true requires include_words = true: speaker labels are stored "
                "per word. Standalone speaker turns belong to the audio_diarization component."
            )
        # The import rebuilds word_segments (with unchanged timings) from the word tier, so the
        # per-speaker segment cycle needs the word tier present.
        if self.include_speaker_segments and not self.include_words:
            raise ValueError(
                "include_speaker_segments = true requires include_words = true: the import rebuilds "
                "word_segments from the word tier, keeping word timings unchanged."
            )
        return self


class ElanImportTranscriptionConfig(BaseModel):
    log_level: str
    log_file_path: Path

    run: dict[str, ElanTranscriptionSequence]
