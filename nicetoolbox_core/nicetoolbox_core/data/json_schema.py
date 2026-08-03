from typing import Dict, List, Optional

from pydantic import BaseModel

# =============================================================================
# Meta
# =============================================================================


class JsonMeta(BaseModel):
    """Provenance and layout for a JSON component output."""

    # which component and detector instance produced this file
    component: str
    algorithm: str
    # ogirinal subsequence start in seconds
    subsequence_start: float
    # subsequence duration in seconds,
    subsequence_length: float
    # helper for generating csv tables
    # e.g. {"tracks": ["segments", "words"]} means: iterate the tracks mapping and, for
    # each entry, emit one CSV per listed field with a row per element.
    tables: Dict[str, List[str]] = {}


# =============================================================================
# Transcription models
# =============================================================================


class AudioTranscriptionTotal(BaseModel):
    # all segments text combined
    text: str
    # first segment start, fallback to audio file start
    start: float
    # last segment end, fallback to audio file end
    end: float
    # mean confidence of all segments
    confidence: Optional[float] = None


class AudioTranscriptionSegment(BaseModel):
    # logical text segment, usually sentence
    text: str
    # start of the segment, fallback to first word start
    start: Optional[float] = None
    # end of the segment, fallback to last word end
    end: Optional[float] = None
    # in whisper, exp(avg_logprob) of the segment, mean word confidence otherwise
    confidence: Optional[float] = None


class AudioTranscriptionWord(BaseModel):
    # single word, usually separated by white spaces
    word: str
    # parent segment to which this word belongs to, null if stray word
    segment_index: Optional[int] = None
    # start of the word if successfully aligned
    start: Optional[float] = None
    # end of the word if successfully aligned
    end: Optional[float] = None
    # in wav2vec alignment score, if available
    confidence: Optional[float] = None


class TrackTranscription(BaseModel):
    total: AudioTranscriptionTotal
    segments: List[AudioTranscriptionSegment]
    words: List[AudioTranscriptionWord]


class AudioTranscription(BaseModel):
    """A transcription component's output: one entry per audio track."""

    meta: JsonMeta
    tracks: Dict[str, TrackTranscription]


# =============================================================================
# Diarization models
# =============================================================================


class DiarizationSegment(BaseModel):
    # speaker identity for the segment
    speaker: str
    # start of the segment
    start: float
    # end of the segment
    end: float


class TrackDiarization(BaseModel):
    segments: List[DiarizationSegment]


class AudioDiarization(BaseModel):
    """A diarization component's output: the speaker turns found on each audio track."""

    meta: JsonMeta
    tracks: Dict[str, TrackDiarization]


# =============================================================================
# Speaker aligned transcription
# =============================================================================


class AlignedWord(AudioTranscriptionWord):
    # dominant overlapping speaker; None when diarization failed
    # doesn't guarantee to match segment spekaer
    speaker: Optional[str]


class AlignedSegment(AudioTranscriptionSegment):
    # dominant spekaer for this segment; None when diarization failed for all words
    speaker: Optional[str]
    # the fraction of its words carrying exactly this speaker.
    # 1.0 = every word agrees; 0.0 = none do (all unlabeled, all a
    # different speaker, or the segment has no words at all).
    speaker_confidence: float


class AlignedTranscription(BaseModel):
    total: AudioTranscriptionTotal
    segments: List[AlignedSegment]
    words: List[AlignedWord]


class SpeakerAlignedTranscription(BaseModel):
    """A speaker-aligned transcription component's output: one entry per audio track.

    Unlike AudioTranscription, every segment and word carries the speaker label assigned by
    diarization.
    """

    meta: JsonMeta
    tracks: Dict[str, AlignedTranscription]
