import math
from typing import Optional

from nicetoolbox_core.data.json_schema import (
    AlignedTranscription,
    AudioDiarization,
    AudioTranscription,
    JsonMeta,
    SpeakerAlignedTranscription,
    TrackDiarization,
    TrackTranscription,
)


def logprob_to_confidence(avg_logprob: Optional[float]) -> Optional[float]:
    """Convert logprob to normalized confidence."""
    if avg_logprob is None:
        return None
    try:
        confidence = math.exp(float(avg_logprob))
    except (OverflowError, ValueError, TypeError):
        return None
    return confidence if math.isfinite(confidence) else None


def _mean(values: list[Optional[float]]) -> Optional[float]:
    """Mean of the present values, or None when there are none."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _speaker_confidence(speaker: Optional[str], raw_words: list[dict]) -> float:
    """Fraction of a segment's words carrying exactly this segment's speaker."""
    if not raw_words:
        return 0.0
    agreeing = sum(1 for word in raw_words if word.get("speaker") is not None and word.get("speaker") == speaker)
    return agreeing / len(raw_words)


def _to_word(raw: dict, segment_index: Optional[int] = None, with_speaker: bool = False) -> dict:
    """A word, from whisperx's aligned word record."""
    word = {
        "word": raw["word"],
        "segment_index": segment_index,
        # Alignment can fail to place a word, leaving it without timings.
        "start": raw.get("start"),
        "end": raw.get("end"),
        "confidence": raw.get("score"),
    }
    if with_speaker:
        word["speaker"] = raw.get("speaker")
    return word


def _to_segment(raw: dict, with_speaker: bool = False) -> dict:
    """A segment, from whisperx's segment record."""
    segment = {
        "text": raw["text"],
        "start": raw["start"],
        "end": raw["end"],
        "confidence": logprob_to_confidence(raw.get("avg_logprob")),
    }

    # check if we have diarization info
    if with_speaker:
        speaker = raw.get("speaker")
        segment["speaker"] = speaker
        segment_words = raw.get("words", [])
        segment["speaker_confidence"] = _speaker_confidence(speaker, segment_words)
    return segment


def _parse(raw_track: dict, with_speaker: bool = False) -> tuple[list[dict], list[dict]]:
    """Extract whisperx words and segments."""
    segments = []
    words = []
    for index, raw_segment in enumerate(raw_track["segments"]):
        # extract segment
        segment = _to_segment(raw_segment, with_speaker=with_speaker)
        segments.append(segment)

        # extract all nested words in current segment
        raw_words = raw_segment.get("words", [])
        for raw_word in raw_words:
            word = _to_word(raw_word, segment_index=index, with_speaker=with_speaker)
            words.append(word)

    return segments, words


def _to_turn(raw: dict) -> dict:
    """A speaker turn, from a pyannote diarization record."""
    return {
        # pyannote labels are not guaranteed to be strings once a subject name is substituted in.
        "speaker": str(raw["speaker"]),
        "start": raw["start"],
        "end": raw["end"],
    }


def _to_total(segments: list[dict], audio_end: float) -> dict:
    """The track's overall span and text, summarised from its segments."""
    text = " ".join(seg["text"].strip() for seg in segments if seg["text"] and seg["text"].strip())
    starts = [seg["start"] for seg in segments if seg["start"] is not None]
    ends = [seg["end"] for seg in segments if seg["end"] is not None]

    return {
        "text": text,
        "start": starts[0] if starts else 0.0,
        "end": ends[-1] if ends else audio_end,
        "confidence": _mean([seg["confidence"] for seg in segments]),
    }


def to_audio_transcription(
    raw_tracks: dict, subsequence_start: float, subsequence_length: float, algorithm: str
) -> AudioTranscription:
    """Convert raw per-track WhisperX aligned results into the AudioTranscription model.

    Args:
        raw_tracks: {track_name: AlignedTranscriptionResult} as produced by whisperx.align.
        subsequence_start: Source-recording second this subsequence begins at.
        subsequence_length: Length of the subsequence in seconds. Not recorded in meta; used only
            as the total's end for a track that transcribed to nothing.
        algorithm: Detector instance name that produced this output.
    """
    tracks = {}
    for track_name, raw in raw_tracks.items():
        segments, words = _parse(raw)
        total = _to_total(segments, audio_end=subsequence_length)
        tracks[track_name] = TrackTranscription(words=words, segments=segments, total=total)

    meta = JsonMeta(
        component="audio_transcription",
        algorithm=algorithm,
        subsequence_start=subsequence_start,
        tables={"tracks": ["segments", "words"]},
    )
    return AudioTranscription(meta=meta, tracks=tracks)


def to_audio_diarization(raw_tracks: dict, subsequence_start: float, algorithm: str) -> AudioDiarization:
    """Convert raw per-track pyannote diarization records into the AudioDiarization model.

    Args:
        raw_tracks: {track_name: [record, ...]} as produced by the diarization pipeline.
        subsequence_start: Source-recording second this subsequence begins at; turn timings are
            relative to it.
        algorithm: Detector instance name that produced this output.
    """
    tracks = {}
    for track_name, records in raw_tracks.items():
        segments = [_to_turn(record) for record in records]
        tracks[track_name] = TrackDiarization(segments=segments)

    meta = JsonMeta(
        component="audio_diarization",
        algorithm=algorithm,
        subsequence_start=subsequence_start,
        tables={"tracks": ["segments"]},
    )
    return AudioDiarization(meta=meta, tracks=tracks)


def to_speaker_aligned_transcription(
    raw_tracks: dict, subsequence_start: float, subsequence_length: float, algorithm: str
) -> SpeakerAlignedTranscription:
    """Convert raw per-track speaker-assigned WhisperX results into the aligned model.

    Args:
        raw_tracks: {track_name: result} as produced by whisperx.assign_word_speakers.
        subsequence_start: Source-recording second this subsequence begins at.
        subsequence_length: Length of the subsequence in seconds. Not recorded in meta; used only
            as the total's end for a track that transcribed to nothing.
        algorithm: Detector instance name that produced this output.
    """
    tracks = {}
    for track_name, raw in raw_tracks.items():
        segments, words = _parse(raw, with_speaker=True)
        total = _to_total(segments, audio_end=subsequence_length)
        tracks[track_name] = AlignedTranscription(words=words, segments=segments, total=total)

    meta = JsonMeta(
        component="speaker_aligned_transcription",
        algorithm=algorithm,
        subsequence_start=subsequence_start,
        tables={"tracks": ["segments", "words"]},
    )
    return SpeakerAlignedTranscription(meta=meta, tracks=tracks)
