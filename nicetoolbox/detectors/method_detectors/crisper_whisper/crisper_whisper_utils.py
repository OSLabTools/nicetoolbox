from nicetoolbox_core.data.json_schema import (
    AudioTranscription,
    AudioTranscriptionSegment,
    AudioTranscriptionTotal,
    AudioTranscriptionWord,
    JsonMeta,
    TrackTranscription,
)

# A word ending in one of these closes a segment. CrisperWhisper emits only word-level chunks,
# so sentence boundaries are the one segmentation signal available without re-running a model.
SENTENCE_END = frozenset({".", "?", "!"})

# Upper bounds on a segment. Spontaneous speech is full of run-on sentences — a single spoken
# "sentence" can hold several clauses joined by "and"/"so" and last 20s — which SENTENCE_END alone
# will not break. These caps bound a segment so downstream consumers (subtitles, ELAN tiers) get
# readable units. They are a backstop, not the primary rule: a segment that ends on punctuation
# before reaching either cap is unaffected.
MAX_SEGMENT_DURATION = 8.0
MAX_SEGMENT_WORDS = 25


def _to_words(chunks: list[dict]) -> list[AudioTranscriptionWord]:
    """Convert CrisperWhisper's word chunks into words.

    A chunk is ``{"text": word, "timestamp": (start, end)}``. There is no per-word score, so
    confidence stays None.
    """
    return [
        AudioTranscriptionWord(
            word=chunk["text"],
            start=chunk["timestamp"][0],
            end=chunk["timestamp"][1],
        )
        for chunk in chunks
    ]


def _exceeds_caps(words: list[AudioTranscriptionWord]) -> bool:
    """True when a run of words has outgrown either length cap."""
    return len(words) > MAX_SEGMENT_WORDS or (words[-1].end - words[0].start) > MAX_SEGMENT_DURATION


def _split_point(words: list[AudioTranscriptionWord]) -> int:
    """Index to cut an over-long run at: the widest pause between two of its words.

    CrisperWhisper's timings are its selling point, and `adjust_pauses_for_hf_pipeline_output`
    has already normalized inter-word gaps, so the widest gap is the best available guess at a
    breath or clause boundary. Cutting there keeps the break off the middle of a phrase, which a
    fixed-position cut could not promise. Returns the number of words kept in the first part.
    """
    gaps = [(words[i + 1].start - words[i].end, i + 1) for i in range(len(words) - 1)]
    return max(gaps)[1]


def _to_segments(words: list[AudioTranscriptionWord]) -> list[AudioTranscriptionSegment]:
    """Group words into segments, breaking on sentence ends and on the length caps.

    CrisperWhisper returns no segments of its own, so they are synthesized here to give the
    component the same shape as whisperx's. A run is closed when a word ends a sentence, or when
    it has grown past MAX_SEGMENT_DURATION / MAX_SEGMENT_WORDS — the latter cut placed at the run's
    widest pause. A trailing run without terminal punctuation still becomes a segment, so no word
    is dropped.

    Each word is stamped with the index of the segment it lands in, as `words` is mutated in place.
    """
    segments: list[AudioTranscriptionSegment] = []
    current: list[AudioTranscriptionWord] = []
    for word in words:
        current.append(word)

        if word.word.rstrip()[-1:] in SENTENCE_END:
            segments.append(_make_segment(current, len(segments)))
            current = []
        # A single word can exceed the duration cap on its own; there is nothing to split then.
        elif len(current) > 1 and _exceeds_caps(current):
            cut = _split_point(current)
            segments.append(_make_segment(current[:cut], len(segments)))
            # The tail carries over: it may still end on punctuation, or grow into its own cap.
            current = current[cut:]

    if current:
        segments.append(_make_segment(current, len(segments)))

    return segments


def _make_segment(words: list[AudioTranscriptionWord], index: int) -> AudioTranscriptionSegment:
    """Build a segment spanning a run of words, pointing those words back at it."""
    for word in words:
        word.segment_index = index

    return AudioTranscriptionSegment(
        text=" ".join(word.word for word in words),
        start=words[0].start,
        end=words[-1].end,
    )


def _build_total(segments: list[AudioTranscriptionSegment], audio_end: float) -> AudioTranscriptionTotal:
    text = " ".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip())
    return AudioTranscriptionTotal(
        text=text,
        start=segments[0].start if segments else 0.0,
        end=segments[-1].end if segments else audio_end,
    )


def to_audio_transcription(
    raw_tracks: dict, subsequence_start: float, subsequence_length: float, algorithm: str
) -> AudioTranscription:
    """Convert raw per-track CrisperWhisper results into the AudioTranscription model.

    Args:
        raw_tracks: {track_name: pipeline_output} as produced by the HF ASR pipeline, i.e. each
            value carries a "chunks" list of word-level timestamps.
        subsequence_start: Source-recording second this subsequence begins at.
        subsequence_length: Length of the subsequence in seconds.
        algorithm: Detector instance name that produced this output.
    """
    tracks: dict[str, TrackTranscription] = {}
    for track_name, raw in raw_tracks.items():
        words = _to_words(raw.get("chunks", []))
        segments = _to_segments(words)
        tracks[track_name] = TrackTranscription(
            total=_build_total(segments, audio_end=subsequence_length),
            segments=segments,
            words=words,
        )

    meta = JsonMeta(
        component="audio_transcription",
        algorithm=algorithm,
        subsequence_start=subsequence_start,
        tables={"tracks": ["segments", "words"]},
    )
    return AudioTranscription(meta=meta, tracks=tracks)
