import logging
from collections import defaultdict

from nicetoolbox_core.data.json_schema import (
    AlignedSegment,
    AlignedTranscription,
    AlignedWord,
    AudioTranscription,
    AudioTranscriptionSegment,
    AudioTranscriptionTotal,
    AudioTranscriptionWord,
    JsonMeta,
    SpeakerAlignedTranscription,
    TrackTranscription,
)

from .elan_data import ElanHeader, Interval, Tier

# The two transcription components this connector handles. Both share the {meta, tracks} envelope
# and differ only in whether words and segments carry a speaker.
TranscriptionModel = AudioTranscription | SpeakerAlignedTranscription
TrackModel = TrackTranscription | AlignedTranscription
SegmentModel = AudioTranscriptionSegment | AlignedSegment
WordModel = AudioTranscriptionWord | AlignedWord

# meta.component value identifying the speaker-aligned variant, whose words and segments carry
# speaker labels; the plain variant is the other. Which model a file is validated against is
# decided by this name, so it must be one of these two — see parse_transcription.
SPEAKER_ALIGNED_COMPONENT = "speaker_aligned_transcription"
AUDIO_TRANSCRIPTION_COMPONENT = "audio_transcription"

COMPONENT_MODELS = {
    AUDIO_TRANSCRIPTION_COMPONENT: AudioTranscription,
    SPEAKER_ALIGNED_COMPONENT: SpeakerAlignedTranscription,
}

DELIM = "__"
SEGMENTS_ROLE = "segments"
WORDS_ROLE = "words"

# Tier label for items diarization left unlabeled. Also catches every item of a plain
# audio_transcription, whose words and segments carry no speaker field at all — an expected
# outcome for that component, not a problem, so landing here is not warned about.
UNASSIGNED_SPEAKER = "unassigned"

# Recorded as meta.algorithm on import: the result is hand-corrected ELAN data, not the output of
# the detector the txt was originally exported from.
ELAN_ALGORITHM = "elan"

# speaker_confidence given to every imported segment. On detector output the score measures how far
# diarization and transcription disagree — two independent estimates. An imported segment has no
# such second opinion: its speaker comes from the tier name an annotator chose, which is by
# definition authoritative. Recomputing it from the word tiers would not measure that, since ELAN
# does not tie words to segments — an annotator may retime either freely, and words falling outside
# a segment's span would read as disagreement rather than as an unrelated edit.
ELAN_SPEAKER_CONFIDENCE = 1.0


# =============================================================================
# JSON -> ELAN tiers (export)
# =============================================================================


def parse_transcription(raw: dict) -> TranscriptionModel:
    """Validate a raw component file into the model its `meta.component` names.

    Raises:
        ValueError: When `meta` is absent, carries no `component`, or names a component this
            connector does not handle.
    """
    meta = raw.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(
            "Transcription file has no 'meta' block, so the component it holds cannot be "
            "determined. Expected a component output of the form {'meta': ..., 'tracks': ...}."
        )

    component = meta.get("component")
    if not component:
        raise ValueError(
            f"Transcription file's meta has no 'component', so it cannot be validated. "
            f"Expected one of: {sorted(COMPONENT_MODELS)}."
        )

    if component not in COMPONENT_MODELS:
        raise ValueError(
            f"Transcription file names component '{component}', which this connector does not "
            f"handle. Expected one of: {sorted(COMPONENT_MODELS)}."
        )

    return COMPONENT_MODELS[component].model_validate(raw)


def select_tracks(transcript: TranscriptionModel, tracks: list[str]) -> TranscriptionModel:
    """Narrow a component output to the named tracks, keeping their order in the file.

    The selection must name at least one track — there is no "all tracks" shortcut, so a run never
    silently widens to tracks the config did not ask for. A requested name the file does not carry
    is an error too: quietly handling fewer tracks than asked for would look like the detector had
    missed them.
    """
    if not tracks:
        raise ValueError(
            f"No tracks selected. Name the tracks to handle explicitly; "
            f"this component output carries: {sorted(transcript.tracks)}."
        )

    missing = [name for name in tracks if name not in transcript.tracks]
    if missing:
        raise ValueError(
            f"Requested track(s) {missing} not found in this component output. "
            f"Available tracks: {sorted(transcript.tracks)}."
        )

    kept = {name: payload for name, payload in transcript.tracks.items() if name in set(tracks)}
    return transcript.model_copy(update={"tracks": kept})


def _per_speaker_tiers(track: str, role: str, items: list, text_of) -> list[Tier]:
    """Split items into one tier per speaker: `<track>__<role>__<speaker>`"""
    per_speaker: dict[str, list[Interval]] = defaultdict(list)
    for item in items:
        # None (diarization left it unlabelled) and no speaker field at all (the plain component)
        # both land in the same bucket, so `or` rather than getattr's default.
        speaker = getattr(item, "speaker", None) or UNASSIGNED_SPEAKER
        per_speaker[speaker].append(Interval(item.start, item.end, text_of(item)))

    return [
        Tier(f"{track}{DELIM}{role}{DELIM}{speaker}", sorted(ivs, key=lambda iv: iv.start_sec))
        for speaker, ivs in sorted(per_speaker.items())
    ]


def _shift(tiers: list[Tier], offset: float) -> list[Tier]:
    """Move every interval by offset seconds."""
    return [
        Tier(
            tier.tier_name,
            [Interval(iv.start_sec + offset, iv.end_sec + offset, iv.annotation) for iv in tier.intervals],
        )
        for tier in tiers
    ]


def _is_timed(item: SegmentModel | WordModel) -> bool:
    """True when an item carries both timings, so it can anchor an ELAN interval."""
    return item.start is not None and item.end is not None


def transcript_to_tiers(
    transcript: TranscriptionModel,
    export_segments: bool = True,
    export_words: bool = False,
) -> list[Tier]:
    """Build ELAN tiers from a transcription component output."""
    tiers: list[Tier] = []

    for track, payload in transcript.tracks.items():
        # Timings are Optional in the schema (alignment can fail to place an item), but an ELAN
        # interval needs both ends. Untimed items are dropped rather than crashing the export.
        segments = [s for s in payload.segments if _is_timed(s)]
        words = [w for w in payload.words if _is_timed(w)]

        untimed_segments = len(payload.segments) - len(segments)
        if untimed_segments:
            logging.warning(
                f"Track '{track}': {untimed_segments} segment(s) without timings, omitted from the ELAN tiers."
            )
        untimed_words = len(payload.words) - len(words)
        if untimed_words:
            logging.warning(f"Track '{track}': {untimed_words} word(s) without timings, omitted from the ELAN tiers.")

        if export_segments:
            tiers.extend(_per_speaker_tiers(track, SEGMENTS_ROLE, segments, lambda s: s.text.strip()))
        if export_words:
            tiers.extend(_per_speaker_tiers(track, WORDS_ROLE, words, lambda w: w.word))

    # Applied once to the finished tiers rather than at each Interval construction above, so every
    # tier kind (segments, words) is guaranteed the same shift.
    offset = transcript.meta.subsequence_start
    logging.info(f"Shifting tiers by subsequence_start = {offset:.3f}s onto the recording timeline.")
    tiers = _shift(tiers, offset)

    return tiers


# =============================================================================
# ELAN tiers -> JSON (import)
# =============================================================================


def _split_tier_name(tier_name: str) -> tuple[str, str, str]:
    """Map a tier name back to its (track, role, speaker).

    Tier names are `<track>__<role>__<speaker>`, and the track itself may contain the delimiter,
    so the role and speaker are taken from the right.
    """
    parts = tier_name.rsplit(DELIM, 2)
    if len(parts) != 3 or parts[1] not in (SEGMENTS_ROLE, WORDS_ROLE):
        raise ValueError(
            f"Tier '{tier_name}' does not follow the '<track>{DELIM}<role>{DELIM}<speaker>' naming, "
            f"where role is '{SEGMENTS_ROLE}' or '{WORDS_ROLE}'. It cannot be imported."
        )
    return parts[0], parts[1], parts[2]


def _group_tiers(tiers: list[Tier], roles_wanted: set[str]) -> dict[str, dict[str, list[tuple[Interval, str]]]]:
    """Group intervals per track and role, tagging each with the speaker from its tier name.

    Per-speaker tiers of the same role are merged back into one time-ordered list, since the split
    exists only to make labelling in ELAN easier — the payload keeps a single list per role. Roles
    outside `roles_wanted` are dropped here, so everything downstream sees one consistent view.
    """
    grouped: dict[str, dict[str, list[tuple[Interval, str]]]] = defaultdict(lambda: {SEGMENTS_ROLE: [], WORDS_ROLE: []})
    for tier in tiers:
        track, role, speaker = _split_tier_name(tier.tier_name)
        if role not in roles_wanted:
            continue
        grouped[track][role].extend((interval, speaker) for interval in tier.intervals)

    for roles in grouped.values():
        for items in roles.values():
            items.sort(key=lambda pair: pair[0].start_sec)
    return grouped


def _speaker_or_none(speaker: str) -> str | None:
    """The tier's speaker label, or None for the bucket holding unlabelled items."""
    return None if speaker == UNASSIGNED_SPEAKER else speaker


def _segment_index_of(word: Interval, segments: list[tuple[Interval, str]]) -> int | None:
    """Position of the segment a word belongs to, or None when no segment covers it.

    ELAN tiers are flat and independently editable, so the grouping the detector recorded is not
    carried by the file and has to be rebuilt. A word belongs to the segment containing its
    midpoint: that is an exact partition, so a word straddling a boundary lands in one segment
    rather than both. A word in a gap between segments belongs to none — hence the None, which is
    a real outcome here rather than an error.
    """
    midpoint = (word.start_sec + word.end_sec) / 2
    for index, (segment, _) in enumerate(segments):
        if segment.start_sec <= midpoint <= segment.end_sec:
            return index
    return None


def _build_total(segments: list[tuple[Interval, str]]) -> AudioTranscriptionTotal:
    texts = [iv.annotation.strip() for iv, _ in segments if iv.annotation.strip()]
    return AudioTranscriptionTotal(
        text=" ".join(texts),
        # The model requires both bounds; an empty tier has none, so it collapses to an empty span.
        start=min((iv.start_sec for iv, _ in segments), default=0.0),
        end=max((iv.end_sec for iv, _ in segments), default=0.0),
    )


def _track_from_tiers(roles: dict[str, list[tuple[Interval, str]]], aligned: bool) -> TrackModel:
    """Rebuild one track's payload from its grouped segment and word intervals.

    Text and timings are taken verbatim: normalization (casing, punctuation, disfluencies) is the
    evaluation metric's decision, not the connector's. A role with no tiers becomes an empty list,
    since the schema requires both fields.
    """
    if aligned:
        words = [
            AlignedWord(
                word=iv.annotation,
                segment_index=_segment_index_of(iv, roles[SEGMENTS_ROLE]),
                start=iv.start_sec,
                end=iv.end_sec,
                speaker=_speaker_or_none(speaker),
            )
            for iv, speaker in roles[WORDS_ROLE]
        ]
        segments = [
            AlignedSegment(
                text=iv.annotation,
                start=iv.start_sec,
                end=iv.end_sec,
                speaker=_speaker_or_none(speaker),
                speaker_confidence=ELAN_SPEAKER_CONFIDENCE,
            )
            for iv, speaker in roles[SEGMENTS_ROLE]
        ]
        return AlignedTranscription(total=_build_total(roles[SEGMENTS_ROLE]), segments=segments, words=words)

    return TrackTranscription(
        total=_build_total(roles[SEGMENTS_ROLE]),
        segments=[
            AudioTranscriptionSegment(text=iv.annotation, start=iv.start_sec, end=iv.end_sec)
            for iv, _ in roles[SEGMENTS_ROLE]
        ],
        words=[
            AudioTranscriptionWord(
                word=iv.annotation,
                segment_index=_segment_index_of(iv, roles[SEGMENTS_ROLE]),
                start=iv.start_sec,
                end=iv.end_sec,
            )
            for iv, _ in roles[WORDS_ROLE]
        ],
    )


def _infer_component(grouped: dict[str, dict[str, list[tuple[Interval, str]]]]) -> str:
    """Decide which component a set of tiers holds, from whether anything carries a speaker.

    A file whose every tier is the `unassigned` bucket came from a plain audio_transcription, which
    has no speaker field at all. Anything with a real label is speaker-aligned. The one ambiguous
    case — a speaker-aligned file where diarization failed on every item — is read as plain, which
    is the honest reading of a file that carries no speaker information.
    """
    labelled = any(
        speaker != UNASSIGNED_SPEAKER for roles in grouped.values() for items in roles.values() for _, speaker in items
    )
    return SPEAKER_ALIGNED_COMPONENT if labelled else AUDIO_TRANSCRIPTION_COMPONENT


def tiers_to_transcript(
    tiers: list[Tier],
    header: ElanHeader,
    algorithm: str = ELAN_ALGORITHM,
    import_segments: bool = True,
    import_words: bool = True,
) -> TranscriptionModel:
    """Rebuild a transcription component output from annotator-corrected ELAN tiers.

    The meta block is derived from the tiers plus ELAN's own media header, so no reference to the
    detector output the txt was exported from is needed:

    - `component` follows whether any tier carries a real speaker label (see _infer_component)
    - `tables` is fixed by the payload shape
    - `subsequence_start` is 0: the export shifted these onto the recording timeline and they stay
      there, so imported annotation is recording-absolute like NPZ ground truth. Placing a
      subsequence back on the timeline is the consumer's job, and these are already on it.
    - `subsequence_length` is the loaded media's duration, which pairs with the 0 start to say
      "spans the whole recording"
    - `algorithm` records that this is hand-corrected ELAN data, not a detector's output

    Args:
        tiers: The parsed tiers of a corrected ELAN txt.
        header: The txt's ELAN header, whose media line carries the duration of the media the
            annotator loaded. Required: the tiers alone only evidence the last annotated moment,
            which silently under-reports trailing silence, and a span guessed from them would be
            indistinguishable from a real one downstream.
        algorithm: Value recorded as `meta.algorithm`.
        import_segments: Read the `segments` tiers. When False the payload's segments are empty and
            every word is imported without a segment_index, there being nothing to point at.
        import_words: Read the `words` tiers.
    """
    roles_wanted = {role for role, wanted in ((SEGMENTS_ROLE, import_segments), (WORDS_ROLE, import_words)) if wanted}
    grouped = _group_tiers(tiers, roles_wanted)
    component = _infer_component(grouped)
    aligned = component == SPEAKER_ALIGNED_COMPONENT

    tracks = {}
    for track, roles in grouped.items():
        payload = _track_from_tiers(roles, aligned)
        # A word in a gap between segments is kept, but it is worth saying so: on detector output
        # every word sits inside a segment, so strays mean the two tiers were retimed apart. Only
        # meaningful when both families were read — without segments there is nothing to fall in.
        stray = sum(1 for word in payload.words if word.segment_index is None)
        if stray and import_segments and import_words:
            logging.warning(
                f"Track '{track}': {stray} of {len(payload.words)} word(s) fall outside every segment "
                "and are imported without a segment_index."
            )
        tracks[track] = payload
    meta = JsonMeta(
        component=component,
        algorithm=algorithm,
        subsequence_start=0.0,  # ELAN export from begging of the file
        subsequence_length=header.duration_ms / 1000.0,  # until the end of the file
        tables={"tracks": [SEGMENTS_ROLE, "words"]},
    )
    return COMPONENT_MODELS[component](meta=meta, tracks=tracks)
