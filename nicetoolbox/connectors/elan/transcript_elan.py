"""Conversion between NICE transcription JSON and ELAN tiers (both directions).

Tier naming follows the transcription JSON's own track keys, joined by a "__" delimiter so track
names and speaker labels (both of which contain single underscores) stay unambiguous:

- "<track>"                 segment tier — the sentence-level text (the WER reference)
- "<track>__words"          word tier — one interval per word (opt-in)
- "<track>__speaker"        turn tier — contiguous same-speaker words merged into turns (opt-in)
- "<track>__<speakerlabel>" per-speaker segment tier — one per speaker on a multi-speaker track,
                            holding just that speaker's segments (opt-in). On import these become
                            the authority for segment timings/text (see tiers_to_transcript).
"""

import logging
from collections import Counter, defaultdict

from .elan_data import Interval, Tier

DELIM = "__"
WORDS_ROLE = "words"
SPEAKER_ROLE = "speaker"

_SEGMENTS = "segments"
_WORDS = "words"
_SPEAKER = "speaker"
_SPEAKER_SEGMENTS = "speaker_segments"


# =============================================================================
# JSON -> ELAN tiers (export)
# =============================================================================


def _merge_speaker_turns(words: list[dict]) -> list[Interval]:
    """Collapse runs of consecutive words sharing a speaker into one turn interval."""
    turns: list[Interval] = []
    previous: str | None = None

    for word in words:
        speaker = word.get("speaker")
        if speaker is None:
            previous = None  # a gap ends the current run
            continue
        if turns and previous == speaker:
            turns[-1] = Interval(turns[-1].start_sec, word["end"], speaker)
        else:
            turns.append(Interval(word["start"], word["end"], speaker))
        previous = speaker

    return turns


def _segment_words(segment: dict, words: list[dict]) -> list[dict]:
    """Words belonging to a segment: those whose midpoint falls inside it (an exact partition)."""
    return [w for w in words if segment["start"] <= (w["start"] + w["end"]) / 2 <= segment["end"] and w.get("speaker")]


def _contiguous_speaker_runs(words: list[dict]) -> list[list[dict]]:
    """Split a word list into maximal runs of consecutive words sharing a speaker."""
    runs: list[list[dict]] = []
    for word in words:
        if runs and runs[-1][-1]["speaker"] == word["speaker"]:
            runs[-1].append(word)
        else:
            runs.append([word])
    return runs


def _speaker_segment_tiers(track: str, segments: list[dict], words: list[dict], strategy: str) -> list[Tier]:
    """Build per-speaker segment tiers for a multi-speaker track (empty for single-speaker tracks).

    `dominant` keeps each segment whole under the speaker owning most of its words; `split`
    breaks a segment at speaker changes into separate segments (text rebuilt from the words).
    """
    speakers = {w["speaker"] for w in words if w.get("speaker")}
    if len(speakers) <= 1:
        return []

    per_speaker: dict[str, list[Interval]] = defaultdict(list)
    for seg in segments:
        seg_words = _segment_words(seg, words)
        span = f"[{seg['start']:.3f}s, {seg['end']:.3f}s]"
        if not seg_words:
            logging.warning(
                f"Track '{track}': segment {span} has no speaker-labeled words, skipped in per-speaker tiers."
            )
            continue

        if strategy == "dominant":
            dominant = Counter(w["speaker"] for w in seg_words).most_common(1)[0][0]
            if len({w["speaker"] for w in seg_words}) > 1:
                logging.warning(
                    f"Track '{track}': segment {span} mixes speakers; assigned whole to dominant '{dominant}'."
                )
            per_speaker[dominant].append(Interval(seg["start"], seg["end"], str(seg.get("text", "")).strip()))
        else:  # split
            for run in _contiguous_speaker_runs(seg_words):
                per_speaker[run[0]["speaker"]].append(
                    Interval(run[0]["start"], run[-1]["end"], " ".join(w["word"] for w in run))
                )

    return [
        Tier(f"{track}{DELIM}{speaker}", sorted(ivs, key=lambda iv: iv.start_sec))
        for speaker, ivs in sorted(per_speaker.items())
    ]


def transcript_to_tiers(
    transcript: dict,
    include_words: bool = False,
    include_speaker: bool = False,
    include_speaker_segments: bool = False,
    mixed_segment_strategy: str = "dominant",
) -> list[Tier]:
    """Build ELAN tiers from a transcription JSON payload keyed by track name."""
    tiers: list[Tier] = []

    for track, payload in transcript.items():
        if not isinstance(payload, dict):
            logging.warning(f"Track '{track}' payload is not an object, skipping.")
            continue

        segments = payload.get("segments") or []
        tiers.append(Tier(track, [Interval(s["start"], s["end"], str(s.get("text", "")).strip()) for s in segments]))

        words = payload.get("word_segments") or []
        if include_words and words:
            tiers.append(
                Tier(f"{track}{DELIM}{WORDS_ROLE}", [Interval(w["start"], w["end"], w["word"]) for w in words])
            )
        if include_speaker and words:
            turns = _merge_speaker_turns(words)
            if turns:
                tiers.append(Tier(f"{track}{DELIM}{SPEAKER_ROLE}", turns))
            else:
                logging.warning(f"Track '{track}': no speaker labels on words, no speaker tier written.")
        if include_speaker_segments and words:
            tiers.extend(_speaker_segment_tiers(track, segments, words, mixed_segment_strategy))

    return tiers


# =============================================================================
# ELAN tiers -> JSON (import)
# =============================================================================


def _split_tier_name(tier_name: str) -> tuple[str, str, str | None]:
    """Map a tier name back to its (track, role, speaker_label).

    speaker_label is set only for per-speaker segment tiers; None otherwise.
    """
    if DELIM not in tier_name:
        return tier_name, _SEGMENTS, None
    track, suffix = tier_name.rsplit(DELIM, 1)
    if suffix == WORDS_ROLE:
        return track, _WORDS, None
    if suffix == SPEAKER_ROLE:
        return track, _SPEAKER, None
    return track, _SPEAKER_SEGMENTS, suffix


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _words_of_segment(segment: Interval, words: list[Interval]) -> list[Interval]:
    """Words belonging to a segment: those whose midpoint falls inside it (an exact partition)."""
    return [w for w in words if segment.start_sec <= (w.start_sec + w.end_sec) / 2 <= segment.end_sec]


def _check_segment_word_consistency(track: str, segments: list[Interval], words: list[Interval]) -> None:
    """Fail when a segment's text disagrees with its underlying word tier.

    Both tiers are annotator-editable, so a mismatch means one of them was corrected and the other
    was not — silently trusting either would corrupt the reference text or the word timings.
    """
    for segment in segments:
        expected = _collapse_ws(segment.annotation)
        joined = _collapse_ws(" ".join(w.annotation for w in _words_of_segment(segment, words)))
        if joined != expected:
            raise ValueError(
                f"Track '{track}': segment [{segment.start_sec:.3f}s, {segment.end_sec:.3f}s] text does not "
                f"match its words.\n  segment text: {expected!r}\n  joined words: {joined!r}\n"
                "Correct both tiers consistently in ELAN, or export without the word tier."
            )


def _speaker_for(word: Interval, turns: list[Interval]) -> str | None:
    """Speaker of the turn overlapping this word the most, or None if no turn overlaps."""
    best: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(word.end_sec, turn.end_sec) - max(word.start_sec, turn.start_sec)
        if overlap > best_overlap:
            best_overlap = overlap
            best = turn.annotation
    return best


def _build_total(segments: list[Interval]) -> dict:
    texts = [s.annotation.strip() for s in segments if s.annotation.strip()]
    return {
        "text": " ".join(texts),
        "start": min((s.start_sec for s in segments), default=None),
        "end": max((s.end_sec for s in segments), default=None),
    }


def _word_segments_from(words: list[Interval], speaker_source: list[Interval], track: str) -> list[dict]:
    """Rebuild word_segments with unchanged timings, tagging each word with its overlapping speaker."""
    out: list[dict] = []
    uncovered = 0
    for word in words:
        entry = {"word": word.annotation, "start": word.start_sec, "end": word.end_sec}
        if speaker_source:
            speaker = _speaker_for(word, speaker_source)
            if speaker is None:
                uncovered += 1
            else:
                entry["speaker"] = speaker
        out.append(entry)
    if uncovered:
        logging.warning(f"Track '{track}': {uncovered} word(s) not covered by any speaker interval.")
    return out


def _group_tiers(tiers: list[Tier]) -> dict[str, dict]:
    """Group tiers per track into their roles, collecting per-speaker segment tiers by label."""
    grouped: dict[str, dict] = defaultdict(lambda: {_SPEAKER_SEGMENTS: {}})
    for tier in tiers:
        track, role, speaker = _split_tier_name(tier.tier_name)
        if role == _SPEAKER_SEGMENTS:
            if speaker in grouped[track][_SPEAKER_SEGMENTS]:
                raise ValueError(f"Track '{track}' has more than one per-speaker tier for '{speaker}'.")
            grouped[track][_SPEAKER_SEGMENTS][speaker] = tier
        else:
            if role in grouped[track]:
                raise ValueError(f"Track '{track}' has more than one '{role}' tier ('{tier.tier_name}').")
            grouped[track][role] = tier
    return grouped


def _transcript_from_speaker_segments(track: str, roles: dict) -> dict:
    """Authority path: per-speaker segment tiers drive segments; word timings stay untouched.

    Annotators corrected only the per-speaker segment tiers, so those define segment timings/text.
    The plain segment tier and turn tier are ignored, and no segment<->word consistency is enforced
    (segments were edited independently of the words).
    """
    if _WORDS not in roles:
        raise ValueError(
            f"Track '{track}' has per-speaker segment tiers but no word tier '{track}{DELIM}{WORDS_ROLE}'. "
            "Word timings are carried by the word tier and must be present to rebuild word_segments. "
            "Re-export from ELAN with the word tier included."
        )

    # Each per-speaker interval is one segment (annotation = its text). The speaker label lives in
    # the tier name, so build a parallel speaker-tagged interval list to re-derive word speakers.
    segments: list[Interval] = []
    speaker_source: list[Interval] = []
    for speaker_label, tier in roles[_SPEAKER_SEGMENTS].items():
        for iv in tier.intervals:
            segments.append(iv)
            speaker_source.append(Interval(iv.start_sec, iv.end_sec, speaker_label))
    segments.sort(key=lambda iv: iv.start_sec)
    speaker_source.sort(key=lambda iv: iv.start_sec)

    words = sorted(roles[_WORDS].intervals, key=lambda iv: iv.start_sec)

    return {
        "total": _build_total(segments),
        "segments": [{"start": s.start_sec, "end": s.end_sec, "text": s.annotation} for s in segments],
        "word_segments": _word_segments_from(words, speaker_source, track),
    }


def _transcript_from_plain_segments(track: str, roles: dict) -> dict:
    """Default path: the plain <track> segment tier is authoritative, checked against the words."""
    if _SEGMENTS not in roles:
        raise ValueError(
            f"Track '{track}' has {sorted(k for k in roles if k != _SPEAKER_SEGMENTS)} tier(s) but no segment "
            f"tier named '{track}'. The segment tier carries the reference text and is required."
        )

    segments = sorted(roles[_SEGMENTS].intervals, key=lambda iv: iv.start_sec)
    payload: dict = {
        "total": _build_total(segments),
        # Text is stored verbatim: normalization (casing, punctuation, disfluencies) is the
        # evaluation metric's decision, not the connector's.
        "segments": [{"start": s.start_sec, "end": s.end_sec, "text": s.annotation} for s in segments],
    }

    turns = sorted(roles[_SPEAKER].intervals, key=lambda iv: iv.start_sec) if _SPEAKER in roles else []
    if turns and _WORDS not in roles:
        # Speaker labels ride on word_segments, so without a word tier there is nowhere to put
        # them and the annotator's turn corrections would be silently dropped.
        raise ValueError(
            f"Track '{track}' has a speaker tier but no word tier '{track}{DELIM}{WORDS_ROLE}'. "
            "Speaker labels are stored per word, so a word tier is required to carry them. "
            "Re-export with include_words = true, or drop the speaker tier."
        )

    if _WORDS in roles:
        words = sorted(roles[_WORDS].intervals, key=lambda iv: iv.start_sec)
        _check_segment_word_consistency(track, segments, words)
        payload["word_segments"] = _word_segments_from(words, turns, track)

    return payload


def tiers_to_transcript(tiers: list[Tier]) -> dict:
    """Rebuild a transcription JSON payload from annotator-corrected ELAN tiers."""
    grouped = _group_tiers(tiers)

    out: dict = {}
    for track, roles in grouped.items():
        if roles[_SPEAKER_SEGMENTS]:
            out[track] = _transcript_from_speaker_segments(track, roles)
        else:
            out[track] = _transcript_from_plain_segments(track, roles)

    return out
