"""Tests for transcription <-> ELAN tier conversion."""

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from nicetoolbox.connectors.elan.elan_configs import ElanExportTranscriptionConfig, ElanImportTranscriptionConfig
from nicetoolbox.connectors.elan.elan_data import Interval, Tier
from nicetoolbox.connectors.elan.transcript_elan import (
    _split_tier_name,
    parse_transcription,
    select_tracks,
    tiers_to_transcript,
    transcript_to_tiers,
)
from nicetoolbox_core.data.json_schema import JsonMeta

# The meta block of a component output. ELAN carries no provenance, so the import copies this
# through from the source file rather than rebuilding it.
_META = JsonMeta(
    component="speaker_aligned_transcription",
    algorithm="whisperx",
    subsequence_start=12.5,
    tables={"tracks": ["segments", "words"]},
)

# A source anchored at 0, for export cases where the shift would otherwise obscure the assertion.
_META_S0 = _META.model_copy(update={"subsequence_start": 0.0})

# The plain component: its words carry no speaker field at all.
_META_PLAIN = _META_S0.model_copy(update={"component": "audio_transcription"})

# Mirrors the whisperX unified component output shape.
_TRANSCRIPT_TRACKS = {
    "left_mic": {
        "total": {"text": "end of this year", "start": 0.6, "end": 1.2},
        "segments": [
            {"start": 0.6, "end": 1.2, "text": " end of this year", "speaker": "person_left", "speaker_confidence": 1.0}
        ],
        "words": [
            {"word": "end", "start": 0.6, "end": 0.7, "speaker": "person_left"},
            {"word": "of", "start": 0.7, "end": 0.8, "speaker": "person_left"},
            {"word": "this", "start": 0.85, "end": 1.0, "speaker": "person_left"},
            {"word": "year", "start": 1.0, "end": 1.2, "speaker": "person_left"},
        ],
    }
}

# A multi-speaker track: two segments, each spoken by a different speaker (the clean room case).
_ROOM_TRACKS = {
    "room": {
        "total": {"text": "end of this year, last year. End of last year, okay.", "start": 0.6, "end": 4.4},
        "segments": [
            {
                "start": 0.6,
                "end": 2.4,
                "text": " end of this year, last year.",
                "speaker": "SPEAKER_00",
                "speaker_confidence": 1.0,
            },
            {
                "start": 2.7,
                "end": 4.4,
                "text": "End of last year, okay.",
                "speaker": "SPEAKER_01",
                "speaker_confidence": 1.0,
            },
        ],
        "words": [
            {"word": "end", "start": 0.6, "end": 0.7, "speaker": "SPEAKER_00"},
            {"word": "of", "start": 0.7, "end": 0.8, "speaker": "SPEAKER_00"},
            {"word": "year,", "start": 1.0, "end": 2.4, "speaker": "SPEAKER_00"},
            {"word": "End", "start": 2.7, "end": 2.8, "speaker": "SPEAKER_01"},
            {"word": "okay.", "start": 4.1, "end": 4.4, "speaker": "SPEAKER_01"},
        ],
    }
}


def _envelope(tracks: dict, meta: JsonMeta | None = None):
    """Wrap per-track payloads in the envelope and validate, as the export task does."""
    return parse_transcription({"meta": (meta or _META_S0).model_dump(), "tracks": tracks})


_TRANSCRIPT = _envelope(_TRANSCRIPT_TRACKS)
_ROOM = _envelope(_ROOM_TRACKS)


def _tier_names(tiers):
    return [t.tier_name for t in tiers]


def _by_name(tiers):
    return {t.tier_name: t for t in tiers}


# --- transcript_to_tiers ---


def test_defaults_produce_segment_tiers_only():
    tiers = transcript_to_tiers(_TRANSCRIPT)
    assert _tier_names(tiers) == ["left_mic__segments__person_left"]
    assert tiers[0].intervals[0].annotation == "end of this year"


def test_export_words_adds_per_speaker_word_tiers():
    tiers = transcript_to_tiers(_TRANSCRIPT, export_words=True)
    assert _tier_names(tiers) == ["left_mic__segments__person_left", "left_mic__words__person_left"]
    assert [iv.annotation for iv in tiers[1].intervals] == ["end", "of", "this", "year"]


def test_families_are_independent():
    # Each flag controls exactly one family.
    assert _tier_names(transcript_to_tiers(_TRANSCRIPT, export_segments=False, export_words=True)) == [
        "left_mic__words__person_left"
    ]
    assert _tier_names(transcript_to_tiers(_TRANSCRIPT, export_words=False)) == ["left_mic__segments__person_left"]


def test_multi_speaker_track_gets_one_tier_per_speaker():
    tiers = _by_name(transcript_to_tiers(_ROOM, export_words=True))
    assert set(tiers) == {
        "room__segments__SPEAKER_00",
        "room__segments__SPEAKER_01",
        "room__words__SPEAKER_00",
        "room__words__SPEAKER_01",
    }
    s00 = tiers["room__segments__SPEAKER_00"].intervals
    s01 = tiers["room__segments__SPEAKER_01"].intervals
    assert [(iv.start_sec, iv.end_sec, iv.annotation) for iv in s00] == [(0.6, 2.4, "end of this year, last year.")]
    assert [(iv.start_sec, iv.end_sec, iv.annotation) for iv in s01] == [(2.7, 4.4, "End of last year, okay.")]


def test_segments_are_placed_by_their_own_speaker():
    # A segment whose words disagree with it stays whole under segment.speaker; speaker_confidence
    # already records the disagreement, so the connector does not re-adjudicate it.
    mixed = _envelope(
        {
            "room": {
                "total": {"text": "a b c", "start": 0.0, "end": 3.0},
                "segments": [{"start": 0.0, "end": 3.0, "text": "a b c", "speaker": "S0", "speaker_confidence": 0.67}],
                "words": [
                    {"word": "a", "start": 0.0, "end": 1.0, "speaker": "S0"},
                    {"word": "b", "start": 1.0, "end": 2.0, "speaker": "S0"},
                    {"word": "c", "start": 2.0, "end": 3.0, "speaker": "S1"},
                ],
            }
        }
    )
    tiers = _by_name(transcript_to_tiers(mixed, export_words=True))
    # The segment goes wholly to S0; its words still split across both speakers' word tiers.
    assert tiers["room__segments__S0"].intervals[0].annotation == "a b c"
    assert "room__segments__S1" not in tiers
    assert [iv.annotation for iv in tiers["room__words__S0"].intervals] == ["a", "b"]
    assert [iv.annotation for iv in tiers["room__words__S1"].intervals] == ["c"]


def test_unlabeled_items_collect_under_unassigned():
    partial = _envelope(
        {
            "room": {
                "total": {"text": "a b", "start": 0.0, "end": 2.0},
                "segments": [{"start": 0.0, "end": 2.0, "text": "a b", "speaker": None, "speaker_confidence": 0.0}],
                "words": [
                    {"word": "a", "start": 0.0, "end": 1.0, "speaker": "S0"},
                    {"word": "b", "start": 1.0, "end": 2.0, "speaker": None},
                ],
            }
        }
    )
    tiers = _by_name(transcript_to_tiers(partial, export_words=True))

    assert set(tiers) == {"room__segments__unassigned", "room__words__S0", "room__words__unassigned"}
    assert [iv.annotation for iv in tiers["room__words__unassigned"].intervals] == ["b"]


def test_plain_component_puts_everything_under_unassigned(caplog):
    # AudioTranscription words and segments have no speaker field at all.
    plain = _envelope(
        {
            "room": {
                "total": {"text": "hi", "start": 0.0, "end": 0.1},
                "segments": [{"start": 0.0, "end": 0.1, "text": "hi"}],
                "words": [{"word": "hi", "start": 0.0, "end": 0.1}],
            }
        },
        _META_PLAIN,
    )
    with caplog.at_level(logging.WARNING):
        tiers = transcript_to_tiers(plain, export_words=True)

    # Landing in `unassigned` is expected for this component, so it is not warned about.
    assert _tier_names(tiers) == ["room__segments__unassigned", "room__words__unassigned"]
    assert not caplog.records


def test_tiers_are_shifted_onto_the_recording_timeline():
    # Payload timestamps are subsequence-relative; ELAN aligns against the whole recording, so a
    # subsequence starting at 12.5s must export tiers at 12.5s + their relative time.
    tiers = transcript_to_tiers(_envelope(_TRANSCRIPT_TRACKS, _META), export_words=True)

    segment = tiers[0].intervals[0]
    assert (segment.start_sec, segment.end_sec) == (13.1, 13.7)  # 0.6/1.2 + 12.5
    # Both tier kinds shift, not just the segments.
    assert [(iv.start_sec, iv.end_sec) for iv in tiers[1].intervals][0] == (13.1, 13.2)


def test_zero_subsequence_start_leaves_times_untouched():
    tiers = transcript_to_tiers(_TRANSCRIPT)
    assert (tiers[0].intervals[0].start_sec, tiers[0].intervals[0].end_sec) == (0.6, 1.2)


def test_parse_transcription_requires_a_meta_block():
    with pytest.raises(ValueError, match="no 'meta' block"):
        parse_transcription({"tracks": _TRANSCRIPT_TRACKS})


def test_parse_transcription_requires_a_component():
    with pytest.raises(ValueError, match="no 'component'"):
        parse_transcription({"meta": {"algorithm": "whisperx"}, "tracks": _TRANSCRIPT_TRACKS})


def test_parse_transcription_rejects_unknown_component():
    # Falling back to the plain model would parse this file with every speaker label dropped.
    raw = {"meta": {**_META_S0.model_dump(), "component": "audio_diarization"}, "tracks": _TRANSCRIPT_TRACKS}
    with pytest.raises(ValueError, match="does not "):
        parse_transcription(raw)


def test_tiers_to_transcript_rejects_unknown_component():
    meta = _META_S0.model_copy(update={"component": "audio_diarization"})
    with pytest.raises(ValueError, match="does not "):
        tiers_to_transcript([Tier("room", [Interval(0.0, 1.0, "hi")])], meta)


def test_select_tracks_keeps_only_the_named_tracks():
    both = _envelope({**_TRANSCRIPT_TRACKS, **_ROOM_TRACKS})
    assert list(select_tracks(both, ["room"]).tracks) == ["room"]
    # File order is preserved, not the order the names were requested in.
    assert list(select_tracks(both, ["room", "left_mic"]).tracks) == ["left_mic", "room"]


def test_select_tracks_rejects_empty_selection():
    # There is no "all tracks" shortcut: a run must never widen beyond what the config names.
    with pytest.raises(ValueError, match="No tracks selected"):
        select_tracks(_TRANSCRIPT, [])


def test_sequence_config_rejects_empty_tracks():
    with pytest.raises(ValidationError, match="tracks"):
        ElanExportTranscriptionConfig(
            log_level="INFO",
            log_file_path=Path("x.log"),
            export_segments=True,
            export_words=False,
            run={"s": {"input": "a.json", "output": "b.txt", "tracks": []}},
        )


def test_select_tracks_rejects_unknown_name():
    # A typo must fail rather than quietly export fewer tracks than asked for.
    with pytest.raises(ValueError, match=r"\['left'\] not found"):
        select_tracks(_TRANSCRIPT, ["left"])


def test_export_config_rejects_all_families_disabled():
    with pytest.raises(ValidationError, match="At least one of"):
        ElanExportTranscriptionConfig(
            log_level="INFO",
            log_file_path=Path("x.log"),
            export_segments=False,
            export_words=False,
            run={},
        )


# --- tier name parsing ---


@pytest.mark.parametrize(
    "tier_name,expected",
    [
        ("left_mic__segments__person_left", ("left_mic", "segments", "person_left")),
        ("left_mic__words__person_left", ("left_mic", "words", "person_left")),
        ("room__segments__unassigned", ("room", "segments", "unassigned")),
        # Track names may contain the delimiter; role and speaker are taken from the right.
        ("a__b__words__SPEAKER_00", ("a__b", "words", "SPEAKER_00")),
        # As may speaker labels, since only the last two fields are fixed.
        ("room__words__SPEAKER_00", ("room", "words", "SPEAKER_00")),
    ],
)
def test_split_tier_name(tier_name, expected):
    assert _split_tier_name(tier_name) == expected


@pytest.mark.parametrize("tier_name", ["room", "room__words", "room__turns__S0"])
def test_split_tier_name_rejects_foreign_names(tier_name):
    with pytest.raises(ValueError, match="does not follow"):
        _split_tier_name(tier_name)


# --- tiers_to_transcript ---


def _tiers(*specs):
    """Build tiers from (name, [(start, end, text), ...]) pairs."""
    return [Tier(name, [Interval(*iv) for iv in intervals]) for name, intervals in specs]


def test_meta_is_derived_entirely_from_the_tiers():
    tiers = _tiers(("room__segments__S0", [(1.0, 2.0, "hello there")]))
    out = tiers_to_transcript(tiers)

    assert out.meta.component == "speaker_aligned_transcription"
    assert out.meta.algorithm == "elan"
    # Imported annotation is recording-absolute, so it is anchored at 0 and times are unshifted.
    assert out.meta.subsequence_start == 0.0
    assert (out.tracks["room"].segments[0].start, out.tracks["room"].segments[0].end) == (1.0, 2.0)
    assert out.meta.tables == {"tracks": ["segments", "words"]}


def test_component_is_plain_when_nothing_carries_a_speaker():
    tiers = _tiers(("room__segments__unassigned", [(0.0, 1.0, "hi")]))
    out = tiers_to_transcript(tiers)

    assert out.meta.component == "audio_transcription"
    # The plain word/segment types have no speaker field at all.
    assert not hasattr(out.tracks["room"].segments[0], "speaker")


def test_algorithm_can_be_overridden():
    tiers = _tiers(("room__segments__unassigned", [(0.0, 1.0, "hi")]))
    assert tiers_to_transcript(tiers, algorithm="manual").meta.algorithm == "manual"


def test_per_speaker_tiers_merge_into_one_time_ordered_list():
    # The per-speaker split exists for labelling convenience; the payload keeps one list per role.
    tiers = _tiers(
        ("room__segments__S1", [(2.0, 3.0, "second")]),
        ("room__segments__S0", [(0.0, 1.0, "first")]),
    )
    out = tiers_to_transcript(tiers)

    segments = out.tracks["room"].segments
    assert [(s.start, s.text, s.speaker) for s in segments] == [(0.0, "first", "S0"), (2.0, "second", "S1")]


def test_unassigned_tier_becomes_a_null_speaker():
    tiers = _tiers(
        ("room__words__S0", [(0.0, 1.0, "a")]),
        ("room__words__unassigned", [(1.0, 2.0, "b")]),
    )
    out = tiers_to_transcript(tiers)

    assert [(w.word, w.speaker) for w in out.tracks["room"].words] == [("a", "S0"), ("b", None)]


@pytest.mark.parametrize(
    "extra_tiers",
    [
        # Words disagreeing with the segment's speaker.
        [("room__words__S0", [(0.0, 1.0, "a")]), ("room__words__S1", [(2.0, 3.0, "c")])],
        # Words retimed outside the segment's span entirely.
        [("room__words__S0", [(10.0, 11.0, "x")])],
        # No word tier at all.
        [],
    ],
    ids=["mixed speakers", "words outside segment", "no words"],
)
def test_imported_segments_are_fully_confident(extra_tiers):
    # An imported segment's speaker comes from the tier name an annotator chose, so there is no
    # second estimate to disagree with it. The word tiers are not consulted: ELAN does not tie
    # words to segments, so their timings say nothing about the segment's speaker.
    tiers = _tiers(("room__segments__S0", [(0.0, 3.0, "a b c")]), *extra_tiers)
    assert tiers_to_transcript(tiers).tracks["room"].segments[0].speaker_confidence == 1.0


def test_total_is_recomputed_from_the_segments():
    tiers = _tiers(("room__segments__S0", [(2.0, 3.0, "second part."), (0.5, 1.5, "First part,")]))
    total = tiers_to_transcript(tiers).tracks["room"].total

    assert (total.text, total.start, total.end) == ("First part, second part.", 0.5, 3.0)


def test_text_is_stored_verbatim():
    tiers = _tiers(("room__segments__S0", [(0.0, 1.0, "End of this year [UH] last year.")]))
    out = tiers_to_transcript(tiers)
    assert out.tracks["room"].segments[0].text == "End of this year [UH] last year."


def test_missing_role_becomes_an_empty_list():
    # The schema requires both fields, so a file with only segments still validates.
    tiers = _tiers(("room__segments__unassigned", [(0.0, 1.0, "hi")]))
    assert tiers_to_transcript(tiers).tracks["room"].words == []


def test_import_flags_select_tier_families():
    tiers = _tiers(
        ("room__segments__S0", [(0.0, 3.0, "a b")]),
        ("room__words__S0", [(0.0, 1.0, "a"), (2.0, 3.0, "b")]),
    )

    both = tiers_to_transcript(tiers).tracks["room"]
    assert (len(both.segments), len(both.words)) == (1, 2)

    segments_only = tiers_to_transcript(tiers, import_words=False).tracks["room"]
    assert (len(segments_only.segments), len(segments_only.words)) == (1, 0)

    # Without segments there is nothing for a word to point at, so every index is null.
    words_only = tiers_to_transcript(tiers, import_segments=False).tracks["room"]
    assert (len(words_only.segments), len(words_only.words)) == (0, 2)
    assert [w.segment_index for w in words_only.words] == [None, None]


def test_import_config_rejects_all_families_disabled():
    with pytest.raises(ValidationError, match="At least one of"):
        ElanImportTranscriptionConfig(
            log_level="INFO",
            log_file_path=Path("x.log"),
            export_srt=False,
            export_csv=False,
            import_segments=False,
            import_words=False,
            run={},
        )


def test_multiple_tracks_are_kept_separate():
    tiers = _tiers(
        ("left_mic__segments__unassigned", [(0.0, 1.0, "left")]),
        ("right_mic__segments__unassigned", [(0.0, 1.0, "right")]),
    )
    out = tiers_to_transcript(tiers)

    assert sorted(out.tracks) == ["left_mic", "right_mic"]
    assert out.tracks["left_mic"].segments[0].text == "left"
