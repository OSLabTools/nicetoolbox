"""Tests for transcription <-> ELAN tier conversion."""

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from nicetoolbox.connectors.elan.elan_configs import ElanExportTranscriptionConfig
from nicetoolbox.connectors.elan.elan_data import Interval, Tier
from nicetoolbox.connectors.elan.transcript_elan import (
    _merge_speaker_turns,
    _split_tier_name,
    tiers_to_transcript,
    transcript_to_tiers,
)

# Mirrors the whisperX unified component output shape.
_TRANSCRIPT = {
    "left_mic": {
        "total": {"text": "end of this year", "start": 0.6, "end": 1.2},
        "segments": [{"start": 0.6, "end": 1.2, "text": " end of this year"}],
        "word_segments": [
            {"word": "end", "start": 0.6, "end": 0.7, "speaker": "person_left"},
            {"word": "of", "start": 0.7, "end": 0.8, "speaker": "person_left"},
            {"word": "this", "start": 0.85, "end": 1.0, "speaker": "person_left"},
            {"word": "year", "start": 1.0, "end": 1.2, "speaker": "person_left"},
        ],
    }
}

# A multi-speaker track: two segments, each spoken by a different speaker (the clean room case).
_ROOM = {
    "room": {
        "segments": [
            {"start": 0.6, "end": 2.4, "text": " end of this year, last year."},
            {"start": 2.7, "end": 4.4, "text": "End of last year, okay."},
        ],
        "word_segments": [
            {"word": "end", "start": 0.6, "end": 0.7, "speaker": "SPEAKER_00"},
            {"word": "of", "start": 0.7, "end": 0.8, "speaker": "SPEAKER_00"},
            {"word": "year,", "start": 1.0, "end": 2.4, "speaker": "SPEAKER_00"},
            {"word": "End", "start": 2.7, "end": 2.8, "speaker": "SPEAKER_01"},
            {"word": "okay.", "start": 4.1, "end": 4.4, "speaker": "SPEAKER_01"},
        ],
    }
}


def _tier_names(tiers):
    return [t.tier_name for t in tiers]


def _by_name(tiers):
    return {t.tier_name: t for t in tiers}


# --- transcript_to_tiers ---


def test_defaults_produce_segment_tiers_only():
    tiers = transcript_to_tiers(_TRANSCRIPT)
    assert _tier_names(tiers) == ["left_mic"]
    assert tiers[0].intervals[0].annotation == "end of this year"


def test_include_words_adds_word_tier():
    tiers = transcript_to_tiers(_TRANSCRIPT, include_words=True)
    assert _tier_names(tiers) == ["left_mic", "left_mic__words"]
    assert [iv.annotation for iv in tiers[1].intervals] == ["end", "of", "this", "year"]


def test_include_speaker_adds_turn_tier():
    tiers = transcript_to_tiers(_TRANSCRIPT, include_words=True, include_speaker=True)
    assert _tier_names(tiers) == ["left_mic", "left_mic__words", "left_mic__speaker"]
    turns = tiers[2].intervals
    assert len(turns) == 1  # four contiguous same-speaker words merged into one turn
    assert (turns[0].start_sec, turns[0].end_sec, turns[0].annotation) == (0.6, 1.2, "person_left")


def test_no_speaker_tier_when_words_lack_speaker(caplog):
    transcript = {"room": {"segments": [], "word_segments": [{"word": "hi", "start": 0.0, "end": 0.1}]}}
    with caplog.at_level(logging.WARNING):
        tiers = transcript_to_tiers(transcript, include_words=True, include_speaker=True)
    assert _tier_names(tiers) == ["room", "room__words"]
    assert any("no speaker labels" in r.message for r in caplog.records)


def test_export_config_rejects_speaker_without_words():
    with pytest.raises(ValidationError, match="requires include_words"):
        ElanExportTranscriptionConfig(
            log_level="INFO",
            log_file_path=Path("x.log"),
            include_words=False,
            include_speaker=True,
            run={},
        )


def test_export_config_rejects_speaker_segments_without_words():
    with pytest.raises(ValidationError, match="include_speaker_segments = true requires include_words"):
        ElanExportTranscriptionConfig(
            log_level="INFO",
            log_file_path=Path("x.log"),
            include_words=False,
            include_speaker_segments=True,
            run={},
        )


# --- per-speaker segment tiers (export) ---


class TestSpeakerSegmentTiers:
    def test_multi_speaker_track_gets_one_tier_per_speaker(self):
        tiers = _by_name(transcript_to_tiers(_ROOM, include_words=True, include_speaker_segments=True))
        assert "room__SPEAKER_00" in tiers
        assert "room__SPEAKER_01" in tiers

        s00 = tiers["room__SPEAKER_00"].intervals
        s01 = tiers["room__SPEAKER_01"].intervals
        assert [(iv.start_sec, iv.end_sec, iv.annotation) for iv in s00] == [(0.6, 2.4, "end of this year, last year.")]
        assert [(iv.start_sec, iv.end_sec, iv.annotation) for iv in s01] == [(2.7, 4.4, "End of last year, okay.")]

    def test_single_speaker_track_gets_no_speaker_segment_tier(self):
        tiers = _tier_names(transcript_to_tiers(_TRANSCRIPT, include_words=True, include_speaker_segments=True))
        assert not any("__person_left" in name for name in tiers)
        assert tiers == ["left_mic", "left_mic__words"]

    def test_dominant_keeps_mixed_segment_whole(self, caplog):
        mixed = {
            "room": {
                "segments": [{"start": 0.0, "end": 3.0, "text": "a b c"}],
                "word_segments": [
                    {"word": "a", "start": 0.0, "end": 1.0, "speaker": "S0"},
                    {"word": "b", "start": 1.0, "end": 2.0, "speaker": "S0"},
                    {"word": "c", "start": 2.0, "end": 3.0, "speaker": "S1"},
                ],
            }
        }
        with caplog.at_level(logging.WARNING):
            tiers = _by_name(transcript_to_tiers(mixed, include_words=True, include_speaker_segments=True))
        # Whole segment assigned to the majority speaker S0; S1 gets no tier.
        assert set(tiers) == {"room", "room__words", "room__S0"}
        assert tiers["room__S0"].intervals[0].annotation == "a b c"
        assert any("mixes speakers" in r.message for r in caplog.records)

    def test_split_breaks_mixed_segment_at_speaker_change(self):
        mixed = {
            "room": {
                "segments": [{"start": 0.0, "end": 3.0, "text": "a b c"}],
                "word_segments": [
                    {"word": "a", "start": 0.0, "end": 1.0, "speaker": "S0"},
                    {"word": "b", "start": 1.0, "end": 2.0, "speaker": "S0"},
                    {"word": "c", "start": 2.0, "end": 3.0, "speaker": "S1"},
                ],
            }
        }
        tiers = _by_name(
            transcript_to_tiers(
                mixed, include_words=True, include_speaker_segments=True, mixed_segment_strategy="split"
            )
        )
        assert tiers["room__S0"].intervals[0].annotation == "a b"
        assert (tiers["room__S0"].intervals[0].start_sec, tiers["room__S0"].intervals[0].end_sec) == (0.0, 2.0)
        assert tiers["room__S1"].intervals[0].annotation == "c"
        assert (tiers["room__S1"].intervals[0].start_sec, tiers["room__S1"].intervals[0].end_sec) == (2.0, 3.0)


class TestMergeSpeakerTurns:
    def test_alternating_speakers_split(self):
        words = [
            {"word": "a", "start": 0.0, "end": 1.0, "speaker": "s1"},
            {"word": "b", "start": 1.0, "end": 2.0, "speaker": "s2"},
            {"word": "c", "start": 2.0, "end": 3.0, "speaker": "s1"},
        ]
        turns = _merge_speaker_turns(words)
        assert [(t.start_sec, t.end_sec, t.annotation) for t in turns] == [
            (0.0, 1.0, "s1"),
            (1.0, 2.0, "s2"),
            (2.0, 3.0, "s1"),
        ]

    def test_gap_without_speaker_breaks_the_run(self):
        words = [
            {"word": "a", "start": 0.0, "end": 1.0, "speaker": "s1"},
            {"word": "b", "start": 1.0, "end": 2.0},
            {"word": "c", "start": 2.0, "end": 3.0, "speaker": "s1"},
        ]
        turns = _merge_speaker_turns(words)
        assert [(t.start_sec, t.end_sec) for t in turns] == [(0.0, 1.0), (2.0, 3.0)]


# --- tier name round-trip ---


@pytest.mark.parametrize(
    "tier_name,expected",
    [
        ("left_mic", ("left_mic", "segments", None)),
        ("left_mic__words", ("left_mic", "words", None)),
        ("left_mic__speaker", ("left_mic", "speaker", None)),
        ("room", ("room", "segments", None)),
        # Per-speaker tiers parse with no base tier present, even with underscored labels.
        ("room__SPEAKER_00", ("room", "speaker_segments", "SPEAKER_00")),
        ("left_mic__person_left", ("left_mic", "speaker_segments", "person_left")),
    ],
)
def test_split_tier_name(tier_name, expected):
    assert _split_tier_name(tier_name) == expected


# --- tiers_to_transcript: plain segment path ---


def test_segments_and_recomputed_total():
    tiers = [Tier("room", [Interval(2.0, 3.0, "second part."), Interval(0.5, 1.5, "First part,")])]
    out = tiers_to_transcript(tiers)

    assert out["room"]["segments"] == [
        {"start": 0.5, "end": 1.5, "text": "First part,"},
        {"start": 2.0, "end": 3.0, "text": "second part."},
    ]
    assert out["room"]["total"] == {"text": "First part, second part.", "start": 0.5, "end": 3.0}


def test_text_is_stored_verbatim():
    tiers = [Tier("room", [Interval(0.0, 1.0, "End of this year [UH] last year.")])]
    out = tiers_to_transcript(tiers)
    assert out["room"]["segments"][0]["text"] == "End of this year [UH] last year."


def test_word_tier_builds_word_segments():
    tiers = [
        Tier("left_mic", [Interval(0.0, 2.0, "a b")]),
        Tier("left_mic__words", [Interval(0.0, 0.5, "a"), Interval(1.0, 1.5, "b")]),
    ]
    out = tiers_to_transcript(tiers)
    assert out["left_mic"]["word_segments"] == [
        {"word": "a", "start": 0.0, "end": 0.5},
        {"word": "b", "start": 1.0, "end": 1.5},
    ]


def test_speaker_tier_attached_to_words():
    tiers = [
        Tier("left_mic", [Interval(0.0, 2.0, "a b")]),
        Tier("left_mic__words", [Interval(0.0, 0.5, "a"), Interval(1.0, 1.5, "b")]),
        Tier("left_mic__speaker", [Interval(0.0, 0.9, "s1"), Interval(0.9, 2.0, "s2")]),
    ]
    out = tiers_to_transcript(tiers)["left_mic"]

    assert [w["speaker"] for w in out["word_segments"]] == ["s1", "s2"]
    # Turns are fully derivable by merging contiguous same-speaker words, so the JSON stays
    # strictly whisperX-shaped; standalone turns belong to the audio_diarization component.
    assert set(out) == {"total", "segments", "word_segments"}


def test_speaker_tier_without_word_tier_raises():
    tiers = [
        Tier("left_mic", [Interval(0.0, 2.0, "a b")]),
        Tier("left_mic__speaker", [Interval(0.0, 2.0, "s1")]),
    ]
    with pytest.raises(ValueError, match="speaker tier but no word tier"):
        tiers_to_transcript(tiers)


def test_no_word_tier_skips_consistency_check():
    # Segment text with no word tier is accepted as-is.
    tiers = [Tier("room", [Interval(0.0, 1.0, "anything at all")])]
    assert "word_segments" not in tiers_to_transcript(tiers)["room"]


def test_missing_segment_tier_raises():
    tiers = [Tier("left_mic__words", [Interval(0.0, 0.5, "a")])]
    with pytest.raises(ValueError, match="no segment tier named 'left_mic'"):
        tiers_to_transcript(tiers)


class TestSegmentWordConsistency:
    def test_consistent_pair_passes(self):
        tiers = [
            Tier("left_mic", [Interval(0.0, 2.0, "end of  this")]),  # extra space collapses
            Tier(
                "left_mic__words",
                [Interval(0.0, 0.5, "end"), Interval(0.6, 1.0, "of"), Interval(1.1, 1.5, "this")],
            ),
        ]
        assert tiers_to_transcript(tiers)["left_mic"]["segments"][0]["text"] == "end of  this"

    def test_mismatch_raises_naming_track_and_segment(self):
        tiers = [
            Tier("left_mic", [Interval(0.0, 2.0, "end of that")]),
            Tier("left_mic__words", [Interval(0.0, 0.5, "end"), Interval(0.6, 1.0, "of")]),
        ]
        with pytest.raises(ValueError, match="Track 'left_mic'"):
            tiers_to_transcript(tiers)

    def test_words_partitioned_by_midpoint_across_segments(self):
        # A word straddling the boundary belongs to whichever segment holds its midpoint.
        tiers = [
            Tier("room", [Interval(0.0, 1.0, "a"), Interval(1.0, 2.0, "b")]),
            Tier("room__words", [Interval(0.0, 0.9, "a"), Interval(0.95, 1.4, "b")]),
        ]
        out = tiers_to_transcript(tiers)  # midpoint of "b" is 1.175 -> second segment
        assert len(out["room"]["word_segments"]) == 2


# --- tiers_to_transcript: per-speaker authority path ---


class TestSpeakerSegmentImport:
    def _tiers(self, *, s00_start=0.6):
        # Plain + turn tiers are present but STALE; per-speaker tiers are the corrected authority.
        return [
            Tier("room", [Interval(9.0, 9.9, "STALE, must be ignored")]),
            Tier("room__speaker", [Interval(9.0, 9.9, "STALE")]),
            Tier(
                "room__words",
                [
                    Interval(0.6, 0.7, "end"),
                    Interval(0.7, 0.8, "of"),
                    Interval(2.7, 2.8, "End"),
                    Interval(4.1, 4.4, "okay."),
                ],
            ),
            Tier("room__SPEAKER_00", [Interval(s00_start, 0.9, "end of")]),
            Tier("room__SPEAKER_01", [Interval(2.7, 4.4, "End okay.")]),
        ]

    def test_per_speaker_tiers_drive_segments(self):
        out = tiers_to_transcript(self._tiers())["room"]
        assert out["segments"] == [
            {"start": 0.6, "end": 0.9, "text": "end of"},
            {"start": 2.7, "end": 4.4, "text": "End okay."},
        ]
        assert out["total"]["text"] == "end of End okay."
        assert "STALE" not in out["total"]["text"]

    def test_word_timings_unchanged_and_speakers_rederived(self):
        out = tiers_to_transcript(self._tiers())["room"]
        assert [(w["start"], w["end"]) for w in out["word_segments"]] == [
            (0.6, 0.7),
            (0.7, 0.8),
            (2.7, 2.8),
            (4.1, 4.4),
        ]
        assert [w["speaker"] for w in out["word_segments"]] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]

    def test_edited_segment_boundary_moves_only_segment(self):
        # Annotator drags SPEAKER_00's segment start; words keep their own timings.
        out = tiers_to_transcript(self._tiers(s00_start=0.4))["room"]
        assert out["segments"][0]["start"] == 0.4
        assert out["word_segments"][0]["start"] == 0.6  # unchanged

    def test_segment_word_text_mismatch_is_allowed(self):
        # Segments were edited independently, so the consistency check must NOT fire here.
        tiers = [
            Tier("room__words", [Interval(0.0, 0.5, "hello")]),
            Tier("room__SPEAKER_00", [Interval(0.0, 0.5, "completely different text")]),
        ]
        out = tiers_to_transcript(tiers)["room"]
        assert out["segments"][0]["text"] == "completely different text"

    def test_missing_word_tier_raises(self):
        tiers = [Tier("room__SPEAKER_00", [Interval(0.0, 0.5, "hi")])]
        with pytest.raises(ValueError, match="per-speaker segment tiers but no word tier"):
            tiers_to_transcript(tiers)
