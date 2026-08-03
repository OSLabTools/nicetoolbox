"""Tests for the ELAN txt writer and its round-trip with the parser."""

import logging

import pytest

from nicetoolbox.connectors.elan.elan_configs import GAZE_9COL, TRANSCRIPTION_4COL, ElanColumnSpec
from nicetoolbox.connectors.elan.elan_data import Interval, Tier
from nicetoolbox.connectors.elan.elan_parser import parse_elan_file, parse_tiers
from nicetoolbox.connectors.elan.elan_time import format_timecode, parse_time_cell
from nicetoolbox.connectors.elan.elan_writer import write_elan_txt
from nicetoolbox.connectors.elan.transcript_elan import parse_transcription, tiers_to_transcript, transcript_to_tiers
from nicetoolbox_core.data.json_schema import JsonMeta

_TIERS = [Tier("left_mic", [Interval(0.573, 2.443, "end of this year"), Interval(2.649, 4.412, "okay.")])]


# --- time cells ---


@pytest.mark.parametrize(
    "cell,expected",
    [("2.443", 2.443), ("00:00:02.443", 2.443), ("0.0", 0.0), ("00:01:00.000", 60.0), (" 1.5 ", 1.5)],
)
def test_parse_time_cell_accepts_seconds_and_timecode(cell, expected):
    assert parse_time_cell(cell) == pytest.approx(expected)


def test_parse_time_cell_rejects_garbage():
    with pytest.raises(ValueError):
        parse_time_cell("not-a-time")


def test_format_timecode():
    assert format_timecode(3661.5) == "01:01:01.500"
    assert format_timecode(0.573) == "00:00:00.573"


# --- writer / parser round-trip ---


@pytest.mark.parametrize("spec", [GAZE_9COL, TRANSCRIPTION_4COL])
def test_round_trip_through_parser(tmp_path, spec):
    path = tmp_path / "out.txt"
    write_elan_txt(path, _TIERS, spec)

    parsed = parse_elan_file(path, spec)
    assert len(parsed.tiers) == 1
    assert parsed.tiers[0].tier_name == "left_mic"
    got = [(iv.start_sec, iv.end_sec, iv.annotation) for iv in parsed.tiers[0].intervals]
    assert got == [(0.573, 2.443, "end of this year"), (2.649, 4.412, "okay.")]


def test_writer_emits_expected_column_count(tmp_path):
    path = tmp_path / "out.txt"
    write_elan_txt(path, _TIERS, TRANSCRIPTION_4COL)
    first = path.read_text().splitlines()[0]
    # No derived duration column: it is just end - start.
    assert first.split("\t") == ["left_mic", "00:00:00.573", "00:00:02.443", "end of this year"]


def test_empty_annotation_survives_round_trip(tmp_path):
    path = tmp_path / "out.txt"
    write_elan_txt(path, [Tier("room", [Interval(0.0, 1.0, "")])], TRANSCRIPTION_4COL)
    parsed = parse_elan_file(path, TRANSCRIPTION_4COL)
    assert parsed.tiers[0].intervals[0].annotation == ""


def test_tab_in_annotation_is_sanitized_and_warned(tmp_path, caplog):
    path = tmp_path / "out.txt"
    with caplog.at_level(logging.WARNING):
        write_elan_txt(path, [Tier("room", [Interval(0.0, 1.0, "has\ta tab")])], TRANSCRIPTION_4COL)
    assert any("tab/newline" in r.message for r in caplog.records)
    assert parse_elan_file(path, TRANSCRIPTION_4COL).tiers[0].intervals[0].annotation == "has a tab"


def test_writer_emits_no_media_header(tmp_path):
    """Every line must be a data row: single-field header rows make ELAN's import dialog infer too
    few columns, leaving the annotation column unselectable."""
    path = tmp_path / "out.txt"
    write_elan_txt(path, _TIERS, TRANSCRIPTION_4COL)

    lines = path.read_text().splitlines()
    assert not any(line.startswith('"#file:///') for line in lines)
    assert {len(line.split("\t")) for line in lines} == {TRANSCRIPTION_4COL.width}


# --- import still accepts headered files coming back from ELAN ---


_VIDEO_HEADER = (
    '"#file:///data/view_top.mp4 -- offset: 0, duration: 00:00:05.000 / 5.000 / 5000, ms per sample: 33.333"'
)
_AUDIO_HEADER = '"#file:///data/audio.wav -- offset: 0, duration: 00:00:05.000 / 5.000 / 5000"'
_DATA_ROW = "room\t00:00:00.500\t00:00:01.500\thello"


def _write(path, *lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_import_accepts_header_with_ms_per_sample(tmp_path):
    path = _write(tmp_path / "in.txt", _VIDEO_HEADER, "", _DATA_ROW)
    parsed = parse_elan_file(path, TRANSCRIPTION_4COL)

    assert parsed.header is not None
    assert parsed.header.duration_ms == 5000
    assert parsed.header.ms_per_sample == pytest.approx(33.333)
    assert parsed.tiers[0].intervals[0].annotation == "hello"


def test_import_accepts_audio_header_without_ms_per_sample(tmp_path):
    """ELAN omits 'ms per sample' for audio-only media; real exports mix both forms."""
    path = _write(tmp_path / "in.txt", _VIDEO_HEADER, _AUDIO_HEADER, "", _DATA_ROW)
    parsed = parse_elan_file(path, TRANSCRIPTION_4COL)

    assert parsed.header is not None
    assert len(parsed.header.media_files) == 2
    assert parsed.header.ms_per_sample == pytest.approx(33.333)


def test_import_accepts_audio_only_header(tmp_path):
    path = _write(tmp_path / "in.txt", _AUDIO_HEADER, "", _DATA_ROW)
    parsed = parse_elan_file(path, TRANSCRIPTION_4COL)

    assert parsed.header is not None
    assert parsed.header.ms_per_sample is None
    assert parsed.tiers[0].intervals[0].annotation == "hello"


def test_import_accepts_headerless_file(tmp_path):
    """Our own exports are headerless and must import back unchanged."""
    path = _write(tmp_path / "in.txt", _DATA_ROW)
    parsed = parse_elan_file(path, TRANSCRIPTION_4COL)

    assert parsed.header is None
    assert parsed.tiers[0].intervals[0].annotation == "hello"


# --- column spec validation ---


def test_too_few_fields_raises():
    with pytest.raises(ValueError, match="expected 4 tab-separated fields"):
        parse_tiers(["room\t0.0\n"], 0, TRANSCRIPTION_4COL)


def test_too_many_fields_raises_naming_tier():
    line = "room\t0.0\t1.0\ttext\textra\n"
    with pytest.raises(ValueError, match="tier 'room'"):
        parse_tiers([line], 0, TRANSCRIPTION_4COL)


def test_spec_width():
    assert TRANSCRIPTION_4COL.width == 4
    assert GAZE_9COL.width == 9
    assert ElanColumnSpec(tier=0, start=1, end=2, annotation=8).width == 9


# --- full transcription round trip ---


def test_transcript_json_round_trip(tmp_path):
    # subsequence_start is 0 so the exported tiers are unshifted and comparable to the source.
    # Speaker labels ride on the words, so this is the speaker-aligned component.
    meta = JsonMeta(
        component="speaker_aligned_transcription",
        algorithm="whisperx",
        subsequence_start=0.0,
        tables={"tracks": ["segments", "words"]},
    )
    transcript = parse_transcription(
        {
            "meta": meta.model_dump(),
            "tracks": {
                "left_mic": {
                    "total": {"text": "end of this year", "start": 0.6, "end": 1.2},
                    "segments": [
                        {
                            "start": 0.6,
                            "end": 1.2,
                            "text": "end of this year",
                            "speaker": "p1",
                            "speaker_confidence": 1.0,
                        }
                    ],
                    "words": [
                        {"word": "end", "start": 0.6, "end": 0.7, "speaker": "p1"},
                        {"word": "of", "start": 0.7, "end": 0.8, "speaker": "p1"},
                        {"word": "this", "start": 0.85, "end": 1.0, "speaker": "p1"},
                        {"word": "year", "start": 1.0, "end": 1.2, "speaker": "p1"},
                    ],
                }
            },
        }
    )
    path = tmp_path / "rt.txt"
    tiers = transcript_to_tiers(transcript, export_words=True)
    assert [t.tier_name for t in tiers] == ["left_mic__segments__p1", "left_mic__words__p1"]
    write_elan_txt(path, tiers, TRANSCRIPTION_4COL)

    out = tiers_to_transcript(parse_elan_file(path, TRANSCRIPTION_4COL).tiers)
    restored = out.tracks["left_mic"]

    # The payload survives the round trip; meta is rebuilt from the tiers, so the component is
    # recovered but the algorithm now records that this is hand-corrected ELAN data.
    assert out.meta.component == meta.component
    assert out.meta.algorithm == "elan"
    assert restored.total.text == "end of this year"
    assert [(s.start, s.end, s.text) for s in restored.segments] == [(0.6, 1.2, "end of this year")]
    assert [w.word for w in restored.words] == ["end", "of", "this", "year"]
    assert {w.speaker for w in restored.words} == {"p1"}
    assert {s.speaker for s in restored.segments} == {"p1"}
