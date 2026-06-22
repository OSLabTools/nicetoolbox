"""Tests for the reusable JSON -> SRT writer."""

from nicetoolbox.utils.srt import SrtWriter


def test_timestamp_formatting():
    assert SrtWriter._timestamp(0) == "00:00:00,000"
    assert SrtWriter._timestamp(3661.5) == "01:01:01,500"
    assert SrtWriter._timestamp(0.36) == "00:00:00,360"


def test_to_srt_highlight_and_gap_blocks():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 1.0, "end": 2.0},  # gap 0.5 -> 1.0
            ],
        }
    ]
    srt = SrtWriter().to_srt(segments)

    # Highlight block for "hello", gap block, highlight block for "world" => 3 blocks.
    assert "<u>hello</u> world" in srt
    assert "hello <u>world</u>" in srt
    # Gap block is plain text (no underline) covering the silence between the two words.
    assert "00:00:00,500 --> 00:00:01,000\nhello world" in srt
    assert srt.count("\n\n") == 2  # three blocks separated by blank lines


def test_to_srt_no_gap_block_when_words_contiguous():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "a b",
            "words": [
                {"word": "a", "start": 0.0, "end": 0.5},
                {"word": "b", "start": 0.5, "end": 1.0},  # no gap
            ],
        }
    ]
    srt = SrtWriter().to_srt(segments)
    assert srt.count("\n\n") == 1  # only the two highlight blocks, no gap block


def test_to_srt_empty_segments():
    assert SrtWriter().to_srt([]) == ""


def test_overlap_reconstruction_and_speaker_prefix():
    # speaker_aligned case: segments have no per-segment words; word_segments carry speakers.
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hi there"},
        {"start": 2.0, "end": 3.0, "text": "bye now"},
    ]
    word_segments = [
        {"word": "hi", "start": 0.0, "end": 0.4, "speaker": "Alice"},
        {"word": "there", "start": 0.5, "end": 1.0, "speaker": "Alice"},
        {"word": "bye", "start": 2.0, "end": 2.4, "speaker": "Bob"},
        {"word": "now", "start": 2.5, "end": 3.0, "speaker": "Bob"},
    ]
    srt = SrtWriter().to_srt(segments, word_segments)

    # Each segment's words are reconstructed by time overlap, prefixed with the speaker label.
    assert "Alice: <u>hi</u> there" in srt
    assert "Bob: <u>bye</u> now" in srt
    # Words from the second segment must not leak into the first.
    assert "Alice: <u>hi</u> there bye" not in srt


def test_write_tracks(tmp_path):
    tracks = {
        "room": {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "words": [{"word": "hello", "start": 0.0, "end": 1.0}],
                }
            ],
            "word_segments": [{"word": "hello", "start": 0.0, "end": 1.0}],
        }
    }
    SrtWriter().write_tracks(tracks, str(tmp_path))
    out = tmp_path / "room.srt"
    assert out.exists()
    assert "<u>hello</u>" in out.read_text()
