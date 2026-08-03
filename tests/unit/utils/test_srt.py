"""Tests for the reusable transcription-model -> SRT writer."""

from nicetoolbox.utils.srt import SrtWriter
from nicetoolbox_core.data.json_schema import (
    AlignedSegment,
    AlignedTranscription,
    AlignedWord,
    AudioTranscriptionSegment,
    AudioTranscriptionTotal,
    AudioTranscriptionWord,
    TrackTranscription,
)


def _seg(start, end, text):
    return AudioTranscriptionSegment(text=text, start=start, end=end)


def _word(word, start, end, segment_index=0):
    return AudioTranscriptionWord(word=word, start=start, end=end, segment_index=segment_index)


def test_timestamp_formatting():
    assert SrtWriter._timestamp(0) == "00:00:00,000"
    assert SrtWriter._timestamp(3661.5) == "01:01:01,500"
    assert SrtWriter._timestamp(0.36) == "00:00:00,360"


def test_to_srt_highlight_and_gap_blocks():
    segments = [_seg(0.0, 2.0, "hello world")]
    words = [_word("hello", 0.0, 0.5), _word("world", 1.0, 2.0)]  # gap 0.5 -> 1.0

    srt = SrtWriter().to_srt(segments, words)

    # Highlight block for "hello", gap block, highlight block for "world" => 3 blocks.
    assert "<u>hello</u> world" in srt
    assert "hello <u>world</u>" in srt
    # Gap block is plain text (no underline) covering the silence between the two words.
    assert "00:00:00,500 --> 00:00:01,000\nhello world" in srt
    assert srt.count("\n\n") == 2  # three blocks separated by blank lines


def test_to_srt_no_gap_block_when_words_contiguous():
    segments = [_seg(0.0, 1.0, "a b")]
    words = [_word("a", 0.0, 0.5), _word("b", 0.5, 1.0)]  # no gap

    srt = SrtWriter().to_srt(segments, words)
    assert srt.count("\n\n") == 1  # only the two highlight blocks, no gap block


def test_to_srt_empty_segments():
    assert SrtWriter().to_srt([], []) == ""


def test_segment_without_words_is_shown_whole():
    # Words are optional. With none to advance the highlight, the segment gets one block spanning
    # its own timings, and nothing is underlined — there is no "current word" to point at.
    srt = SrtWriter().to_srt([_seg(0.0, 1.0, " hi there ")], [])
    assert srt == "1\n00:00:00,000 --> 00:00:01,000\nhi there\n"


def test_display_text_comes_from_the_segment_not_the_words():
    # The segment text is the reference transcript, so a word deleted from the word list must not
    # change what the subtitle shows — it only stops being highlighted.
    srt = SrtWriter().to_srt([_seg(0.0, 3.0, "a b c")], [_word("a", 0.0, 1.0), _word("c", 2.0, 3.0)])

    assert "<u>a</u> b c" in srt
    assert "a b <u>c</u>" in srt
    assert "a c" not in srt  # the old behaviour joined the surviving words instead


def test_repeated_word_highlights_each_occurrence_in_turn():
    # Locating words by a plain search would underline the first "the" twice; the cursor only
    # moves forward, so the second word finds the second occurrence.
    segments = [_seg(0.0, 4.0, "the cat the dog")]
    words = [_word("the", 0.0, 1.0), _word("cat", 1.0, 2.0), _word("the", 2.0, 3.0), _word("dog", 3.0, 4.0)]

    srt = SrtWriter().to_srt(segments, words)

    assert "<u>the</u> cat the dog" in srt
    assert "the cat <u>the</u> dog" in srt


def test_word_matching_ignores_case():
    # The segment text and the word tier are edited independently, so a sentence-initial capital
    # need not survive into the word. The text's own spelling is what gets displayed.
    srt = SrtWriter().to_srt([_seg(0.0, 1.0, "Okay then")], [_word("okay", 0.0, 1.0)])
    assert "<u>Okay</u> then" in srt


def test_word_matching_tolerates_edge_punctuation():
    # A word carrying a comma the segment text lacks still finds its place.
    srt = SrtWriter().to_srt([_seg(0.0, 2.0, "yes I do")], [_word("yes,", 0.0, 1.0), _word("I", 1.0, 2.0)])
    assert "<u>yes</u> I do" in srt


def test_repeated_word_still_advances_when_matching_loosely():
    # Case-insensitive matching must not let a later word rematch an earlier occurrence.
    segments = [_seg(0.0, 4.0, "The cat the Dog")]
    words = [_word("the", 0.0, 1.0), _word("cat", 1.0, 2.0), _word("The", 2.0, 3.0), _word("dog", 3.0, 4.0)]

    srt = SrtWriter().to_srt(segments, words)

    assert "<u>The</u> cat the Dog" in srt
    assert "The cat <u>the</u> Dog" in srt
    assert "The cat the <u>Dog</u>" in srt


def test_word_absent_from_the_segment_text_is_not_highlighted():
    # The tiers disagree: the word exists but its text does not appear in the segment.
    srt = SrtWriter().to_srt([_seg(0.0, 2.0, "hello world")], [_word("goodbye", 0.0, 1.0)])

    assert "<u>" not in srt
    assert "hello world" in srt


def test_untimed_words_are_skipped():
    # Alignment can leave a word with no timings; it cannot anchor an SRT block.
    segments = [_seg(0.0, 2.0, "hello world")]
    words = [_word("hello", 0.0, 0.5), AudioTranscriptionWord(word="42")]

    srt = SrtWriter().to_srt(segments, words)
    assert "<u>hello</u>" in srt
    assert "42" not in srt


def test_untimed_segment_is_skipped():
    srt = SrtWriter().to_srt([AudioTranscriptionSegment(text="no timings")], [_word("a", 0.0, 1.0)])
    assert srt == ""


def test_overlap_reconstruction_and_speaker_prefix():
    # speaker_aligned case: words point back at their segment and carry speakers.
    segments = [
        AlignedSegment(text="hi there", start=0.0, end=1.0, speaker="Alice", speaker_confidence=1.0),
        AlignedSegment(text="bye now", start=2.0, end=3.0, speaker="Bob", speaker_confidence=1.0),
    ]
    words = [
        AlignedWord(word="hi", start=0.0, end=0.4, speaker="Alice", segment_index=0),
        AlignedWord(word="there", start=0.5, end=1.0, speaker="Alice", segment_index=0),
        AlignedWord(word="bye", start=2.0, end=2.4, speaker="Bob", segment_index=1),
        AlignedWord(word="now", start=2.5, end=3.0, speaker="Bob", segment_index=1),
    ]

    srt = SrtWriter().to_srt(segments, words)

    # Each segment's words are gathered by segment_index, prefixed with the speaker label.
    assert "Alice: <u>hi</u> there" in srt
    assert "Bob: <u>bye</u> now" in srt
    # Words from the second segment must not leak into the first.
    assert "Alice: <u>hi</u> there bye" not in srt


def test_no_speaker_prefix_on_plain_transcription():
    # AudioTranscriptionWord has no speaker attribute at all.
    srt = SrtWriter().to_srt([_seg(0.0, 1.0, "hello")], [_word("hello", 0.0, 1.0)])
    assert ":" not in srt.split("\n")[2]  # the text line carries no "<speaker>: " prefix


def test_write_tracks(tmp_path):
    tracks = {
        "room": TrackTranscription(
            total=AudioTranscriptionTotal(text="hello", start=0.0, end=1.0),
            segments=[_seg(0.0, 1.0, "hello")],
            words=[_word("hello", 0.0, 1.0)],
        )
    }

    SrtWriter().write_tracks(tracks, str(tmp_path))

    out = tmp_path / "room.srt"
    assert out.exists()
    assert "<u>hello</u>" in out.read_text()


def test_write_tracks_accepts_aligned_tracks(tmp_path):
    tracks = {
        "room": AlignedTranscription(
            total=AudioTranscriptionTotal(text="hi", start=0.0, end=1.0),
            segments=[AlignedSegment(text="hi", start=0.0, end=1.0, speaker="Alice", speaker_confidence=1.0)],
            words=[AlignedWord(word="hi", start=0.0, end=1.0, speaker="Alice", segment_index=0)],
        )
    }

    SrtWriter().write_tracks(tracks, str(tmp_path))

    assert "Alice: <u>hi</u>" in (tmp_path / "room.srt").read_text()
