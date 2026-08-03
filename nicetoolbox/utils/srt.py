"""
Reusable SRT writer for audio transcription components.

Consumes the pydantic models from ``nicetoolbox_core.data.json_schema`` (a TrackTranscription or
AlignedTranscription per track).

Segments are the subtitle: each block shows the segment's own text, which is the reference
transcript. Words only supply timing — they say when to advance the highlight within that text, so
a segment whose words were edited or deleted still displays what the transcript claims was said.
A segment with no usable words is shown whole, spanning its own start and end.

When the items carry a speaker (speaker-aligned transcription), each line is prefixed with
``<speaker>: ``.
"""

import logging
import os

from nicetoolbox_core.data.json_schema import (
    AlignedTranscription,
    AudioTranscriptionSegment,
    AudioTranscriptionWord,
    TrackTranscription,
)

TrackModel = TrackTranscription | AlignedTranscription

# Stripped from a word's edges when its exact form is not found in the segment text. The two are
# edited independently, so a word may carry punctuation the text does not, or vice versa.
_EDGE_PUNCTUATION = " \t.,!?;:\"'()[]…-–—"


def _match_forms(word: str) -> list[str]:
    """The forms to try when locating a word in segment text, most exact first."""
    exact = word.lower()
    bare = exact.strip(_EDGE_PUNCTUATION)
    return [exact, bare] if bare and bare != exact else [exact]


class SrtWriter:
    """Build word-level highlighted SRT files from transcription component models."""

    @staticmethod
    def _timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds % 1) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _is_timed(item) -> bool:
        """True when an item carries both timings, so it can anchor an SRT block."""
        return item.start is not None and item.end is not None

    def _segment_words(self, index: int, words: list[AudioTranscriptionWord]) -> list[AudioTranscriptionWord]:
        """The timed words belonging to a segment, by their recorded `segment_index`.

        May be empty: words are optional, and a producer that emitted none — or an annotator who
        deleted them — leaves the segment to be shown whole.
        """
        return [word for word in words if word.segment_index == index and self._is_timed(word)]

    @staticmethod
    def _speaker_prefix(segment: AudioTranscriptionSegment, words: list[AudioTranscriptionWord]) -> str:
        """The speaker label to prefix a line with, taken from the segment, else from its words."""
        speaker = getattr(segment, "speaker", None)
        if not speaker:
            speaker = next((s for s in (getattr(w, "speaker", None) for w in words) if s), None)
        return f"{speaker}: " if speaker else ""

    @staticmethod
    def _highlight_spans(text: str, words: list[AudioTranscriptionWord]) -> list[tuple[int, int] | None]:
        """Locate each word inside the segment text, as (start, end) character offsets.

        Matching is case-insensitive, and falls back to the word stripped of surrounding punctuation
        when the exact form is not present: the two tiers are edited independently, so a word may
        carry a comma the segment text does not, or differ in case at a sentence start. Offsets
        index the original text, so what gets underlined is always its own spelling.

        The cursor only moves forward, so a word repeated later in the text cannot match an earlier
        occurrence that a previous word already consumed. A word that cannot be found at all yields
        None and is simply not highlighted.
        """
        # `lower`, not `casefold`: casefolding can change a string's length (ß -> ss), which would
        # invalidate the offsets these spans index the original text with.
        haystack = text.lower()
        spans: list[tuple[int, int] | None] = []
        cursor = 0

        for word in words:
            for needle in _match_forms(word.word):
                found = haystack.find(needle, cursor)
                if found != -1:
                    spans.append((found, found + len(needle)))
                    cursor = found + len(needle)
                    break
            else:
                spans.append(None)

        return spans

    def _line(self, text: str, prefix: str, span: tuple[int, int] | None) -> str:
        """The segment text, with the span at `span` underlined."""
        if span is None:
            return prefix + text
        start, end = span
        return f"{prefix}{text[:start]}<u>{text[start:end]}</u>{text[end:]}"

    def to_srt(self, segments: list[AudioTranscriptionSegment], words: list[AudioTranscriptionWord]) -> str:
        """Render segments as an SRT string, highlighting each word in turn within its segment."""
        blocks: list[str] = []
        n = 1

        for index, segment in enumerate(segments):
            if not self._is_timed(segment):
                continue

            text = segment.text.strip()
            segment_words = self._segment_words(index, words)
            prefix = self._speaker_prefix(segment, segment_words)

            # No words to advance the highlight: show the segment whole, for its own span.
            if not segment_words:
                blocks.append(
                    f"{n}\n{self._timestamp(segment.start)} --> {self._timestamp(segment.end)}\n{prefix}{text}"
                )
                n += 1
                continue

            spans = self._highlight_spans(text, segment_words)
            for i, word in enumerate(segment_words):
                blocks.append(
                    f"{n}\n{self._timestamp(word.start)} --> "
                    f"{self._timestamp(word.end)}\n"
                    f"{self._line(text, prefix, spans[i])}"
                )
                n += 1
                # Gap block: silence between this word and the next, with nothing highlighted.
                if i < len(segment_words) - 1:
                    gap_start, gap_end = word.end, segment_words[i + 1].start
                    if gap_end > gap_start:
                        blocks.append(
                            f"{n}\n{self._timestamp(gap_start)} --> "
                            f"{self._timestamp(gap_end)}\n"
                            f"{self._line(text, prefix, None)}"
                        )
                        n += 1

        return "\n\n".join(blocks) + "\n" if blocks else ""

    def write_tracks(self, tracks: dict[str, TrackModel], out_dir: str, suffix: str = ".srt") -> None:
        """
        Write one SRT file per track into ``out_dir``.

        Args:
            tracks: ``{track_name: TrackTranscription | AlignedTranscription}`` — the ``tracks``
                mapping of a transcription component model.
            out_dir: Directory to write ``<track><suffix>`` files into (created if missing).
            suffix: File suffix for each track's SRT file.
        """
        os.makedirs(out_dir, exist_ok=True)
        for track_name, track in tracks.items():
            srt = self.to_srt(track.segments, track.words)
            path = os.path.join(out_dir, f"{track_name}{suffix}")
            with open(path, "w", encoding="utf-8") as f:
                f.write(srt)
            logging.info(f"Wrote SRT for track '{track_name}' to {path}")
