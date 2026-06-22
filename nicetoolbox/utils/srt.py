"""
Reusable JSON -> SRT writer for audio transcription detectors.

Consumes our unified component output (per track: ``{"segments": [...], "word_segments": [...]}``)
and produces word-level highlighted SRT files: every word gets its own timestamp block with the
current word underlined (``<u>word</u>``), and gaps between words get a plain-text block showing the
full segment text. When the words carry a ``speaker`` field (speaker-aligned transcription), each
line is prefixed with ``<speaker>: ``.
"""

import logging
import os


class SrtWriter:
    """Build word-level highlighted SRT files from unified transcription segments."""

    SENTENCE_END = frozenset({".", "?", "!"})

    @staticmethod
    def _timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds % 1) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _segment_words(self, segment: dict, word_segments: list | None) -> list:
        """
        Return the word list for a segment.

        If the segment already carries its ``words`` (audio_transcription), use them directly.
        Otherwise reconstruct them by time-overlap from ``word_segments`` (speaker_aligned, where
        per-segment words have been stripped but ``word_segments`` retain speaker labels). Falls
        back to a single synthetic word spanning the segment if nothing overlaps.
        """
        words = segment.get("words")
        if words:
            return words

        if word_segments:
            seg_start, seg_end = segment["start"], segment["end"]
            overlapping = [w for w in word_segments if w["start"] < seg_end and w["end"] > seg_start]
            if overlapping:
                return overlapping

        return [
            {
                "word": segment.get("text", "").strip(),
                "start": segment["start"],
                "end": segment["end"],
            }
        ]

    @staticmethod
    def _speaker_prefix(words: list) -> str:
        for word in words:
            speaker = word.get("speaker")
            if speaker:
                return f"{speaker}: "
        return ""

    def _line(self, words: list, highlight_idx: int | None) -> str:
        prefix = self._speaker_prefix(words)
        parts = []
        for i, word in enumerate(words):
            if i == highlight_idx:
                parts.append(f"<u>{word['word']}</u>")
            else:
                parts.append(word["word"])
        return prefix + " ".join(parts)

    def to_srt(self, segments: list, word_segments: list | None = None) -> str:
        """Render segments as a word-level highlighted SRT string."""
        if not segments:
            return ""

        blocks = []
        n = 1
        for segment in segments:
            words = self._segment_words(segment, word_segments)
            for i, word in enumerate(words):
                # Highlight block: current word underlined, full segment text shown.
                blocks.append(
                    f"{n}\n{self._timestamp(word['start'])} --> "
                    f"{self._timestamp(word['end'])}\n"
                    f"{self._line(words, i)}"
                )
                n += 1
                # Gap block: silence between this word and the next (only when a real gap exists).
                if i < len(words) - 1:
                    gap_start = word["end"]
                    gap_end = words[i + 1]["start"]
                    if gap_end > gap_start:
                        blocks.append(
                            f"{n}\n{self._timestamp(gap_start)} --> "
                            f"{self._timestamp(gap_end)}\n"
                            f"{self._line(words, None)}"
                        )
                        n += 1
        return "\n\n".join(blocks) + "\n"

    def write_tracks(self, tracks: dict, out_dir: str, suffix: str = ".srt") -> None:
        """
        Write one SRT file per track into ``out_dir``.

        Args:
            tracks: ``{track_name: {"segments": [...], "word_segments": [...]}}``.
            out_dir: Directory to write ``<track><suffix>`` files into (created if missing).
            suffix: File suffix for each track's SRT file.
        """
        os.makedirs(out_dir, exist_ok=True)
        for track_name, data in tracks.items():
            srt = self.to_srt(data.get("segments", []), data.get("word_segments"))
            path = os.path.join(out_dir, f"{track_name}{suffix}")
            with open(path, "w", encoding="utf-8") as f:
                f.write(srt)
            logging.info(f"Wrote SRT for track '{track_name}' to {path}")
