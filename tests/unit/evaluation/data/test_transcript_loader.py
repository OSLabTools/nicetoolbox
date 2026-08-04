"""Tests for loading a single track out of a transcription JSON."""

import json

import pytest
from pydantic import ValidationError

from nicetoolbox.configs.schemas.evaluation_transcript_ref import TranscriptRef
from nicetoolbox.evaluation.data.transcript_loader import load_transcript
from tests.unit.evaluation.metrics.audio.conftest import write_transcript


class TestHappyPath:
    def test_returns_meta_and_track(self, tmp_path):
        path = write_transcript(tmp_path, "pred", {"room": "hello world"}, algorithm="crisper_whisper")

        meta, track = load_transcript(TranscriptRef(path=path, track="room"))

        assert meta.algorithm == "crisper_whisper"
        assert meta.component == "audio_transcription"
        assert track.total.text == "hello world"

    def test_selects_the_requested_track(self, tmp_path):
        path = write_transcript(tmp_path, "pred", {"room": "in the room", "left_mic": "on the left"})

        _, track = load_transcript(TranscriptRef(path=path, track="left_mic"))

        assert track.total.text == "on the left"

    def test_speaker_aligned_file_is_accepted(self, tmp_path):
        """The aligned schema only adds fields, and pydantic ignores extras.

        This is load-bearing: it is what lets one model serve every transcription component.
        Adding extra="forbid" to json_schema.py would break the metric, and this test.
        """
        path = write_transcript(tmp_path, "aligned", {"room": "hello"}, component="speaker_aligned_transcription")
        raw = json.loads(path.read_text())
        raw["tracks"]["room"]["segments"] = [
            {"text": "hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "speaker_confidence": 1.0}
        ]
        raw["tracks"]["room"]["words"] = [{"word": "hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        path.write_text(json.dumps(raw))

        meta, track = load_transcript(TranscriptRef(path=path, track="room"))

        assert meta.component == "speaker_aligned_transcription"
        assert track.total.text == "hello"


class TestErrors:
    def test_missing_track_lists_available_ones(self, tmp_path):
        path = write_transcript(tmp_path, "pred", {"room": "hello", "left_mic": "hi"})

        with pytest.raises(KeyError, match="Available.*left_mic.*room"):
            load_transcript(TranscriptRef(path=path, track="right_mic"))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_transcript(TranscriptRef(path=tmp_path / "nope.json", track="room"))

    def test_malformed_file_raises_pydantic_error_unchanged(self, tmp_path):
        """Deliberately unwrapped: pydantic's message names the fields that actually failed."""
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps({"room": {"total": {"text": "hello"}}}))

        with pytest.raises(ValidationError, match="meta"):
            load_transcript(TranscriptRef(path=path, track="room"))


class TestRefValidation:
    def test_wildcard_track_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="Wildcards are not allowed"):
            TranscriptRef(path=tmp_path / "a.json", track="*")
