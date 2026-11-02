from types import SimpleNamespace

import pytest

from nicetoolbox.configs.schemas.dataset_properties import SequenceConfig
from nicetoolbox.detectors.feature_detectors.subject_system.audio_subject_detector import AudioSubjectDetector
from nicetoolbox.detectors.subjects import assign_transcription_speakers
from nicetoolbox_core.data.json_schema import AudioDiarization, DiarizationSegment, TrackTranscription

SUBJECTS = ["person_left", "person_right"]


@pytest.fixture
def make_detector():
    """
    Build an AudioSubjectDetector without the full BaseFeature init: only the fields
    compute() reads are set, with tracks taken from a real SequenceConfig.
    """

    def _make(tracks, subjects=SUBJECTS, start=12.0, length=30.0):
        sequence = SequenceConfig(sequence_id="s1", subjects=subjects, audio={"tracks": tracks})
        detector = AudioSubjectDetector.__new__(AudioSubjectDetector)
        detector.subsequence_context = SimpleNamespace(sequence_properties=sequence)
        detector.subjects_descr = subjects
        detector.algorithm_instance = "audio_subject_detector"
        detector.subsequence_start = start
        detector.subsequence_length = length
        return detector

    return _make


def test_compute_none_and_assign_tracks(make_detector):
    """
    Given: A room track with `none` and two close mics with `assign`.
    When:  compute() runs.
    Then:  `none` yields no segments; each `assign` track is one segment spanning the
           whole subsequence, labeled with its subject; meta records the time range.
    """
    detector = make_detector(
        {
            "room": {"path": "/tmp/room.wav", "subjects": {"rule": "none"}},
            "left_mic": {"camera": "view_left", "subjects": {"rule": "assign", "subject": "person_left"}},
            "right_mic": {"camera": "view_right", "subjects": {"rule": "assign", "subject": "person_right"}},
        }
    )
    out = detector.compute()
    out.validate(detector.outputs)
    result = out.get_json("audio_diarization")

    assert result.tracks["room"].segments == []
    left = result.tracks["left_mic"].segments
    assert [(s.speaker, s.start, s.end) for s in left] == [("person_left", 0.0, 30.0)]
    right = result.tracks["right_mic"].segments
    assert [(s.speaker, s.start, s.end) for s in right] == [("person_right", 0.0, 30.0)]

    assert result.meta.component == "audio_diarization"
    assert result.meta.subsequence_start == 12.0
    assert result.meta.subsequence_length == 30.0


def test_compute_without_audio_tracks(make_detector):
    """A sequence with no audio tracks produces an empty, still valid output."""
    out = make_detector({}).compute()
    out.validate(AudioSubjectDetector.outputs)
    assert out.get_json("audio_diarization").tracks == {}


def test_compute_path_rule(make_detector, tmp_path):
    """
    Given: A `path` rule reading track `mix` of an audio_diarization JSON from a run starting at 10s,
           with a turn of a non-subject speaker.
    When:  compute() runs on a subsequence starting at 12s.
    Then:  Turns are shifted into the subsequence time base and clipped to it; the non-subject
           speaker is dropped.
    """
    source = AudioDiarization(
        meta={"component": "audio_diarization", "algorithm": "x", "subsequence_start": 10.0, "subsequence_length": 40},
        tracks={
            "mix": {
                "segments": [
                    {"speaker": "person_left", "start": 0.0, "end": 5.0},
                    {"speaker": "SPEAKER_07", "start": 5.0, "end": 6.0},
                    {"speaker": "person_right", "start": 6.0, "end": 50.0},
                ]
            }
        },
    )
    source_path = tmp_path / "diarization.json"
    source_path.write_text(source.model_dump_json())
    rule = {"rule": "path", "source": str(source_path), "track": "mix"}
    detector = make_detector({"room": {"path": "/tmp/room.wav", "subjects": rule}})

    segments = detector.compute().get_json("audio_diarization").tracks["room"].segments

    assert [(s.speaker, s.start, s.end) for s in segments] == [("person_left", 0.0, 3.0), ("person_right", 4.0, 30.0)]


def test_assign_unknown_subject_rejected_at_config(make_detector):
    """An `assign` rule naming an undeclared subject fails when the sequence config loads, naming track and subject."""
    with pytest.raises(ValueError, match="left_mic.*ghost"):
        make_detector({"left_mic": {"camera": "view_left", "subjects": {"rule": "assign", "subject": "ghost"}}})


def test_assign_transcription_speakers():
    """
    Given: A segment of three words spoken across a left -> right turn change, plus an untimed
           word, and a second segment outside every turn.
    When:  assign_transcription_speakers labels the transcription.
    Then:  Each word takes the speaker overlapping it the longest (None if untimed or uncovered);
           each segment takes its own dominant speaker, with the fraction of agreeing words.
    """
    transcription = TrackTranscription(
        total={"text": "a b c d e", "start": 0.0, "end": 12.0},
        segments=[
            {"text": "a b c d", "start": 0.0, "end": 4.0},
            {"text": "e", "start": 10.0, "end": 12.0},
        ],
        words=[
            {"word": "a", "segment_index": 0, "start": 0.0, "end": 1.0},
            {"word": "b", "segment_index": 0, "start": 1.0, "end": 2.0},
            {"word": "c", "segment_index": 0, "start": 2.4, "end": 3.0},
            {"word": "d", "segment_index": 0},
            {"word": "e", "segment_index": 1, "start": 10.0, "end": 12.0},
        ],
    )
    diarization = [
        DiarizationSegment(speaker="person_left", start=0.0, end=2.5),
        DiarizationSegment(speaker="person_right", start=2.5, end=4.0),
    ]

    result = assign_transcription_speakers(transcription, diarization)

    assert [w.speaker for w in result.words] == ["person_left", "person_left", "person_right", None, None]
    assert [(s.speaker, s.speaker_confidence) for s in result.segments] == [("person_left", 0.5), (None, 0.0)]
    assert result.total == transcription.total
