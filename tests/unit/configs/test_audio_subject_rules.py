import pytest
from pydantic import ValidationError

from nicetoolbox.configs.schemas.dataset_properties import AudioTrackConfig
from nicetoolbox.configs.schemas.subject_rules import AssignRuleConfig, NoneRuleConfig


def test_audio_track_parses_assign_rule():
    """
    Given: An audio track with an `assign` subjects rule.
    When:  AudioTrackConfig is validated.
    Then:  `subjects` is parsed into AssignRuleConfig carrying the subject name.
    """
    track = AudioTrackConfig(camera="view_left", subjects={"rule": "assign", "subject": "person_left"})
    assert isinstance(track.subjects, AssignRuleConfig)
    assert track.subjects.subject == "person_left"


def test_audio_track_parses_none_rule():
    """
    Given: An audio track with a `none` subjects rule.
    When:  AudioTrackConfig is validated.
    Then:  `subjects` is parsed into NoneRuleConfig.
    """
    track = AudioTrackConfig(path="/tmp/room.wav", subjects={"rule": "none"})
    assert isinstance(track.subjects, NoneRuleConfig)


def test_audio_track_requires_subjects():
    """
    Given: An audio track without a `subjects` rule.
    When:  AudioTrackConfig is validated.
    Then:  Validation fails — every track must declare a rule.
    """
    with pytest.raises(ValidationError, match="subjects"):
        AudioTrackConfig(path="/tmp/room.wav")


def test_audio_track_missing_rule_key_raises():
    """
    Given: A `subjects` table without a `rule` key.
    When:  AudioTrackConfig is validated.
    Then:  A KeyError names the missing `rule` key.
    """
    with pytest.raises(KeyError, match="rule"):
        AudioTrackConfig(path="/tmp/room.wav", subjects={"subject": "person_left"})


@pytest.mark.parametrize("rule", ["vsplit", "bbox", "unknown"])
def test_audio_track_rejects_non_audio_rules(rule):
    """
    Given: A rule that is not registered for audio (video-only or unknown).
    When:  AudioTrackConfig is validated.
    Then:  Parsing fails.
    """
    with pytest.raises(Exception, match=rule):
        AudioTrackConfig(path="/tmp/room.wav", subjects={"rule": rule})
