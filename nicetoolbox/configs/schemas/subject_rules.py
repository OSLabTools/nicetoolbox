"""
Subject identity rules for video and audio tracks.
"""

from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, List, Optional, Union

from pydantic import BaseModel, Field

from ..models.models_registry import ModelsRegistry
from ..models.video_timestamp import VideoTimeRange

VIDEO_SUBJECT_RULES_REGISTRY = ModelsRegistry()
video_subject_rule = VIDEO_SUBJECT_RULES_REGISTRY.register

AUDIO_SUBJECT_RULES_REGISTRY = ModelsRegistry()
audio_subject_rule = AUDIO_SUBJECT_RULES_REGISTRY.register


# =======================================
#           Base Rule Config
# ========================================


class BaseRuleConfig(BaseModel):
    rule: str

    @abstractmethod
    def expected_subjects(self) -> Optional[List[str]]:
        """Subjects expected on this track/camera, as declared by the rule."""
        ...


def parse_subjects_rule(values: Any, registry: ModelsRegistry) -> Any:
    """Parse the raw subjects dict of a track into the rule config selected by its rule key."""
    # check if subjects already parsed
    raw = values.get("subjects") if isinstance(values, dict) else None
    if raw is None or isinstance(raw, BaseRuleConfig):
        return values
    # get rule id
    if not isinstance(raw, dict) or "rule" not in raw:
        raise KeyError("Missing 'rule' key in dataset property assignment rule")
    rule = raw["rule"]
    # parse raw dict into subject assignment rule
    values = deepcopy(values)
    values["subjects"] = registry.parse(rule, raw)
    return values


# =======================================
#          Shared Rules Configs
# ========================================


# This rules configs are the same for video and audio, so we reuse them
@audio_subject_rule("none")
@video_subject_rule("none")
class NoneRuleConfig(BaseRuleConfig):
    def expected_subjects(self):
        return []  # no subjects expected


@audio_subject_rule("assign")
@video_subject_rule("assign")
class AssignRuleConfig(BaseRuleConfig):
    subject: str

    def expected_subjects(self):
        return [self.subject]  # single subject expected


# ========================================
#           Video Rules Config
# ========================================


@video_subject_rule("vsplit")
class VSplitRuleConfig(BaseRuleConfig):
    order: List[str]
    ratio: Union[float, List[float], None] = None

    def expected_subjects(self):
        return list(self.order)


@video_subject_rule("hsplit")
class HSplitRuleConfig(BaseRuleConfig):
    order: List[str]
    ratio: Union[float, List[float], None] = None

    def expected_subjects(self):
        return list(self.order)


@video_subject_rule("path")
class VideoPathRuleConfig(BaseRuleConfig):
    # npz with precomputed subject regions
    source: Path

    def expected_subjects(self):
        return None  # known only once the file is loaded


@video_subject_rule("bbox")
class BBoxRuleConfig(BaseRuleConfig):
    boxes: dict[str, List[float]]

    def expected_subjects(self):
        return list(self.boxes)


# ========================================
#           Audio Rules Config
# ========================================


@audio_subject_rule("path")
class AudioPathRuleConfig(BaseRuleConfig):
    source: Path
    track: str

    def expected_subjects(self):
        return None  # known only once the file is loaded


@audio_subject_rule("audio_range")
class AudioRangeRuleConfig(BaseRuleConfig):
    algorithm: str
    ranges: dict[str, Annotated[List[VideoTimeRange], Field(min_length=1)]] = Field(min_length=1)

    def expected_subjects(self):
        return list(self.ranges)
