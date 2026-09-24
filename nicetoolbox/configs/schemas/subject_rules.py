"""
Subject identity rules for video and audio tracks.
"""

from pathlib import Path
from typing import List, Union

from pydantic import BaseModel

from ..models.models_registry import ModelsRegistry

VIDEO_SUBJECT_RULES_REGISTRY = ModelsRegistry()
video_subject_rule = VIDEO_SUBJECT_RULES_REGISTRY.register

AUDIO_SUBJECT_RULES_REGISTRY = ModelsRegistry()
audio_subject_rule = AUDIO_SUBJECT_RULES_REGISTRY.register


# =======================================
#           Base Rule Config
# ========================================


class BaseRuleConfig(BaseModel):
    rule: str


# =======================================
#          Shared Rules Configs
# ========================================


# This rules configs are the same for video and audio, so we reuse them
@audio_subject_rule("none")
@video_subject_rule("none")
class NoneRuleConfig(BaseRuleConfig): ...


@audio_subject_rule("assign")
@video_subject_rule("assign")
class AssignRuleConfig(BaseRuleConfig):
    subject: str


@audio_subject_rule("path")
@video_subject_rule("path")
class PathRuleConfig(BaseRuleConfig):
    path: Path


# ========================================
#           Video Rules Config
# ========================================


@video_subject_rule("vsplit")
class VSplitRuleConfig(BaseRuleConfig):
    order: List[str]
    ratio: Union[float, List[float], None] = None


@video_subject_rule("hsplit")
class HSplitRuleConfig(BaseRuleConfig):
    order: List[str]
    ratio: Union[float, List[float], None] = None


@video_subject_rule("bbox")
class BBoxRuleConfig(BaseRuleConfig):
    boxes: dict[str, List[float]]
