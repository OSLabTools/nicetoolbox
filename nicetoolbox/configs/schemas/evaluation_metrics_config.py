from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from ..models.models_registry import ModelsRegistry
from .evaluation_aggr import AggSpec
from .evaluation_group_by import GroupBySpec
from .evaluation_input_block import BaseInputBlock, InputBlock
from .evaluation_transcript_ref import TranscriptRef

METRICS_REGISTRY = ModelsRegistry()
metric_config = METRICS_REGISTRY.register


class BaseMetricConfig(BaseModel):
    """Base schema for all evaluation metrics."""

    metric_type: str
    _metric_name: str = PrivateAttr()

    def input_blocks(self) -> list[BaseInputBlock]:
        """Return all InputBlock fields on this metric config."""
        result = []
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, BaseInputBlock):
                result.append(value)
        return result


# =============================================================================
# Pose estimation metrics
# =============================================================================


@metric_config("bone_length")
class BoneLengthConfig(BaseMetricConfig):
    visualize: bool = True

    predictions: InputBlock

    summary_group_by: GroupBySpec
    summary_aggr: AggSpec

    @model_validator(mode="after")
    def _require_bone_dims(self) -> "BoneLengthConfig":
        mandatory = {"subject", "sequence", "label"}
        if not self.summary_group_by.contains(*mandatory):
            missing = mandatory - set(self.summary_group_by.dims)
            raise ValueError(
                f"group_by for bone_length must include {missing}. "
                f"Bone length is a per-person, per-sequence, per-bone property — "
                f"pooling across any of these dimensions produces meaningless averages."
            )
        return self


@metric_config("distance_error")
class DistanceErrorConfig(BaseMetricConfig):
    visualize: bool = True
    norm: Literal["l1", "l2"] = "l2"
    broadcast_single: bool = False

    predictions: InputBlock
    ground_truth: InputBlock

    summary_group_by: GroupBySpec
    summary_aggr: AggSpec


@metric_config("pck")
class PCKConfig(BaseMetricConfig):
    visualize: bool = True
    threshold: float  # joint is correct if L2 dist <= threshold (input units, e.g. pixels)
    broadcast_single: bool = False

    predictions: InputBlock
    ground_truth: InputBlock

    summary_group_by: GroupBySpec
    summary_aggr: AggSpec


@metric_config("missing_points")
class MissingPointsConfig(BaseMetricConfig):
    visualize: bool = True

    predictions: InputBlock
    min_confidence: float | None = None

    missing_points_summary_group_by: GroupBySpec
    missing_points_summary_aggr: AggSpec


# =============================================================================
# Categorical metrics
# =============================================================================


@metric_config("roc_auc")
class RocAucConfig(BaseMetricConfig):
    visualize: bool = True
    broadcast_single: bool = False
    negate_scores: bool = False  # set True when lower score = more positive (e.g. distance)

    predictions: InputBlock  # float confidence scores
    ground_truth: InputBlock  # bool labels

    compute_group_by: GroupBySpec


@metric_config("pr_curve")
class PrCurveConfig(BaseMetricConfig):
    visualize: bool = True
    broadcast_single: bool = False
    negate_scores: bool = False  # set True when lower score = more positive (e.g. distance)

    predictions: InputBlock  # float confidence scores
    ground_truth: InputBlock  # bool labels

    compute_group_by: GroupBySpec


@metric_config("confusion_matrix")
class ConfusionMatrixConfig(BaseMetricConfig):
    visualize: bool = True
    broadcast_single: bool = False

    predictions: InputBlock
    ground_truth: InputBlock

    compute_group_by: GroupBySpec


# =============================================================================
# Audio metrics
# =============================================================================


@metric_config("transcription_error_rate")
class TranscriptionErrorRateConfig(BaseMetricConfig):
    """
    WER/CER of one predicted transcript track against one reference track.

    Scores exactly one (prediction, ground truth) pair, so there is nothing to group by or
    aggregate - to compare several algorithms, declare one [metrics.*] entry per algorithm.
    """

    visualize: bool = True

    predictions: TranscriptRef
    ground_truth: TranscriptRef

    # Normalization matrix. Mandatory on purpose: WER is only interpretable next to the regime
    # that produced it, so every entry has to state all five axes rather than inherit a default.
    remove_filler: bool  # filler words like "[UM]"
    lower_case: bool
    strip_punctuation: bool
    expand_contractions: bool  # won't -> will not
    normalize_numbers: bool  # 1,000 -> 1000

    # Override for the CrisperWhisper-token -> annotation-token filler mapping, used only when
    # remove_filler = false. Empty means use the built-in defaults in normalization.py.
    filler_map: dict[str, str] = Field(default_factory=dict)

    measures: list[Literal["wer", "mer", "wil", "wip", "cer"]] = Field(
        default_factory=lambda: ["wer", "mer", "wil", "wip", "cer"]
    )
