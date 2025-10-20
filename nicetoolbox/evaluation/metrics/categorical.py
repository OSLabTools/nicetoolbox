"""
Categorical metrics for classification tasks, including accuracy, precision,
recall, and F1 score.
"""

from collections import defaultdict
from inspect import FrameInfo
from typing import Dict, List, Tuple

import torch

from ..data.discovery import ChunkWorkItem
from .base_metric import Metric, MetricHandler, MetricReturnType
from .results_schema import AggregatedResult


class CategoricalMetric(MetricHandler):
    """
    Handler for all categorical metrics.
    """

    @property
    def name(self) -> str:
        return "categorical_metrics"

    def _create_metrics(self) -> Dict[str, Metric]:
        """Instantiates the individual metric objects this handler manages."""
        metric_map = {
            "accuracy": Accuracy,
            "precision": Precision,
            "recall": Recall,
            "f1_score": F1Score,
        }
        return {
            name: metric_map[name]()
            for name in self.cfg.metric_names
            if name in metric_map
        }


class _BinaryClassificationMetric(Metric):
    """Base class for binary classification metrics like precision, recall, accuracy."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset the metric's internal state: counts of TP, TN, FP, FN."""
        # Key: (component, algorithm, person, camera)
        self.counts: Dict[Tuple[str, str, str, str], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def update(
        self,
        preds: torch.Tensor,
        gts: torch.Tensor,
        meta_chunk: ChunkWorkItem,
        meta_frames: List[FrameInfo],
    ) -> None:
        """
        Update metric state with a batch of data and its corresponding metadata,
        accumulating counts of true positives, true negatives, false positives, and
        false negatives. Assumes binary classification with predictions and ground
        truths as boolean tensors.

        Args:
            preds (torch.Tensor): Predictions tensor (binary).
            gts (torch.Tensor): Ground truth tensor (binary).
            meta_chunk (ChunkWorkItem): Metadata for the chunk.
            meta_frames (List[FrameInfo]): List of frame metadata.
        """
        if gts is None:
            return

        preds_bool, gts_bool = preds.bool(), gts.bool()
        assert preds_bool.shape == gts_bool.shape, "Preds and gts shape does not match."

        context_keys = [
            (meta_chunk.component, meta_chunk.algorithm, frame.person, frame.camera)
            for frame in meta_frames
        ]
        unique_keys = set(context_keys)

        for key in unique_keys:
            select_frames = [k == key for k in context_keys]
            if not select_frames:
                continue
            preds = preds_bool[select_frames]
            gts = gts_bool[select_frames]
            assert preds.shape == gts.shape, "Preds and gts must have the same shape."

            self.counts[key]["tp"] += torch.sum(preds & gts).item()
            self.counts[key]["tn"] += torch.sum(~preds & ~gts).item()
            self.counts[key]["fp"] += torch.sum(preds & ~gts).item()
            self.counts[key]["fn"] += torch.sum(~preds & gts).item()

    def compute(self) -> Dict[Tuple[str, str, str], MetricReturnType]:
        raise NotImplementedError

    def get_axis3(self):
        raise NotImplementedError  # Not needed currently for summary metrics


class Accuracy(_BinaryClassificationMetric):
    """Calculate the accuracy for binary classification.

    Returns:
        Dict[Tuple[str, str, str, str, str], MetricReturnType]: Dictionary with key
        (component, algorithm, person, camera, "accuracy") and the corresponding
        aggregated accuracy value.
    """

    def compute(self) -> Dict[Tuple[str, str, str, str, str], MetricReturnType]:
        results = {}
        for (component, algorithm, person, camera), counts in self.counts.items():
            denominator = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
            value = (
                (counts["tp"] + counts["tn"]) / denominator if denominator > 0 else 0.0
            )
            key = (component, algorithm, person, camera, "accuracy")
            results[key] = AggregatedResult(
                value=value,
                metric_type="categorical_metrics",
                metric_name="accuracy",
                component=component,
                algorithm=algorithm,
                person=person,
                camera=camera,
            )
        return results


class Precision(_BinaryClassificationMetric):
    """Calculate the precision for binary classification.

    Returns:
        Dict[Tuple[str, str, str, str, str], MetricReturnType]: Dictionary with key
        (component, algorithm, person, camera, "precision") and the corresponding
        aggregated precision value.
    """

    def compute(self) -> Dict[Tuple[str, str, str, str, str], MetricReturnType]:
        results = {}
        for (component, algorithm, person, camera), counts in self.counts.items():
            denominator = counts["tp"] + counts["fp"]
            value = counts["tp"] / denominator if denominator > 0 else 0.0
            key = (component, algorithm, person, camera, "precision")
            results[key] = AggregatedResult(
                value=value,
                metric_type="categorical_metrics",
                metric_name="precision",
                component=component,
                algorithm=algorithm,
                person=person,
                camera=camera,
            )
        return results


class Recall(_BinaryClassificationMetric):
    """Calculate the recall for binary classification.

    Returns:
        Dict[Tuple[str, str, str, str, str], MetricReturnType]: Dictionary with key
        (component, algorithm, person, camera, "recall") and the corresponding
        aggregated recall value.
    """

    def compute(self) -> Dict[Tuple[str, str, str, str, str], MetricReturnType]:
        results = {}
        for (component, algorithm, person, camera), counts in self.counts.items():
            denominator = counts["tp"] + counts["fn"]
            value = counts["tp"] / denominator if denominator > 0 else 0.0
            key = (component, algorithm, person, camera, "recall")
            results[key] = AggregatedResult(
                value=value,
                metric_type="categorical_metrics",
                metric_name="recall",
                component=component,
                algorithm=algorithm,
                person=person,
                camera=camera,
            )
        return results


class F1Score(_BinaryClassificationMetric):
    """Calculate the F1 score for binary classification.

    Returns:
        Dict[Tuple[str, str, str, str, str], MetricReturnType]: Dictionary with key
        (component, algorithm, person, camera, "f1_score") and the corresponding
        aggregated F1 score value.
    """

    def compute(self) -> Dict[Tuple[str, str, str, str, str], MetricReturnType]:
        results = {}
        for (component, algorithm, person, camera), counts in self.counts.items():
            precision = (
                counts["tp"] / (counts["tp"] + counts["fp"])
                if (counts["tp"] + counts["fp"]) > 0
                else 0.0
            )
            recall = (
                counts["tp"] / (counts["tp"] + counts["fn"])
                if (counts["tp"] + counts["fn"]) > 0
                else 0.0
            )
            denominator = precision + recall
            f1_score = (
                (2 * precision * recall / denominator) if denominator > 0 else 0.0
            )
            key = (component, algorithm, person, camera, "f1_score")
            results[key] = AggregatedResult(
                value=f1_score,
                metric_type="categorical_metrics",
                metric_name="f1_score",
                component=component,
                algorithm=algorithm,
                person=person,
                camera=camera,
            )
        return results
