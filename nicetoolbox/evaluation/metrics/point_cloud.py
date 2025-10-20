"""
Point cloud metrics for evaluating 3D point cloud data.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from ..data.discovery import ChunkWorkItem, FrameInfo
from .base_metric import Metric, MetricHandler
from .results_schema import BatchResult, MetricReturnType


class PointCloudMetric(MetricHandler):
    """
    Handler for all point cloud metrics.
    """

    @property
    def name(self) -> str:
        return "point_cloud_metrics"

    def _create_metrics(self) -> Dict[str, Metric]:
        """Instantiates the individual metric objects this handler manages."""
        metric_map = {
            "jpe": Jpe
            # ...
        }
        return {
            name: metric_map[name]()  # Metric class is instantiated here
            for name in self.cfg.metric_names
            if name in metric_map
        }


class Jpe(Metric):
    """Calculates the per joint position error for each frame."""

    def reset(self) -> None:
        """Reset the metric's internal state."""
        self.storage: Dict[Tuple[str, str, str], List[BatchResult]] = defaultdict(list)

    def get_axis3(self, chunk: ChunkWorkItem) -> List[str]:
        """Get the joint names the metric is concerned with."""
        # If reconciliation happened, the map tells us which keypoints were used.
        if "axis3" in chunk.pred_reconciliation_map:
            label_indices = chunk.pred_reconciliation_map["axis3"]
            pred_axis3 = chunk.pred_data_description_axis3
            return [pred_axis3[idx] for idx in label_indices]
        # Otherwise, all input keypoints were used.
        return chunk.pred_data_description_axis3

    def update(
        self,
        preds: torch.Tensor,
        gts: torch.Tensor,
        meta_chunk: ChunkWorkItem,
        meta_frames: List[FrameInfo],
    ) -> None:
        """Computes the per joint position error and stores it."""
        if gts is None:
            return

        error = torch.linalg.norm(preds - gts, dim=-1)  # L2 norm per joint

        comp, algo = meta_chunk.component, meta_chunk.algorithm
        key = (comp, algo, "jpe")
        description = self.get_axis3(meta_chunk)
        self.storage[key].append(
            BatchResult(error.cpu(), description, meta_chunk, meta_frames)
        )

    def compute(self) -> Dict[Tuple[str, str, str], MetricReturnType]:
        """Compute the final metric from the stored state."""
        return self.storage
