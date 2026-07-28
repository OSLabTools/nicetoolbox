from __future__ import annotations

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF_PER_LABEL, AnyOf
from nicetoolbox_core.data.loaded_array import NpzArrayWithMeta

from ....configs.schemas.evaluation_aggr import AggSpec
from ....configs.schemas.evaluation_metrics_config import PCKConfig
from ...data.input_loader import align_arrays, get_meta_type, load_input
from ...data.plots import plot_frame_line, plot_score, plot_score_heatmap
from ...data.summary import aggregate_summary, split_aligned_arrays, summarize_with_group_by
from ..base_metric import BaseMetric
from ..metric_result import FrameResult, MetricResult, PlotResult, SummaryResult


class PCKMetric(BaseMetric):
    """Per-frame, per-joint PCK (Percentage of Correct Keypoints) with a fixed distance threshold.

    A joint is "correct" if its L2 distance to ground truth is within the
    configured threshold (in the same units as the input coords — e.g. pixels for 2D).

    Works with 2D and 3D keypoints; the last coordinate dimension is assumed
    to be confidence and is dropped.
    """

    metric_config: PCKConfig
    input_schema = AnyOf(VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF_PER_LABEL)

    def compute(self) -> MetricResult:
        preds = load_input(self.metric_config.predictions, schema=self.input_schema)
        gt = load_input(self.metric_config.ground_truth, schema=self.input_schema)
        pairs = align_arrays(preds, gt, self.metric_config.broadcast_single)
        if not pairs:
            raise ValueError("Failed to compute PCK: no aligned prediction/GT pairs found!")

        pck_arrays: list[NpzArrayWithMeta] = []
        for pred, gt_arr in pairs:
            pck_arrays.append(self._compute_pck(pred, gt_arr))

        meta = get_meta_type(pck_arrays)

        summary = summarize_with_group_by(
            pck_arrays,
            self.metric_config.summary_group_by,
            self.metric_config.summary_aggr,
        )

        mean_summary = summarize_with_group_by(
            pck_arrays,
            self.metric_config.summary_group_by,
            AggSpec.of_type("mean"),
        )
        score = aggregate_summary(mean_summary, agg_col="mean", meta_type=meta)

        figures = {}
        if self.metric_config.visualize:
            compare_dim = meta.comparable_dim()

            if "label" in mean_summary.columns:
                figures["pck_per_joint_heatmap"] = plot_score_heatmap(
                    mean_summary,
                    x_col="label",
                    y_col=compare_dim,
                    value_col="mean",
                    title=f"{self.metric_name} — PCK per joint",
                    x_label="joint",
                    y_label=compare_dim,
                )

            figures |= plot_frame_line(
                pck_arrays,
                series_col=compare_dim,
                base_title=self.metric_name,
                y_label="PCK",
            )

            figures["pck_score"] = plot_score(
                score,
                x_col=compare_dim,
                y_col="mean",
                title="Mean PCK Score",
                x_label=compare_dim,
                y_label="PCK (higher is better)",
            )

        pred_aligned, gt_aligned = split_aligned_arrays(*pairs)
        return MetricResult(
            self.metric_name,
            frames=FrameResult({"pck": pck_arrays, "predictions": pred_aligned, "ground_truth": gt_aligned}),
            plots=PlotResult(figures),
            summary=SummaryResult({"pck_score": score, "summary": summary}),
        )

    def _compute_pck(self, pred: NpzArrayWithMeta, gt: NpzArrayWithMeta) -> NpzArrayWithMeta:
        """Compute per-frame, per-joint PCK correctness.

        Returns MetaNpzArray with shape (subjects, cameras, frames, joints)
        containing 1.0 (correct) or 0.0 (incorrect) per joint per frame.
        """
        # drop confidence — last coord axis
        p = pred.data[..., :-1]
        g = gt.data[..., :-1]

        # per-joint L2 distance between pred and GT
        dist = np.linalg.norm(p - g, axis=-1)

        correct = (dist <= self.metric_config.threshold).astype(np.float32)

        # correct collapses the coordinate axis, so drop axis4 (data) labels.
        result_axes = pred.axes.make_4d()
        return pred.replace(data=correct, axes=result_axes)
