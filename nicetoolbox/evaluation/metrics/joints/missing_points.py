import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF_PER_LABEL, AnyOf
from nicetoolbox_core.data.loaded_array import NpzArrayWithMeta

from ....configs.schemas.evaluation_aggr import AggSpec
from ....configs.schemas.evaluation_group_by import GroupBySpec
from ....configs.schemas.evaluation_metrics_config import MissingPointsConfig
from ...data.input_loader import get_meta_type, load_input
from ...data.plots import plot_candle_per_group, plot_score, plot_score_heatmap
from ...data.summary import aggregate_summary, summarize_with_group_by
from ..base_metric import BaseMetric
from ..metric_result import FrameResult, MetricResult, PlotResult, SummaryResult


class MissingPointsMetric(BaseMetric):
    """Detect missing joints/landmarks per frame as a binary flag.

    A point is considered missing when its spatial coordinates are all NaN.
    Optionally, points whose confidence score falls below ``min_confidence``
    are also treated as missing (requires a confidence in axis4).

    Output shape: (subjects, cameras, frames, labels) — float32, 1.0 = missing, 0.0 = present.
    Summary and score are expressed as detection rate (1 - missing rate), so higher is better.
    """

    metric_config: MissingPointsConfig
    input_schema = AnyOf(VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF_PER_LABEL)

    def compute(self) -> MetricResult:
        arrays = load_input(self.metric_config.predictions, schema=self.input_schema)
        meta = get_meta_type(arrays)

        # counting missing points in dataset
        # first we save npz_key for each point per frame
        # next we calculate percantage of missing points per frame (convenience)
        missing_arrays: list[NpzArrayWithMeta] = []
        detected_pct_arrays: list[NpzArrayWithMeta] = []
        confidence_arrays: list[NpzArrayWithMeta] = []
        for arr in arrays:
            missing = self._compute_missing(arr)
            missing_arrays.append(missing)
            detected_pct = self._compute_detected_pct(missing)
            detected_pct_arrays.append(detected_pct)
            confidence = self._extract_confidence(arr)
            confidence_arrays.append(confidence)

        # generate users summary based on selected aggregations
        summary = summarize_with_group_by(
            missing_arrays,
            self.metric_config.missing_points_summary_group_by,
            self.metric_config.missing_points_summary_aggr,
        )

        # calculate final score - detection rate per desired group by
        # aggregate it for final comparing
        score_summary = summarize_with_group_by(
            missing_arrays,
            GroupBySpec(dims=["label"]),
            AggSpec.of_type(one_minus_mean="detection_rate"),
        )
        score = aggregate_summary(score_summary, agg_col="detection_rate", meta_type=meta)

        # optional visualization
        figures = {}
        if self.metric_config.visualize:
            compare_dim = meta.comparable_dim()

            # overall score bar chart
            figures["detection_rate_score"] = plot_score(
                score,
                x_col=compare_dim,
                y_col="detection_rate",
                title="Missing Points Detection Rate Score",
                x_label=compare_dim,
                y_label="Detection Rate (higher is better)",
            )

            # confidence distribution per joint — one figure per algorithm
            figures |= plot_candle_per_group(
                confidence_arrays,
                x_col="label",
                base_title="Points Confidence Distribution",
                y_label="Confidence",
                split_by=GroupBySpec(dims=[compare_dim] if compare_dim else []),
            )

            # detection rate heatmap: joints on X, algorithms on Y
            figures["detection_rate_per_point"] = plot_score_heatmap(
                score_summary,
                x_col="label",
                y_col=compare_dim,
                value_col="detection_rate",
                title="Missing Points Detection Rate per Joint",
                x_label="Joint",
                y_label=compare_dim,
            )

        return MetricResult(
            metric_name=self.metric_name,
            frames=FrameResult(
                arrays={
                    "missing": missing_arrays,
                    "detected_pct": detected_pct_arrays,
                    "confidence": confidence_arrays,
                    "predictions": arrays,
                }
            ),
            plots=PlotResult(figures),
            summary=SummaryResult({"detection_rate_score": score, "summary": summary}),
        )

    def _compute_missing(self, arr: NpzArrayWithMeta) -> NpzArrayWithMeta:
        """Compute per-frame missing flag for each joint in a single loaded array.

        Returns a MetaNpzArray with shape (subjects, cameras, frames, labels),
        values are 1.0 where the point is missing and 0.0 where it is present.
        """
        data = arr.data  # (subjects, cameras, frames, labels, data)

        # missing if ANY coord of a joint is NaN
        # e.g. joint with coords [1.2, NaN, 0.5] -> missing = True
        missing = np.any(np.isnan(data), axis=-1)  # (S, C, F, labels)

        # --- Confidence mask (optional) ---
        # Confidence is always the last coordinate (e.g. [..., x, y, z, confidence_score])
        if self.metric_config.min_confidence is not None:
            conf = data[..., -1]  # (S, C, F, labels)
            missing = missing | (conf < self.metric_config.min_confidence)

        result_data = missing.astype(np.float32)
        # missing collapses the coordinate axis, so drop axis4 (data) labels.
        result_axes = arr.axes.make_4d()
        return arr.replace(data=result_data, axes=result_axes)

    def _extract_confidence(self, arr: NpzArrayWithMeta) -> NpzArrayWithMeta:
        """Extract the confidence coordinate (last coord) as a (S, C, F, fields) array."""
        return arr.replace(data=arr.data[..., -1])

    def _compute_detected_pct(self, missing: NpzArrayWithMeta) -> NpzArrayWithMeta:
        """Compute per-frame fraction of detected (non-missing) joints from the binary missing array."""
        detected_data = 1.0 - missing.data.mean(axis=-1, keepdims=True)  # (S, C, F, 1)
        pct_axes = missing.axes.replace(labels=["detected_pct"])
        return missing.replace(data=detected_data, axes=pct_axes)
