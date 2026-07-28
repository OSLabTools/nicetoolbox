import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from nicetoolbox_core.data.array_schema import BOOLEAN_NAN, FLOAT

from ....configs.schemas.evaluation_metrics_config import PrCurveConfig
from ...data.input_loader import align_arrays, get_meta_type, load_input
from ...data.plots import plot_pr_curves
from ...data.summary import pair_arrays_to_df, resolve_group_levels, split_aligned_arrays
from ..base_metric import BaseMetric
from ..metric_result import FrameResult, MetricResult, PlotResult, SummaryResult


class PrCurveMetric(BaseMetric):
    """Precision-recall curve and average precision (AP) per pre-compute group.

    Predictions must be float confidence scores; ground truth must be bool.
    AP is computed once per group by pooling all frames — same rationale as
    RocAucMetric (avoids averaging per-pair scores across groups with
    different support and class balance).
    """

    metric_config: PrCurveConfig

    def compute(self) -> MetricResult:
        preds = load_input(self.metric_config.predictions, schema=FLOAT)
        gt = load_input(self.metric_config.ground_truth, schema=BOOLEAN_NAN)
        pairs = align_arrays(preds, gt, self.metric_config.broadcast_single)
        if not pairs:
            raise ValueError(f"Failed to compute PR curve '{self.metric_name}': no aligned pred/GT pairs found.")

        df = pair_arrays_to_df(pairs)

        n_before = len(df)
        df = df.dropna(subset=["pred", "gt"])
        n_dropped = n_before - len(df)
        if n_dropped:
            logging.warning(
                f"[{self.metric_name}] Dropped {n_dropped}/{n_before} rows with NaN pred or gt "
                f"({100 * n_dropped / n_before:.1f}%). Missing predictions are excluded."
            )

        _validate_pr_inputs(df, self.metric_name)

        meta_type = get_meta_type([p for p, _ in pairs])
        group_levels = resolve_group_levels(df, meta_type, self.metric_config.compute_group_by)

        always = meta_type.always_iterate()
        label_levels = [lvl for lvl in group_levels if lvl not in always] or group_levels
        # drop dimensions that are constant across all groups — they add length without distinguishing curves
        label_levels = [lvl for lvl in label_levels if df.index.get_level_values(lvl).nunique() > 1] or label_levels

        records: list[dict] = []
        curves: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []

        for keys, group in df.groupby(level=group_levels, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)

            key_map = dict(zip(group_levels, keys))
            y_true = group["gt"].to_numpy(dtype=bool)
            y_score = group["pred"].to_numpy(dtype=float)
            if self.metric_config.negate_scores:
                y_score = -y_score

            prevalence = float(y_true.mean())

            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                logging.warning(
                    "[%s] Skipping degenerate group %s — only one class present (positives=%d/%d).",
                    self.metric_name,
                    keys,
                    int(y_true.sum()),
                    len(y_true),
                )
                ap = float("nan")
                optimal_threshold = float("nan")
                precision_pts = recall_pts = np.array([1.0, 0.0])
            else:
                precision_pts, recall_pts, thresholds = precision_recall_curve(y_true, y_score)
                ap = float(average_precision_score(y_true, y_score))
                # Optimal threshold: maximise F1 over the threshold-aligned prefix (exclude the
                # appended (precision=1, recall=0) sentinel that has no corresponding threshold).
                with np.errstate(invalid="ignore"):
                    f1 = 2 * precision_pts[:-1] * recall_pts[:-1] / (precision_pts[:-1] + recall_pts[:-1])
                f1 = np.nan_to_num(f1)
                optimal_threshold = float(thresholds[np.argmax(f1)])
                if self.metric_config.negate_scores:
                    optimal_threshold = -optimal_threshold

            label = " | ".join(f"{lvl}={key_map[lvl]}" for lvl in label_levels)
            curves.append((label, recall_pts, precision_pts, ap, prevalence))
            records.append({**key_map, "ap": ap, "optimal_threshold": optimal_threshold, "support": len(y_true)})

        summary = pd.DataFrame(records)

        figures = {}
        if self.metric_config.visualize:
            figures["pr_curves"] = plot_pr_curves(curves, self.metric_name)

        pred_aligned, gt_aligned = split_aligned_arrays(*pairs)
        return MetricResult(
            self.metric_name,
            frames=FrameResult({"predictions": pred_aligned, "ground_truth": gt_aligned}),
            summary=SummaryResult({"summary": summary}),
            plots=PlotResult(figures),
        )


def _validate_pr_inputs(df: pd.DataFrame, metric_name: str) -> None:
    if not pd.api.types.is_bool_dtype(df["gt"]) and not pd.api.types.is_float_dtype(df["gt"]):
        raise TypeError(f"[{metric_name}] Column 'gt' must be bool or float, got {df['gt'].dtype}.")
    unique = df["gt"].unique()
    if not set(unique).issubset({0.0, 1.0}):
        raise TypeError(f"[{metric_name}] Column 'gt' must contain only 0/1 or True/False, got {unique}.")
    if not pd.api.types.is_float_dtype(df["pred"]):
        raise TypeError(
            f"[{metric_name}] Column 'pred' must be float (confidence scores), got {df['pred'].dtype}. "
            "Bool predictions belong in confusion_matrix, not pr_curve."
        )
