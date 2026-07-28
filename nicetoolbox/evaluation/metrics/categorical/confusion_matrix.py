import logging

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from nicetoolbox_core.data.array_schema import BOOLEAN_NAN

from ....configs.schemas.evaluation_metrics_config import ConfusionMatrixConfig
from ...data.input_loader import align_arrays, get_meta_type, load_input
from ...data.plots import plot_confusion_matrix_grid
from ...data.summary import pair_arrays_to_df, resolve_group_levels, split_aligned_arrays
from ..base_metric import BaseMetric
from ..metric_result import FrameResult, MetricResult, PlotResult, SummaryResult


class ConfusionMatrixMetric(BaseMetric):
    """Binary confusion matrix with precision, recall, F1 per pre-compute group.

    Pools all pred/gt boolean values within each group, then runs sklearn's
    confusion_matrix once per group. This avoids Simpson's paradox from
    averaging per-pair F1s across groups with different support.

    Input arrays must have bool dtype (or castable to bool). Confidence floats
    should use a separate metric — routing floats here silently corrupts results.
    """

    metric_config: ConfusionMatrixConfig

    def compute(self) -> MetricResult:
        # load gt and labels and align them frame by frame
        preds = load_input(self.metric_config.predictions, schema=BOOLEAN_NAN)
        gt = load_input(self.metric_config.ground_truth, schema=BOOLEAN_NAN)
        pairs = align_arrays(preds, gt, self.metric_config.broadcast_single)
        if not pairs:
            raise ValueError(
                f"Failed to compute Confusion Matrix '{self.metric_name}': no aligned pred/GT pairs found."
            )

        # convert pred and gt into one dataframe
        df = pair_arrays_to_df(pairs)

        # drop nan
        n_before = len(df)
        df = df.dropna(subset=["pred", "gt"])
        n_dropped = n_before - len(df)
        if n_dropped:
            logging.warning(
                f"[{self.metric_name}] Dropped {n_dropped}/{n_before} rows with NaN pred or gt "
                f"({100 * n_dropped / n_before:.1f}%). Missing predictions are excluded, not counted as wrong."
            )

        # figure out group by levels (always iterate + what user selected)
        meta_type = get_meta_type([p for p, _ in pairs])
        group_levels = resolve_group_levels(df, meta_type, self.metric_config.compute_group_by)

        # generate summary
        summary = df.groupby(level=group_levels, observed=True).apply(self._compute_cm_row).reset_index()

        # optional visualization
        figures = {}
        if self.metric_config.visualize:
            figures = plot_confusion_matrix_grid(summary, group_levels, self.metric_name)

        pred_aligned, gt_aligned = split_aligned_arrays(*pairs)
        return MetricResult(
            self.metric_name,
            frames=FrameResult({"predictions": pred_aligned, "ground_truth": gt_aligned}),
            summary=SummaryResult({"summary": summary}),
            plots=PlotResult(figures),
        )

    @staticmethod
    def _compute_cm_row(group: pd.DataFrame) -> pd.Series:
        y_true = group["gt"].to_numpy(dtype=bool)
        y_pred = group["pred"].to_numpy(dtype=bool)

        cm = confusion_matrix(y_true, y_pred, labels=[False, True])
        tn, fp, fn, tp = cm.ravel()

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        support = int(len(y_true))
        accuracy = float(tp + tn) / support if support else float("nan")

        if precision == 0.0 and recall == 0.0:
            logging.warning(
                "Degenerate group (all-negative or all-positive): precision and recall are both 0. "
                "Support=%d, positives=%d",
                support,
                int(y_true.sum()),
            )

        return pd.Series(
            {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "accuracy": accuracy,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
