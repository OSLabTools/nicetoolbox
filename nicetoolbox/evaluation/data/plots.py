import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from nicetoolbox_core.data.loaded_array import NpzArrayWithMeta

from ...configs.schemas.evaluation_group_by import GroupBySpec

_NO_SPLIT = GroupBySpec(dims=[])
_TAB10 = plt.cm.tab10.colors


class _LabelColorMap:
    """Assigns a unique tab10 color to each label on first encounter."""

    def __init__(self):
        self._map: dict[str, tuple] = {}
        self._counter = 0

    def __call__(self, label: str) -> tuple:
        if label not in self._map:
            self._map[label] = _TAB10[self._counter % len(_TAB10)]
            self._counter += 1
        return self._map[label]


_label_color = _LabelColorMap()


def _plot_candle_single(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    y_label: str,
    series_col: str | None = None,
) -> plt.Figure:
    """Render one candle chart for a single (subject, sequence, ...) slice.

    When series_col is None, renders all data as a single color with no legend.
    """
    x_values = df[x_col].unique()
    n_x = len(x_values)
    fig, ax = plt.subplots(figsize=(max(8, n_x * 1.2), 6))
    x_positions = np.arange(n_x)

    if series_col is None:
        series_list = [(None, df, _TAB10[0])]
        width = 0.8
    else:
        series = df[series_col].unique()
        n_series = len(series)
        width = 0.8 / n_series
        series_list = [(s, df[df[series_col] == s], _label_color(s)) for s in series]

    for i, (label, series_df, color) in enumerate(series_list):
        series_df = series_df.set_index(x_col)
        if series_col is not None:
            n_series = len(series_list)
            offsets = x_positions + (i - n_series / 2 + 0.5) * width
        else:
            offsets = x_positions

        for j, x_val in enumerate(x_values):
            if x_val not in series_df.index:
                continue
            row = series_df.loc[x_val]
            low, high = row["min"], row["max"]
            q25, q75, median = row["q25"], row["q75"], row["median"]

            cap = width * 0.3
            ax.plot([offsets[j], offsets[j]], [low, high], color=color, linewidth=1)
            ax.plot([offsets[j] - cap, offsets[j] + cap], [low, low], color=color, linewidth=1)
            ax.plot([offsets[j] - cap, offsets[j] + cap], [high, high], color=color, linewidth=1)
            ax.bar(offsets[j], q75 - q25, bottom=q25, width=width * 0.9, color=color, alpha=0.6)
            ax.plot([offsets[j] - width * 0.4, offsets[j] + width * 0.4], [median, median], color=color, linewidth=2.5)

        if label is not None:
            ax.plot([], [], color=color, label=label)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_values, rotation=45, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title, wrap=True)
    if series_col is not None:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_score(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> plt.Figure:
    """Bar chart comparing a scalar score across groups (e.g. algorithm vs CV).

    If multiple rows share the same x_col value, their y_col values are averaged.
    """
    score = df.groupby(x_col)[y_col].mean().sort_values()

    fig, ax = plt.subplots(figsize=(max(6, len(score) * 1.2), 5))
    bars = ax.bar(range(len(score)), score.values, color=[_label_color(label) for label in score.index])
    ax.set_xticks(range(len(score)))
    ax.set_xticklabels(score.index, rotation=45, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    return fig


def plot_score_grouped(
    df: pd.DataFrame,
    x_col: str,
    series_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> plt.Figure:
    """Grouped bar chart comparing a scalar score across x groups, with one color per series.

    If multiple rows share the same (x_col, series_col) pair, their y_col values are averaged.
    """
    df = df.groupby([x_col, series_col])[y_col].mean().reset_index()
    x_values = df[x_col].unique()
    series = df[series_col].unique()
    n_x = len(x_values)
    n_series = len(series)

    x_positions = np.arange(n_x)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(max(8, n_x * 1.2), 5))
    for i, s in enumerate(series):
        series_df = df[df[series_col] == s].set_index(x_col)
        offsets = x_positions + (i - n_series / 2 + 0.5) * width
        values = [series_df.loc[x, y_col] if x in series_df.index else float("nan") for x in x_values]
        bars = ax.bar(offsets, values, width=width * 0.9, color=_label_color(s), label=s)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_values, rotation=45, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_score_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> plt.Figure:
    """Heatmap of scalar scores with x_col on X axis and y_col on Y axis.

    If multiple rows share the same (x_col, y_col) pair, their values are averaged.
    Each cell is annotated with its numeric value.
    """
    pivot = df.groupby([y_col, x_col])[value_col].mean().unstack(x_col)

    n_x = pivot.shape[1]
    n_y = pivot.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n_x * 0.7), max(3, n_y * 0.6)))

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    for i in range(n_y):
        for j in range(n_x):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xticks(range(n_x))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(n_y))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_frame_line(
    arrays: list[NpzArrayWithMeta],
    base_title: str,
    y_label: str,
    series_col: str | None = None,
    split_by: GroupBySpec = _NO_SPLIT,
) -> dict[str, plt.Figure]:
    """Generate per-frame line charts from raw arrays.

    For each split_by group, produces a line chart with frames on X axis
    and mean metric value (across subjects, cameras, fields) on Y axis.
    Different series (e.g. algorithms) are shown as separate colored lines.

    Returns a dict mapping filename -> figure.
    """
    from .summary import arrays_to_dataframe

    df = arrays_to_dataframe(arrays)

    # Determine which columns to group by for each line
    extra = [series_col] if series_col else []
    present_split = split_by.intersect(df.columns) if split_by.dims else []

    # Group by split + series + frame, average over the rest
    group_cols = present_split + extra + ["frame"]
    agg_df = df.groupby(group_cols)["value"].mean().reset_index()

    if not present_split:
        fig = _plot_frame_line_single(agg_df, base_title, y_label, series_col)
        return {f"{base_title}_per_frame": fig}

    figures: dict[str, plt.Figure] = {}
    for keys, group_df in agg_df.groupby(present_split):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = " | ".join(f"{c}={v}" for c, v in zip(present_split, keys))
        title = f"{base_title} — {label}"
        filename = f"{base_title}_per_frame_{'_'.join(str(v) for v in keys)}"
        figures[filename] = _plot_frame_line_single(group_df, title, y_label, series_col)

    return figures


def _plot_frame_line_single(
    df: pd.DataFrame,
    title: str,
    y_label: str,
    series_col: str | None = None,
) -> plt.Figure:
    """Render one per-frame line chart."""
    fig, ax = plt.subplots(figsize=(12, 5))

    df = df.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")

    if series_col is None or series_col not in df.columns:
        df = df.sort_values("frame")
        ax.plot(df["frame"], df["value"], color=_TAB10[0], linewidth=0.8, alpha=0.8)
    else:
        for name, group in df.groupby(series_col):
            group = group.sort_values("frame")
            ax.plot(group["frame"], group["value"], color=_label_color(name), label=name, linewidth=0.8, alpha=0.8)
        ax.legend()

    ax.set_xlabel("Frame")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_candle_per_group(
    arrays: list[NpzArrayWithMeta],
    x_col: str,
    base_title: str,
    y_label: str,
    series_col: str | None = None,
    split_by: GroupBySpec = _NO_SPLIT,
) -> dict[str, plt.Figure]:
    """Generate one candle chart per unique combination of split_by columns.

    Computes candle statistics (min, q25, median, q75, max) from raw arrays.
    group_by is derived automatically as split_by + x_col + series_col (if provided).

    Returns a dict mapping filename -> figure.
    """
    from ...configs.schemas.evaluation_aggr import AggSpec
    from .summary import summarize_with_group_by

    extra = [series_col] if series_col else []
    group_by = GroupBySpec(dims=(split_by.dims or []) + [x_col] + extra)
    df = summarize_with_group_by(arrays, group_by=group_by, agg=AggSpec.of_type("min", "q25", "median", "q75", "max"))

    if series_col and series_col not in df.columns:
        logging.warning(
            f"Candle chart skipped for '{base_title}': "
            f"series column '{series_col}' not found in data. Available columns: {list(df.columns)}"
        )
        return {}

    present = split_by.intersect(df.columns)
    if not present:
        fig = _plot_candle_single(df, x_col, base_title, y_label, series_col)
        return {f"{base_title}": fig}

    figures: dict[str, plt.Figure] = {}
    for keys, group_df in df.groupby(present):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = " | ".join(f"{c}={v}" for c, v in zip(present, keys))
        title = f"{base_title} — {label}"
        filename = f"{base_title}_{'_'.join(str(v) for v in keys)}"
        figures[filename] = _plot_candle_single(group_df, x_col, title, y_label, series_col)

    return figures


def plot_roc_curves(
    curves: list[tuple[str, np.ndarray, np.ndarray, float]],
    base_title: str,
) -> plt.Figure:
    """ROC curve figure with one line per group.

    Args:
        curves: list of (label, fpr, tpr, auc) tuples, one per group row.
        base_title: figure suptitle.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for label, fpr, tpr, auc in curves:
        ax.plot(fpr, tpr, color=_label_color(label), linewidth=1.5, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    ax.set_title(base_title)
    fig.tight_layout()
    return fig


def plot_pr_curves(
    curves: list[tuple[str, np.ndarray, np.ndarray, float, float]],
    base_title: str,
) -> plt.Figure:
    """Precision-recall curve figure with one line per group.

    Each group also gets a dashed horizontal baseline at its positive class
    prevalence, which is what a random classifier achieves at every threshold.

    Args:
        curves: list of (label, recall, precision, ap, prevalence) tuples, one per group row.
        base_title: figure suptitle.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    for label, recall, precision, ap, prevalence in curves:
        color = _label_color(label)
        ax.plot(recall, precision, color=color, linewidth=1.5, label=f"{label} (AP={ap:.3f})")
        ax.axhline(prevalence, color=color, linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    ax.set_title(base_title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix_grid(
    summary: pd.DataFrame,
    group_levels: list[str],
    base_title: str,
) -> dict[str, plt.Figure]:
    """One figure per row in *summary*: CM heatmap + accuracy/precision/recall/F1 bar chart.

    Returns a dict mapping filename -> figure.
    """
    figures: dict[str, plt.Figure] = {}

    for _, row in summary.iterrows():
        group_parts = [f"{lvl}={row[lvl]}" for lvl in group_levels if lvl in row.index]
        group_label = "_".join(str(row[lvl]) for lvl in group_levels if lvl in row.index)
        subtitle = "\n".join(group_parts) if group_parts else ""
        fig_key = f"cm_{group_label}" if group_label else "cm"

        cm_vals = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4 + 0.2 * len(group_parts)))

        im = ax.imshow(cm_vals, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Neg", "Pred Pos"])
        ax.set_yticklabels(["True Neg", "True Pos"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm_vals[i, j]), ha="center", va="center", fontsize=14)
        ax.set_title(f"Confusion Matrix\n{subtitle}", fontsize=9)

        metric_names = ["accuracy", "precision", "recall", "f1"]
        values = [float(row[m]) for m in metric_names]
        bars = ax2.bar(metric_names, values, color=_TAB10)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("Score")
        ax2.set_title(f"Metrics\n{subtitle}", fontsize=9)
        ax2.bar_label(bars, fmt="%.3f", padding=3)

        fig.suptitle(base_title)
        fig.tight_layout()
        figures[fig_key] = fig

    return figures


def plot_error_composition(
    df: pd.DataFrame,
    group_col: str,
    title: str,
) -> plt.Figure:
    """Stacked bar of substitution/deletion/insertion *rates* (count / reference words) per group.

    Shows what kind of errors dominate (e.g. deletion-heavy = dropped speech, insertion-heavy =
    hallucination). Expects `substitutions`/`deletions`/`insertions`/`ref_words` count
    columns (as produced by the transcription metric's summary).
    """
    parts = ["substitutions", "deletions", "insertions"]
    grp = df.groupby(group_col)[parts + ["ref_words"]].sum()
    ref = grp["ref_words"].where(grp["ref_words"] != 0, np.nan)
    rates = pd.DataFrame({p: (grp[p] / ref).fillna(0.0) for p in parts})

    labels = list(rates.index)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))

    bottom = np.zeros(len(labels))
    for i, part in enumerate(parts):
        values = rates[part].to_numpy()
        bars = ax.bar(x, values, bottom=bottom, label=part, color=_TAB10[i])
        ax.bar_label(bars, fmt="%.3f", label_type="center", fontsize=7)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(group_col)
    ax.set_ylabel("errors per reference word")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
