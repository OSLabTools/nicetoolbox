import logging
from pathlib import Path

from matplotlib import pyplot as plt

from nicetoolbox_core.data.loaded_array import NpzArray, save_arrays
from nicetoolbox_core.data.npz_meta import AnnotationMeta, ExperimentMeta, NpzMeta, PathMeta

from ...utils.to_csv import results_to_csv
from ..metrics.metric_result import FrameResult, MetricResult, PlotResult, SummaryResult


def save_results(result: MetricResult, output_dir: Path) -> None:
    """Save all metric results (frames, summaries, plots) to disk.

    Args:
        result: Complete metric output containing optional frames, summaries, and plots.
        output_dir: Root output directory; a subdirectory named after the metric is created.
    """
    metric_dir = output_dir / result.metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)

    if result.frames is not None:
        save_frame_arrays(result.frames, metric_dir)

        csv_dir = metric_dir / "_csv_export"
        csv_dir.mkdir(parents=True, exist_ok=True)
        results_to_csv(metric_dir, csv_dir)  # TODO: only one level depth - fix it

    if result.summary is not None:
        save_summary_csv(result.summary, metric_dir)

    if result.plots is not None:
        save_plots(result.plots, metric_dir)


def save_summary_csv(summary: SummaryResult, metric_dir: Path) -> None:
    """Save each named summary DataFrame to its own CSV file."""
    if not summary.summaries:
        logging.warning(f"No summaries in '{metric_dir.name}', skipping CSV.")
        return

    for name, df in summary.summaries.items():
        if df.empty:
            logging.warning(f"Summary '{name}' in '{metric_dir.name}' is empty, skipping.")
            continue
        path = metric_dir / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Saved summary CSV: {path}")


def save_plots(plot_result: PlotResult, metric_dir: Path) -> None:
    """Save each named figure as a PNG under metric_dir/visualization/."""
    for name, fig in plot_result.figures.items():
        path = metric_dir / "visualization" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)  # release the figure so long runs don't accumulate open figures
        logging.info(f"Saved plot: {path}")


def save_frame_arrays(frame_result: FrameResult, metric_dir: Path) -> None:
    """Save per-frame result arrays as NPZ files under metric_dir/npz/."""
    keys = list(frame_result.arrays.keys())
    if not keys:
        return

    for arrs in zip(*[frame_result.arrays[k] for k in keys]):
        first = arrs[0]
        rel_path = _build_output_path(first.meta)
        out_path = metric_dir / "npz" / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {key: NpzArray(data=arr.data, axes=arr.axes) for key, arr in zip(keys, arrs)}
        save_arrays(arrays, out_path)


def _build_output_path(meta: NpzMeta) -> Path:
    """Build relative output path from meta fields."""
    if isinstance(meta, (ExperimentMeta)):
        subsequence = f"{meta.sequence}_{meta.subsequence.subsequence_index}"
        return Path(meta.dataset) / subsequence / meta.component / f"{meta.algorithm}.npz"
    if isinstance(meta, AnnotationMeta):
        return Path(meta.dataset) / meta.sequence / f"{meta.component}.npz"
    if isinstance(meta, PathMeta):
        return Path(meta.npz_path.stem + ".npz")
    raise ValueError(f"Unknown meta structure: {meta}")
