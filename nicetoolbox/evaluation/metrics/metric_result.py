from dataclasses import dataclass, field

import pandas as pd
from matplotlib import pyplot as plt

from nicetoolbox_core.data.loaded_array import NpzArrayWithMeta


@dataclass
class SummaryResult:
    """Tabular results written to CSV.

    Attributes:
        summaries: Named DataFrames; each key becomes a separate CSV file.
    """

    summaries: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class FrameResult:
    """Per-frame result stored as NPZ files.

    Arrays at the same index across all keys belong to the same NPZ file.

    Attributes:
        arrays: Maps NPZ key name to a list of MetaNpzArrays (one per sequence/algorithm).
    """

    arrays: dict[str, list[NpzArrayWithMeta]] = field(default_factory=dict)


@dataclass
class PlotResult:
    """Matplotlib figures to save as PNG.

    Attributes:
        figures: Named figures; each key becomes a separate PNG file.
    """

    figures: dict[str, plt.Figure] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Complete output from a single metric run.

    Attributes:
        metric_name: Identifier used to name the output directory.
        summary: Optional tabular results written to CSV.
        frames: Optional per-frame arrays written to NPZ.
        plots: Optional figures written to PNG.
    """

    metric_name: str
    summary: SummaryResult | None = None
    frames: FrameResult | None = None
    plots: PlotResult | None = None
