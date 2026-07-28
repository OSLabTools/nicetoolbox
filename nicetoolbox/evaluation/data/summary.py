import pandas as pd

from nicetoolbox_core.data.loaded_array import NpzArrayWithMeta
from nicetoolbox_core.data.npz_meta import NpzMeta

from ...configs.schemas.evaluation_aggr import AggSpec
from ...configs.schemas.evaluation_group_by import GroupBySpec
from .input_loader import get_meta_type

_AXIS_LEVELS = {"subject", "camera", "frame", "label", "data"}


def split_aligned_arrays(*pairs: tuple[NpzArrayWithMeta, ...]) -> list[list[NpzArrayWithMeta]]:
    """Split N-tuples of MetaNpzArrays into N separate lists, mirroring zip semantics."""
    return [list(group) for group in zip(*pairs)]


def pair_arrays_to_df(
    pairs: list[tuple[NpzArrayWithMeta, NpzArrayWithMeta]],
    value_names: tuple[str, str] = ("pred", "gt"),
) -> pd.DataFrame:
    """Join paired prediction/ground-truth arrays into a single long-format DataFrame.

    Args:
        pairs (list): List of (prediction, ground_truth) MetaNpzArray pairs as returned by
            align_arrays.
        value_names: Column names for the prediction and ground truth values respectively.

    Returns:
        Long-format DataFrame with one row per (meta..., subject, camera, frame, label)
        coordinate, or an empty DataFrame if pairs is empty.

    Raises:
        ValueError: If the axis merge loses rows, indicating a violated align_arrays
            invariant.
    """
    pred_name, gt_name = value_names
    parts: list[pd.DataFrame] = []
    for pred, gt in pairs:
        pred_df = arrays_to_dataframe([pred]).rename(columns={"value": pred_name}).reset_index()
        gt_df = arrays_to_dataframe([gt]).rename(columns={"value": gt_name}).reset_index()

        axis_cols = [c for c in pred_df.columns if c in _AXIS_LEVELS]
        merged = pred_df.merge(gt_df[axis_cols + [gt_name]], on=axis_cols, how="inner")

        if len(merged) != len(pred_df):
            raise ValueError(
                f"Axis merge lost rows: pred={len(pred_df)}, merged={len(merged)}. "
                "align_arrays invariant violated — pred and gt must share axis labels per pair."
            )
        struct_cols = [c for c in merged.columns if c not in set(value_names)]
        parts.append(merged.set_index(struct_cols))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts)


def arrays_to_dataframe(arrays: list[NpzArrayWithMeta]) -> pd.DataFrame:
    """Convert a list of MetaNpzArrays into a single long-format DataFrame.

    Each row represents one scalar value at a specific (meta..., subject,
    camera, frame, label) coordinate.

    Args:
        arrays: MetaNpzArray instances to convert. All arrays must share the same
            meta key structure.

    Returns:
        Long-format DataFrame with a single ``value`` column and a MultiIndex
        built from meta fields and axis labels.
    """
    meta_keys = [tuple(arr.meta.to_dict().keys()) for arr in arrays]
    assert len(set(meta_keys)) <= 1, f"All arrays must have the same meta keys, got: {set(meta_keys)}"

    series_parts: list[pd.Series] = []
    for arr in arrays:
        meta = arr.meta.to_dict()

        # Build index levels: meta fields first, then axes.
        level_names: list[str] = []
        level_values: list[list] = []

        for key, value in meta.items():
            level_names.append(key)
            level_values.append([value])

        # Axes must follow numpy axis order: subject(0), camera(1), frame(2), label(3).
        level_names.append("subject")
        level_values.append(arr.axes.subjects)

        level_names.append("camera")
        level_values.append(arr.axes.cameras)

        level_names.append("frame")
        level_values.append(list(arr.axes.frames))

        level_names.append("label")
        level_values.append(arr.axes.labels)

        idx = pd.MultiIndex.from_product(level_values, names=level_names)
        series_parts.append(pd.Series(arr.data.flatten(), index=idx))

    return pd.concat(series_parts).to_frame(name="value")


def aggregate_summary(summary: pd.DataFrame, agg_col: str, meta_type: type[NpzMeta]) -> pd.DataFrame:
    """Aggregate a summary DataFrame by always-iterate columns, averaging over agg_col.

    Args:
        summary: Summary DataFrame containing metric results per group.
        agg_col: Name of the column to average.
        meta_type: NpzMeta subclass whose always_iterate() defines the grouping keys.

    Returns:
        Aggregated DataFrame with one row per unique combination of grouping columns.

    Raises:
        ValueError: If none of the always-iterate columns are present in summary.
    """
    always_iterate = meta_type.always_iterate()
    group_by = [c for c in always_iterate if c in summary.columns]
    if not group_by:
        raise ValueError(f"Cannot aggregate summary: no grouping columns from {always_iterate} found.")
    return summary.groupby(group_by)[agg_col].mean().reset_index()


def resolve_group_levels(
    df: pd.DataFrame,
    meta_type: type[NpzMeta],
    group_by: GroupBySpec,
    exclude: frozenset[str] = frozenset({"frame"}),
) -> list[str]:
    """Return the columns to group by for a categorical metric computation.

    Always includes always_iterate columns from meta_type that are present in df,
    then adds user-requested dimensions from group_by, minus anything in exclude.

    Args:
        df: DataFrame containing the metric data to be grouped.
        meta_type: NpzMeta subclass whose always_iterate() defines mandatory grouping keys.
        group_by: User-specified grouping dimensions.
        exclude: Column names to never include in the result.

    Returns:
        Ordered list of column names to pass to a groupby call.
    """
    always = meta_type.always_iterate()
    available = [n for n in df.index.names if n is not None]

    user_dims = group_by.resolve([c for c in available if c not in exclude])

    seen: set[str] = set()
    levels: list[str] = []
    for col in list(always) + user_dims:
        if col in available and col not in seen and col not in exclude:
            levels.append(col)
            seen.add(col)
    return levels


def summarize_with_group_by(
    arrays: list[NpzArrayWithMeta],
    group_by: GroupBySpec,
    agg: AggSpec,
) -> pd.DataFrame:
    """Build a summary DataFrame from frame-level arrays, grouped by specified dimensions.

    Meta-level fields are always iterated; axes-level fields are iterated only
    when listed in group_by.

    Args:
        arrays: Frame-level MetaNpzArray instances to summarize.
        group_by: User-specified grouping dimensions applied on top of mandatory
            always-iterate fields.
        agg: Aggregation specification mapping output column names to functions.

    Returns:
        Summary DataFrame with one row per group and one column per aggregation
        function, or an empty DataFrame if arrays is empty.
    """
    if not arrays:
        return pd.DataFrame()

    always_iterate = get_meta_type(arrays).always_iterate()
    df = arrays_to_dataframe(arrays)

    all_levels = list(df.index.names)
    user_group_by = group_by.resolve(all_levels)

    # Always-iterate levels + whatever the user requested.
    group_levels = [lvl for lvl in all_levels if lvl in always_iterate or lvl in user_group_by]

    grouped = df.groupby(level=group_levels, observed=True)["value"]
    return grouped.agg(**agg.resolve()).reset_index()
