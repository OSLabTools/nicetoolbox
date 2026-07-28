import glob
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from nicetoolbox_core.data.array_schema import BaseArraySchema
from nicetoolbox_core.data.loaded_array import NpzArrayAxes, NpzArrayWithMeta, load_and_filter_array
from nicetoolbox_core.data.npz_meta import AnnotationMeta, ExperimentMeta, NpzMeta, PathMeta, SubsequenceInfo

from ...configs.placeholders import get_placeholders_str, resolve_placeholders
from ...configs.schemas.detectors_run_file import ResolvedSubsequenceConfig
from ...configs.schemas.evaluation_input_block import AnnotationInput, BaseInputBlock, ExperimentInput, PathInput
from ...configs.schemas.experiment_config import DetectorsExperimentConfig
from ...configs.utils import dict_to_model, get_latest_experiment_config_path, load_raw_config
from ...detectors.main import get_algo_components
from ...utils.logging_utils import abbrev_list

# =============================================================================
# Helpers and utils
# =============================================================================


def matches_filter(value: str, filter_value: str | Any | list[Any]) -> bool:
    """Match a discovered dimension value against an InputBlock filter.

    Args:
        value: The actual dimension value to test (e.g. dataset name, session ID).
        filter_value: Filter to match against. Accepts ``"*"`` to accept any value,
            a literal string for exact match, or a list of values for membership test.

    Returns:
        True if value satisfies the filter, False otherwise.
    """
    if filter_value == "*":
        return True
    if isinstance(filter_value, list):
        return value in filter_value
    return value == filter_value


# =============================================================================
# Path resolution: InputBlock -> list[Meta]
# =============================================================================


def load_experiment_config(experiment_folder: Path) -> DetectorsExperimentConfig:
    """Load the latest saved detector experiment config from an experiment folder.

    Args:
        experiment_folder: Path to the experiment output folder containing saved config files.

    Returns:
        Parsed DetectorsExperimentConfig from the most recently saved config file.
    """
    cfg_path = get_latest_experiment_config_path(experiment_folder)
    cfg_raw = load_raw_config(cfg_path)
    return dict_to_model(cfg_raw, DetectorsExperimentConfig)


def get_npz_files(input_block: BaseInputBlock) -> list[NpzMeta]:
    """Resolve an InputBlock to the concrete NPZ file paths it refers to.

    Args:
        input_block: Input configuration specifying source type and filters.

    Returns:
        List of NpzMeta instances sorted by file path.

    Raises:
        ValueError: If the input block has no path set and no default experiment
            was resolved, or if the input block type is not recognized.
        FileNotFoundError: If a resolved NPZ path does not exist on disk.
    """
    if isinstance(input_block, PathInput):
        return _resolve_path_source(input_block)

    if input_block.path is None:
        raise ValueError(f"Input block {input_block} has no path set and no default_experiment was resolved.")
    exp_cfg = load_experiment_config(input_block.path)
    if isinstance(input_block, ExperimentInput):
        return _resolve_experiment_source(input_block, exp_cfg)
    if isinstance(input_block, AnnotationInput):
        return _resolve_annotation_source(input_block, exp_cfg)
    raise ValueError(f"Unknown input block type: {type(input_block)}")


def _resolve_experiment_source(
    input_block: ExperimentInput, exp_cfg: DetectorsExperimentConfig
) -> list[ExperimentMeta]:
    """Resolve experiment-source input blocks into concrete NPZ files."""
    out: list[ExperimentMeta] = []
    run_file = exp_cfg.run_config

    # Iterate each dataset run section from the saved experiment config.
    for dataset_name, run_ds in run_file.run.items():
        if not matches_filter(dataset_name, input_block.dataset):
            continue

        # load dataset config
        if dataset_name not in exp_cfg.dataset_config:
            raise KeyError(f"Dataset {dataset_name} is presented in detectors_run_file, but not in dataset_properties")

        # "components" in run_ds is the list of enabled components for this dataset.
        for subsequence_idx, run_seq in enumerate(run_ds.sequences):
            if not matches_filter(run_seq.sequence_id, input_block.sequence):
                continue
            if not matches_filter(subsequence_idx, input_block.subsequence):
                continue

            for algorithm_name in run_file.algorithms:
                algo_cfg = exp_cfg.detector_config.algorithms[algorithm_name]
                for component_name in get_algo_components(algo_cfg):
                    # Filter component and algorithms
                    if not matches_filter(component_name, input_block.component):
                        continue
                    if not matches_filter(algorithm_name, input_block.algorithm):
                        continue

                    # TODO: move this logic to detectors somehow?
                    # We need to resolve where experiment saved this subsequence data
                    # Resolve detector result folder for one concrete
                    # (dataset, sequence, component, algorithm)
                    ctx = {
                        "cur_dataset_name": dataset_name,
                        "cur_sequence_id": run_seq.sequence_id,
                        "cur_video_start": str(run_seq.video_start),
                        "cur_video_length": str(run_seq.video_length),
                        "cur_component_name": component_name,
                        "cur_algorithm_name": algorithm_name,
                    }

                    # get subsequence meta information
                    out_sub_folder = resolve_placeholders(run_file.io.out_sub_folder, ctx)
                    meta_path = out_sub_folder / "subsequence_meta.toml"
                    if not meta_path.exists():
                        raise FileNotFoundError(
                            f"Missing subsequence_meta.toml for dataset={dataset_name}, "
                            f"sequence={run_seq.sequence_id} at '{meta_path}'. "
                            "Re-run the detectors pipeline to produce it."
                        )
                    subsequence_meta = dict_to_model(load_raw_config(meta_path), ResolvedSubsequenceConfig)

                    # get npz path with detectors results
                    result_folder: Path = resolve_placeholders(run_file.io.detector_final_result_folder, ctx)
                    npz_path = result_folder / f"{algorithm_name}.npz"
                    if not npz_path.exists():
                        raise FileNotFoundError(
                            "Expected NPZ output is missing for configured detector: "
                            f"dataset={dataset_name}, sequence={run_seq.sequence_id}, "
                            f"video_start={run_seq.video_start}, video_length={run_seq.video_length}, "
                            f"component={component_name}, algorithm={algorithm_name}, path={npz_path}"
                        )

                    subseq = SubsequenceInfo(
                        subsequence_index=subsequence_idx,
                        video_start=subsequence_meta.video_start,
                        video_length=subsequence_meta.video_length,
                    )
                    meta = ExperimentMeta(
                        dataset=dataset_name,
                        sequence=run_seq.sequence_id,
                        component=component_name,
                        algorithm=algorithm_name,
                        subsequence=subseq,
                        npz_path=npz_path,
                        npz_key=input_block.npz_key,
                    )
                    out.append(meta)

    return sorted(out, key=lambda x: str(x.npz_path))


def _resolve_annotation_source(
    input_block: AnnotationInput, exp_cfg: DetectorsExperimentConfig
) -> list[AnnotationMeta]:
    """Resolve annotation-source input blocks into concrete NPZ files."""
    out: list[AnnotationMeta] = []
    run_file = exp_cfg.run_config

    for dataset_name in run_file.run:
        if not matches_filter(dataset_name, input_block.dataset):
            continue
        if dataset_name not in exp_cfg.dataset_config:
            raise KeyError(f"Dataset {dataset_name} is presented in detectors_run_file, but not in dataset_properties")
        ds_cfg = exp_cfg.dataset_config[dataset_name]

        for seq_cfg in ds_cfg.sequences:
            if not matches_filter(seq_cfg.sequence_id, input_block.sequence):
                continue
            for comp_name, comp_cfg in seq_cfg.annotation.components.items():
                if not matches_filter(comp_name, input_block.component):
                    continue
                annotation_path = comp_cfg.path
                unresolved = get_placeholders_str(str(annotation_path))
                if unresolved:
                    raise ValueError(
                        f"Annotation path for dataset='{dataset_name}', sequence='{seq_cfg.sequence_id}', "
                        f"component='{comp_name}' has unresolved placeholders {sorted(unresolved)}: "
                        f"'{annotation_path}'. Use sibling references (e.g. <sequence_id>) inside "
                        "dataset_properties, not runtime placeholders (e.g. <cur_sequence_id>)."
                    )
                if not annotation_path.exists():
                    continue

                meta = AnnotationMeta(
                    dataset=dataset_name,
                    sequence=seq_cfg.sequence_id,
                    component=comp_name,
                    npz_path=annotation_path,
                    npz_key=input_block.npz_key,
                )
                out.append(meta)

    return sorted(out, key=lambda x: str(x.npz_path))


def _resolve_path_source(input_block: PathInput) -> list[PathMeta]:
    """Resolve direct path-source input blocks using glob expansion."""
    paths = input_block.paths_list()
    matches: set[Path] = set()
    for p in paths:
        found = set(Path(m) for m in glob.glob(str(p), recursive=True))
        if not found:
            raise FileNotFoundError(f"No files matched path pattern: {p}")
        matches.update(found)

    all_npzs: list[PathMeta] = []
    for path in sorted(matches):
        meta = PathMeta(npz_path=path, npz_key=input_block.npz_key)
        all_npzs.append(meta)

    return all_npzs


# =============================================================================
# Alignment: predictions <-> ground truth
# =============================================================================


def _intersect_axis(
    data_a: np.ndarray,
    data_b: np.ndarray,
    labels_a: list[str],
    labels_b: list[str],
    axis: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Keep only labels present in both arrays on the given axis."""
    b_map = {label: i for i, label in enumerate(labels_b)}
    common = [label for label in labels_a if label in b_map]
    if not common:
        return data_a, data_b, []

    idx_a = [i for i, label in enumerate(labels_a) if label in b_map]
    idx_b = [b_map[label] for label in common]

    data_a = np.take(data_a, indices=idx_a, axis=axis)
    data_b = np.take(data_b, indices=idx_b, axis=axis)
    return data_a, data_b, common


def align_arrays(
    predictions: list[NpzArrayWithMeta],
    ground_truth: list[NpzArrayWithMeta],
    broadcast_single: bool = False,
) -> list[tuple[NpzArrayWithMeta, NpzArrayWithMeta]]:
    """Pair predictions with ground truth arrays and align their axes.

    Matches pairs by shared meta fields (dataset, session, sequence, component).
    For each pair, all axes (subjects, cameras, frames, labels, data) are intersected
    so both arrays have identical labels in the same order. Pairs with an empty
    intersection on any required axis are skipped with a warning.

    Args:
        predictions: Loaded prediction arrays to match against ground truth.
        ground_truth: Loaded ground truth arrays. A single PathMeta entry acts as
            a wildcard matched to any prediction without a structured match.
        broadcast_single: When True, axes where both sides have exactly one element
            are paired directly regardless of label mismatch, keeping the prediction
            label.

    Returns:
        List of (prediction, ground_truth) pairs with aligned axes.

    Raises:
        ValueError: If more than one path-based array is provided on either side.
    """
    # PathMeta has no metadata to match on — more than one path-based file on either side
    # makes it impossible to reliably pair predictions with ground truth.
    path_preds = [p for p in predictions if isinstance(p.meta, PathMeta)]
    path_gts = [g for g in ground_truth if isinstance(g.meta, PathMeta)]
    if len(path_preds) > 1 or len(path_gts) > 1:
        raise ValueError(
            f"Cannot reliably align path-based arrays: found {len(path_preds)} prediction(s) "
            f"and {len(path_gts)} ground truth file(s). "
            "Path-based inputs have no metadata for matching — use at most one file on each side."
        )

    # A single PathMeta GT acts as a wildcard — it matches any pred that has no better match.
    wildcard_gt = path_gts[0] if path_gts else None

    # Index ground truth by alignment key
    gt_by_key: dict[tuple, NpzArrayWithMeta] = {}
    for gt in ground_truth:
        key = gt.meta.align_key()
        if key is not None:
            gt_by_key[key] = gt

    out: list[tuple[NpzArrayWithMeta, NpzArrayWithMeta]] = []
    for pred in predictions:
        key = pred.meta.align_key()
        gt = (gt_by_key.get(key) if key is not None else None) or wildcard_gt
        if gt is None:
            logging.warning(f"No ground truth match for predictions: {pred.meta}")
            continue

        pred_data = pred.data
        gt_data = gt.data

        # Intersect each axis
        pred_axes = asdict(pred.axes)
        gt_axes = asdict(gt.axes)
        aligned: dict[str, list[str]] = {}
        for axis_idx, axis_name in enumerate(pred_axes):
            pred_labels = pred_axes[axis_name]
            gt_labels = gt_axes[axis_name]

            # Both empty (e.g. optional data axis) — skip
            if not pred_labels and not gt_labels:
                aligned[axis_name] = []
                continue

            # When both sides have exactly one element, pair them directly
            if broadcast_single and len(pred_labels) == 1 and len(gt_labels) == 1:
                aligned[axis_name] = pred_labels
                continue

            pred_data, gt_data, common = _intersect_axis(pred_data, gt_data, pred_labels, gt_labels, axis=axis_idx)
            aligned[axis_name] = common

        # Skip if any required axis is empty
        required = ("subjects", "cameras", "frames", "labels")
        empty = [name for name in required if not aligned[name]]
        if empty:
            for name in empty:
                pred_labels = asdict(pred.axes)[name]
                gt_labels = asdict(gt.axes)[name]
                logging.warning(
                    f"Empty intersection on '{name}' for {pred.meta}: "
                    f"pred={abbrev_list(pred_labels)} (n={len(pred_labels)}), "
                    f"gt={abbrev_list(gt_labels)} (n={len(gt_labels)})"
                )
            continue

        aligned_axes = NpzArrayAxes(**aligned)
        aligned_pred = NpzArrayWithMeta.create(meta=pred.meta, data=pred_data, axes=aligned_axes)
        aligned_gt = NpzArrayWithMeta.create(meta=gt.meta, data=gt_data, axes=aligned_axes)
        out.append((aligned_pred, aligned_gt))

    return out


# =============================================================================
# Convenience pipeline
# =============================================================================


def get_meta_type(arrays: list[NpzArrayWithMeta]) -> type[NpzMeta]:
    """Extract the shared NpzMeta type from a list of arrays.

    Args:
        arrays: Non-empty list of MetaNpzArray instances all sharing the same meta type.

    Returns:
        The common NpzMeta subclass used by all arrays in the list.

    Raises:
        ValueError: If arrays is empty or contains mixed meta types.
    """
    if not arrays:
        raise ValueError("Cannot determine meta type: array list is empty.")
    types = {type(arr.meta) for arr in arrays}
    if len(types) > 1:
        raise ValueError(f"Mixed meta types in array list: {types}. All arrays must share the same meta type.")
    return types.pop()


def load_input(input_block: BaseInputBlock, schema: BaseArraySchema | None = None) -> list[NpzArrayWithMeta]:
    """Load and prepare all arrays for a given input block.

    Resolves the relevant NPZ files based on the input block's source type
    (experiment/annotation/path), loads each one, and applies axis filters
    (subjects, cameras, labels, etc.).

    Args:
        input_block: Configuration describing what data to load and how to filter it.
        schema: Optional array schema. When provided, every loaded array is validated
            against it and a mismatch raises, so metrics fail fast on unexpected data.

    Returns:
        List of loaded arrays ready for metric iteration, sorted by source NPZ file.

    Raises:
        RuntimeError: If all resolved NPZ files are filtered out or no data is found.
        FileNotFoundError: If a resolved NPZ path does not exist on disk.
        ValueError: If a loaded array does not satisfy the given schema.
    """
    # First step is to figure out what npz paths we need to load
    # Given the current input block source type and configuration
    # this function will go to experiment/annotation/raw path
    # and find for us npz files that we are looking for
    npz_paths = get_npz_files(input_block)

    # Next, we will load founded npz files one by one
    # and filter the data further by axis filter
    arrays: list[NpzArrayWithMeta] = []
    for meta in npz_paths:
        loaded = load_and_filter_array(meta, input_block.axis_filters())
        # did this npz was completely filtered out?
        if loaded is None:
            continue
        # does this array aligned with schema?
        if schema is not None:
            errors = schema.validate(loaded.array)
            if errors:
                details = "\n".join(errors)
                raise ValueError(f"Array from {meta} failed schema validation:\n{details}")
        arrays.append(loaded)
    if not arrays:
        raise RuntimeError(f"Input block {input_block} is to strict or data is missing!")

    # By this point we have all data loaded
    # It's divided by different npz sources (dataset/session/sequence/path)
    # And filtered out inside by axis (subjects, cameras, etc.)
    # Now metric can naturally iterate over all arrays
    return arrays
