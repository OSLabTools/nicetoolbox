import glob
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...configs.placeholders import resolve_placeholders
from ...configs.schemas.detectors_run_file import ResolvedSubsequenceMeta
from ...configs.schemas.evaluation_input_block import (
    AnnotationInput,
    BaseInputBlock,
    ExperimentInput,
    NpzAxis,
    PathInput,
)
from ...configs.schemas.experiment_config import DetectorsExperimentConfig
from ...configs.utils import dict_to_model, get_latest_experiment_config_path, load_raw_config
from ...detectors.main import get_algo_components
from ...utils.logging_utils import abbrev_list

# =============================================================================
# Data classes
# =============================================================================


@dataclass
class SubsequenceInfo:
    subsequence_index: int
    video_start: int  # resolved frame index
    video_length: int  # resolved frame count


@dataclass
class NpzMeta(ABC):
    """Abstract base for all NPZ metadata types."""

    npz_path: Path
    npz_key: str

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return user-facing metadata fields as a flat dict."""
        ...

    @staticmethod
    @abstractmethod
    def always_iterate() -> frozenset[str]:
        """Return columns that must always get their own row in summaries (never pooled)."""
        ...

    @staticmethod
    @abstractmethod
    def comparable_dim() -> str | None:
        """Return the dimension used to compare results side-by-side (e.g. as series/color in charts)."""
        ...

    def align_key(self) -> tuple | None:
        """Return the key used to pair this array with its counterpart during alignment."""
        return None


@dataclass
class ExperimentMeta(NpzMeta):
    dataset: str
    session: str
    sequence: str
    component: str
    algorithm: str
    subsequence: SubsequenceInfo

    @classmethod
    def always_iterate(cls) -> frozenset[str]:
        return frozenset({"component", "algorithm", "npz_key"})

    @staticmethod
    def comparable_dim() -> str | None:
        return "algorithm"

    def align_key(self) -> tuple:
        return (self.dataset, self.session, self.sequence, self.component)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "session": self.session,
            "sequence": self.sequence,
            "subsequence": self.subsequence.subsequence_index,
            "subsequence_start": self.subsequence.video_start,
            "subsequence_length": self.subsequence.video_length,
            "component": self.component,
            "algorithm": self.algorithm,
            "npz_key": self.npz_key,
        }


@dataclass
class AnnotationMeta(NpzMeta):
    dataset: str
    session: str
    sequence: str
    component: str

    @classmethod
    def always_iterate(cls) -> frozenset[str]:
        return frozenset({"component", "npz_key"})

    @staticmethod
    def comparable_dim() -> str | None:
        return None

    def align_key(self) -> tuple:
        return (self.dataset, self.session, self.sequence, self.component)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "session": self.session,
            "sequence": self.sequence,
            "component": self.component,
            "npz_key": self.npz_key,
        }


@dataclass
class PathMeta(NpzMeta):
    @classmethod
    def always_iterate(cls) -> frozenset[str]:
        return frozenset({"npz_file_name", "npz_key"})

    @staticmethod
    def comparable_dim() -> str | None:
        return "npz_file_name"

    def to_dict(self) -> dict[str, Any]:
        return {
            "npz_file_name": self.npz_path.stem,
            "npz_key": self.npz_key,
        }


@dataclass
class ArrayAxes:
    subjects: list[str]
    cameras: list[str]
    frames: list[str]
    labels: list[str]
    data: list[str] = field(default_factory=list)


@dataclass
class LoadedArray:
    meta: NpzMeta
    data: np.ndarray
    axes: ArrayAxes


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
        for subsequence_idx, video in enumerate(run_ds.videos):
            # Filter sessions and sequences
            if not matches_filter(video.session_ID, input_block.session):
                continue
            if not matches_filter(video.sequence_ID, input_block.sequence):
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
                    # (dataset, session, sequence, component, algorithm)
                    ctx = {
                        "cur_dataset_name": dataset_name,
                        "cur_session_ID": video.session_ID,
                        "cur_sequence_ID": video.sequence_ID,
                        "cur_video_start": str(video.video_start),
                        "cur_video_length": str(video.video_length),
                        "cur_component_name": component_name,
                        "cur_algorithm_name": algorithm_name,
                    }

                    # get subsequence meta information
                    out_sub_folder = resolve_placeholders(run_file.io.out_sub_folder, ctx)
                    meta_path = out_sub_folder / "subsequence_meta.toml"
                    if not meta_path.exists():
                        raise FileNotFoundError(
                            f"Missing subsequence_meta.toml for dataset={dataset_name}, session={video.session_ID}, "
                            f"sequence={video.sequence_ID} at '{meta_path}'. "
                            "Re-run the detectors pipeline to produce it."
                        )
                    subsequence_meta = dict_to_model(load_raw_config(meta_path), ResolvedSubsequenceMeta)

                    # get npz path with detectors results
                    result_folder: Path = resolve_placeholders(run_file.io.detector_final_result_folder, ctx)
                    npz_path = result_folder / f"{algorithm_name}.npz"
                    if not npz_path.exists():
                        raise FileNotFoundError(
                            "Expected NPZ output is missing for configured detector: "
                            f"dataset={dataset_name}, session={video.session_ID}, sequence={video.sequence_ID}, "
                            f"video_start={video.video_start}, video_length={video.video_length}, "
                            f"component={component_name}, algorithm={algorithm_name}, path={npz_path}"
                        )

                    subseq = SubsequenceInfo(
                        subsequence_index=subsequence_idx,
                        video_start=subsequence_meta.video_start,
                        video_length=subsequence_meta.video_length,
                    )
                    meta = ExperimentMeta(
                        dataset=dataset_name,
                        session=video.session_ID,
                        sequence=video.sequence_ID,
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

        # Filter component and algorithms
        # TODO: we get all possible session / sequence from dataset
        # should we filter them further based on what we have in run_detectors?
        sessions = [ses for ses in ds_cfg.session_IDs if matches_filter(ses, input_block.session)]
        sequences = [seq for seq in ds_cfg.sequence_IDs if matches_filter(seq, input_block.sequence)]

        for session_id in sessions:
            for sequence_id in sequences:
                for comp_name, comp_cfg in ds_cfg.annotation.components.items():
                    if not matches_filter(comp_name, input_block.component):
                        continue
                    # Reconstruct annotation path from ctx
                    ctx = {
                        "cur_dataset_name": dataset_name,
                        "cur_session_ID": session_id,
                        "cur_sequence_ID": sequence_id,
                        "cur_component_name": comp_name,
                    }
                    annotation_path = Path(resolve_placeholders(comp_cfg.path, ctx))
                    if not annotation_path.exists():
                        continue

                    meta = AnnotationMeta(
                        dataset=dataset_name,
                        session=session_id,
                        sequence=sequence_id,
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
# NPZ loading: meta -> LoadedArray
# =============================================================================


def load_array(meta: NpzMeta, filters: NpzAxis) -> LoadedArray | None:
    """Load one NPZ entry, read axis labels from data_description, and apply filters.

    Args:
        meta: Metadata describing the NPZ file path and key to load.
        filters: Axis filter spec (subjects, cameras, labels, data) to apply after loading.

    Returns:
        LoadedArray with filtered data and axis labels, or None if any filtered
        axis has no overlap with the available labels.

    Raises:
        KeyError: If data_description is missing from the NPZ file or the requested
            key is absent from data_description.
        ValueError: If the data array shape does not match data_description axis lengths.
    """
    npz_path = meta.npz_path
    npz_key = meta.npz_key

    with np.load(npz_path, allow_pickle=True) as f:
        # check data description and find desired npz_key
        if "data_description" not in f.files:
            raise KeyError(f"Key data_description not found in '{npz_path}'. Available: {f.files}")
        descr = f["data_description"].item()
        if npz_key not in descr:
            logging.warning(f"Key '{npz_key}' not in data_description of '{npz_path}'. Available: {list(descr.keys())}")
            return None
        descr = descr[npz_key]
        # find relevant data by npz_key
        if npz_key not in f.files:
            raise KeyError(
                f"Key '{npz_key}' not found in '{npz_path}', but present in data_description. Available: {f.files}"
            )
        data = f[npz_key]

    # Validate data description matches numpy shape
    _validate_shape(data, descr, npz_key, npz_path)

    # Apply filters from input recipe.
    data, subjects = _apply_filter(data, descr["axis0"], filters.subject, axis=0, npz_path=npz_path)
    data, cameras = _apply_filter(data, descr["axis1"], filters.camera, axis=1, npz_path=npz_path)
    frames = list(descr["axis2"])  # ! frames doesn't support filtering, getting them as is
    data, labels = _apply_filter(data, descr["axis3"], filters.label, axis=3, npz_path=npz_path)

    axis_to_validate = [subjects, cameras, labels]
    # axis4 (data) is optional — some components store scalar values per label.
    if "axis4" in descr:
        data, data_axis = _apply_filter(data, descr["axis4"], filters.data, axis=4, npz_path=npz_path)
        axis_to_validate.append(data_axis)
    else:
        data_axis = []

    # If any filtered axis is empty, this NPZ has no usable data for the request.
    # For example, we filtered out all subjects or camera names
    # So we discard it completely
    if any(not lst for lst in axis_to_validate):
        return None

    return LoadedArray(meta, data, ArrayAxes(subjects, cameras, frames, labels, data_axis))


def _validate_shape(data: np.ndarray, descr: dict, npz_key: str, npz_path: Path) -> None:
    """Validate that data array shape matches axis labels in data_description.

    These are written independently by the detector — a mismatch means a bug
    in the writer, and we want a clear error rather than a silent wrong result
    or a raw numpy IndexError later.
    """
    required_axes = ["axis0", "axis1", "axis2", "axis3"]
    if data.ndim < len(required_axes):
        raise ValueError(
            f"Data array for key '{npz_key}' in '{npz_path}' has {data.ndim} dimensions, "
            f"expected at least {len(required_axes)}."
        )
    for dim, axis_key in enumerate(required_axes):
        if axis_key not in descr:
            raise KeyError(f"'{axis_key}' missing from data_description['{npz_key}'] in '{npz_path}'.")
        n_labels = len(descr[axis_key])
        if data.shape[dim] != n_labels:
            raise ValueError(
                f"Shape mismatch on {axis_key} for key '{npz_key}' in '{npz_path}': "
                f"data.shape[{dim}]={data.shape[dim]} but data_description has {n_labels} labels: "
                f"{list(descr[axis_key])}."
            )
    if "axis4" in descr:
        if data.ndim < 5:
            raise ValueError(
                f"data_description has axis4 for key '{npz_key}' in '{npz_path}' "
                f"but data only has {data.ndim} dimensions."
            )
        n_labels = len(descr["axis4"])
        if data.shape[4] != n_labels:
            raise ValueError(
                f"Shape mismatch on axis4 for key '{npz_key}' in '{npz_path}': "
                f"data.shape[4]={data.shape[4]} but data_description has {n_labels} labels: "
                f"{list(descr['axis4'])}."
            )


def _apply_filter(
    data: np.ndarray, labels: list, filter_value: str | list[str], axis: int, npz_path: Path
) -> tuple[np.ndarray, list[str]]:
    """Filter one axis by label names. Wildcard '*' keeps everything.
    Missing labels are logged and skipped (intersection kept)."""
    labels = [str(v) for v in labels]

    if filter_value == "*":
        return data, labels

    wanted = [filter_value] if isinstance(filter_value, str) else list(filter_value)
    label_to_idx = {name: i for i, name in enumerate(labels)}

    missing = [name for name in wanted if name not in label_to_idx]
    if missing:
        logging.warning(
            f"Requested labels not found on axis {axis} in '{npz_path}': {missing}. "
            f"Available: {labels}. Keeping intersection only."
        )

    matched = [name for name in wanted if name in label_to_idx]
    if not matched:
        return data, []

    idx = [label_to_idx[name] for name in matched]
    return np.take(data, indices=idx, axis=axis), matched


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
    predictions: list[LoadedArray],
    ground_truth: list[LoadedArray],
    broadcast_single: bool = False,
) -> list[tuple[LoadedArray, LoadedArray]]:
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
    gt_by_key: dict[tuple, LoadedArray] = {}
    for gt in ground_truth:
        key = gt.meta.align_key()
        if key is not None:
            gt_by_key[key] = gt

    out: list[tuple[LoadedArray, LoadedArray]] = []
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

        aligned_axes = ArrayAxes(**aligned)
        aligned_pred = LoadedArray(meta=pred.meta, data=pred_data, axes=aligned_axes)
        aligned_gt = LoadedArray(meta=gt.meta, data=gt_data, axes=aligned_axes)
        out.append((aligned_pred, aligned_gt))

    return out


# =============================================================================
# Convenience pipeline
# =============================================================================


def get_meta_type(arrays: list[LoadedArray]) -> type[NpzMeta]:
    """Extract the shared NpzMeta type from a list of arrays.

    Args:
        arrays: Non-empty list of LoadedArray instances all sharing the same meta type.

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


def load_input(input_block: BaseInputBlock) -> list[LoadedArray]:
    """Load and prepare all arrays for a given input block.

    Resolves the relevant NPZ files based on the input block's source type
    (experiment/annotation/path), loads each one, and applies axis filters
    (subjects, cameras, labels, etc.).

    Args:
        input_block: Configuration describing what data to load and how to filter it.

    Returns:
        List of loaded arrays ready for metric iteration, sorted by source NPZ file.

    Raises:
        RuntimeError: If all resolved NPZ files are filtered out or no data is found.
        FileNotFoundError: If a resolved NPZ path does not exist on disk.
    """
    # First step is to figure out what npz paths we need to load
    # Given the current input block source type and configuration
    # this function will go to experiment/annotation/raw path
    # and find for us npz files that we are looking for
    npz_paths = get_npz_files(input_block)

    # Next, we will load founded npz files one by one
    # and filter the data further by axis filter
    arrays: list[LoadedArray] = []
    for meta in npz_paths:
        loaded = load_array(meta, input_block.axis_filters())
        # did this npz was completely filtered out?
        if loaded is not None:
            arrays.append(loaded)
    if not arrays:
        raise RuntimeError(f"Input block {input_block} is to strict or data is missing!")

    # By this point we have all data loaded
    # It's divided by different npz sources (dataset/session/sequence/path)
    # And filtered out inside by axis (subjects, cameras, etc.)
    # Now detector can naturally iterate over all arrays
    return arrays
