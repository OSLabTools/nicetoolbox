import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .npz_meta import NpzMeta, PathMeta


@dataclass
class NpzArrayAxes:
    """Fixed named axes according to the NICE Toolbox convention."""

    subjects: list[str]  # axis0
    cameras: list[str]  # axis1
    frames: list[str]  # axis2
    labels: list[str]  # axis3
    data: list[str] = field(default_factory=list)  # axis4 - optional

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape implied by the axis label counts.

        Includes axis4 (data) only when it is present, matching the toolbox convention where
        an empty data axis means a 4D array.
        """
        dims = [len(self.subjects), len(self.cameras), len(self.frames), len(self.labels)]
        if self.data:
            dims.append(len(self.data))
        return tuple(dims)

    def replace(self, **changes) -> "NpzArrayAxes":
        """Return a copy with the given axis fields replaced."""
        return replace(self, **changes)

    def make_4d(self) -> "NpzArrayAxes":
        """Return a copy with the axis4 (data) labels dropped."""
        return replace(self, data=[])


@dataclass
class NpzArray:
    """Array with named axes."""

    data: np.ndarray
    axes: NpzArrayAxes


@dataclass
class NpzArrayWithMeta:
    """An NpzArray together with the file metadata describing where it came from."""

    array: NpzArray
    npz_meta: NpzMeta

    @classmethod
    def create(cls, meta: NpzMeta, data: np.ndarray, axes: NpzArrayAxes) -> "NpzArrayWithMeta":
        """Build meta array from flat parts"""
        return cls(array=NpzArray(data=data, axes=axes), npz_meta=meta)

    def replace(self, *, data: np.ndarray | None = None, axes: NpzArrayAxes | None = None) -> "NpzArrayWithMeta":
        """Return a copy with new data and/or axes, carrying the same meta forward."""
        return NpzArrayWithMeta(
            array=NpzArray(
                data=self.data if data is None else data,
                axes=self.axes if axes is None else axes,
            ),
            npz_meta=self.npz_meta,
        )

    @property
    def data(self) -> np.ndarray:
        return self.array.data

    @property
    def axes(self) -> NpzArrayAxes:
        return self.array.axes

    @property
    def meta(self) -> NpzMeta:
        return self.npz_meta


# =============================================================================
# Array Loading
# =============================================================================


def _validate_shape(data: np.ndarray, descr: dict, npz_key: str, npz_path: Path) -> None:
    """
    Validate that data array shape matches axis labels in data_description.
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


def load_array(meta: NpzMeta) -> NpzArrayWithMeta | None:
    """Load one NPZ entry and read its axis labels from data_description.

    Args:
        meta: Metadata describing the NPZ file path and key to load.

    Returns:
        NpzArrayWithMeta with the full data and axis labels, or None if the requested
        key is absent from the file's data_description.

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

    # Build the unfiltered array from data_description.
    axes = NpzArrayAxes(
        subjects=[str(v) for v in descr["axis0"]],
        cameras=[str(v) for v in descr["axis1"]],
        frames=[str(v) for v in descr["axis2"]],
        labels=[str(v) for v in descr["axis3"]],
        data=[str(v) for v in descr["axis4"]] if "axis4" in descr else [],
    )
    return NpzArrayWithMeta(array=NpzArray(data, axes), npz_meta=meta)


def load_array_from_path(npz_path: Path, npz_key: str) -> NpzArray | None:
    """Load a single NPZ array by path and key, without meta.

    Args:
        npz_path: Path to the NPZ file to load.
        npz_key: Key of the array to read within the file's data_description.

    Returns:
        The bare NpzArray with full data and axis labels, or None if the requested key is
        absent from the file's data_description.

    Raises:
        KeyError: If data_description is missing from the NPZ file or the requested
            key is absent from data_description.
        ValueError: If the data array shape does not match data_description axis lengths.
    """
    loaded = load_array(PathMeta(npz_path=npz_path, npz_key=npz_key))
    return loaded.array if loaded is not None else None


# =============================================================================
# Filtering
# =============================================================================


@dataclass
class NpzAxisFilters:
    """Detached axis filters extracted from InputBlock."""

    subject: str | list[str]
    camera: str | list[str]
    label: str | list[str]
    data: str | list[str]


def _apply_filter(
    data: np.ndarray, labels: list, filter_value: str | list[str], axis: int, npz_path: Path | None = None
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


def select_array(
    array: NpzArray,
    *,
    subjects: list[str] | None = None,
    cameras: list[str] | None = None,
    frames: list[str] | None = None,
    labels: list[str] | None = None,
    data: list[str] | None = None,
) -> NpzArray:
    """Narrow an array to the named entries on one or more axes.

    Each argument takes the labels to keep, in the given order (so this also reorders).
    Omitted axes are kept whole; the input array is not modified.

    Raises when a requested name is absent, so a detector asking for a camera its upstream
    does not provide fails loudly.

    Example:
        select_array(landmarks, cameras=["view_left", "view_right"])

    Raises:
        ValueError: if any requested name is not present on its axis.
    """
    selections = {
        "subjects": (subjects, 0),
        "cameras": (cameras, 1),
        "frames": (frames, 2),
        "labels": (labels, 3),
        "data": (data, 4),
    }

    new_data = array.data
    changes: dict[str, list[str]] = {}
    for axis_name, (wanted, axis) in selections.items():
        if wanted is None:
            continue
        available = getattr(array.axes, axis_name)
        missing = [name for name in wanted if name not in available]
        if missing:
            raise ValueError(f"Requested {axis_name} {missing} not present on axis{axis}. Available: {available}.")
        new_data = np.take(new_data, [available.index(name) for name in wanted], axis=axis)
        changes[axis_name] = list(wanted)

    return NpzArray(new_data, array.axes.replace(**changes))


def filter_array(array: NpzArray, filters: NpzAxisFilters, npz_path: Path | None = None) -> NpzArray | None:
    """Filter an array by label names on the subject/camera/label/data axes.

    Frames (axis2) are never filtered. Wildcard '*' on any axis keeps everything.

    Args:
        array: The array to filter.
        filters: Axis filter spec (subject, camera, label, data).
        npz_path: Optional source path, used only for clearer log messages.

    Returns:
        A new NpzArray with filtered data and axis labels, or None if any required
        axis (subjects, cameras, labels, or data when present) has no overlap with
        the requested filter.
    """
    axes = array.axes
    data = array.data

    data, subjects = _apply_filter(data, axes.subjects, filters.subject, axis=0, npz_path=npz_path)
    data, cameras = _apply_filter(data, axes.cameras, filters.camera, axis=1, npz_path=npz_path)
    frames = list(axes.frames)  # ! frames doesn't support filtering, getting them as is
    data, labels = _apply_filter(data, axes.labels, filters.label, axis=3, npz_path=npz_path)

    axis_to_validate = [subjects, cameras, labels]
    # axis4 (data) is optional — some components store scalar values per label.
    if axes.data:
        data, data_axis = _apply_filter(data, axes.data, filters.data, axis=4, npz_path=npz_path)
        axis_to_validate.append(data_axis)
    else:
        data_axis = []

    # If any filtered axis is empty, this NPZ has no usable data for the request.
    # For example, we filtered out all subjects or camera names, so we discard it completely.
    if any(not lst for lst in axis_to_validate):
        return None

    return NpzArray(data, NpzArrayAxes(subjects, cameras, frames, labels, data_axis))


def load_and_filter_array(meta: NpzMeta, filters: NpzAxisFilters) -> NpzArrayWithMeta | None:
    """Load one NPZ entry and apply axis filters in a single step.

    Args:
        meta: Metadata describing the NPZ file path and key to load.
        filters: Axis filter spec (subject, camera, label, data) to apply after loading.

    Returns:
        NpzArrayWithMeta with filtered data and axis labels, or None if the key is
        absent or any filtered axis has no overlap with the available labels.
    """
    loaded = load_array(meta)
    if loaded is None:
        return None
    filtered = filter_array(loaded.array, filters, npz_path=meta.npz_path)
    if filtered is None:
        return None
    return NpzArrayWithMeta(array=filtered, npz_meta=meta)


# =============================================================================
# Array Saving
# =============================================================================


def save_arrays(arrays: dict[str, NpzArray], npz_path: Path) -> None:
    """Save named arrays to an NPZ file using the toolbox data_description convention.

    Each array's ArrayAxes is serialized to axis0..axis3 (and axis4 when the data
    axis is present), collected into a single ``data_description`` dict keyed by the
    array name, matching the format produced by detector outputs.

    Args:
        arrays: Mapping of npz_key -> NpzArray to write.
        npz_path: Destination NPZ file path. Parent directories must already exist.
    """
    data_arrays: dict[str, np.ndarray] = {}
    data_description: dict[str, dict] = {}
    for key, arr in arrays.items():
        data_arrays[key] = arr.data
        descr = {
            "axis0": arr.axes.subjects,
            "axis1": arr.axes.cameras,
            "axis2": arr.axes.frames,
            "axis3": arr.axes.labels,
        }
        if arr.axes.data:
            descr["axis4"] = arr.axes.data
        data_description[key] = descr

    np.savez_compressed(npz_path, **data_arrays, data_description=data_description)
    logging.info(f"Saved NPZ with keys {list(arrays)}: {npz_path}")
