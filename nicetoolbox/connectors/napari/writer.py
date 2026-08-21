import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL
from nicetoolbox_core.data.loaded_array import NpzArray, load_array_from_path

from ...configs.utils import resolve_filter
from .napari_configs import NapariExportBodyJointsConfig, NapariExportSequenceConfig, NapariWindowConfig

H5_KEY = "df_with_missing"
COORDS = ("x", "y", "likelihood")
SCORER = "NICEToolbox"
IMAGE_EXT = ".png"
H5_FILENAME = f"CollectedData_{SCORER}.h5"


def load_body_joints_array(input_path: Path, npz_key: str) -> NpzArray:
    """Load a body_joints NPZ and validate it is a 2d pose array.

    Uses the shared loader (which checks data_description against the array shape)
    and the shared VECTOR_2D_CONF_PER_LABEL schema, so the export refuses anything
    that is not (subjects, cameras, frames, joints, [x, y, confidence]).
    """
    array = load_array_from_path(input_path, npz_key)
    if array is None:
        raise ValueError(f"Key '{npz_key}' not found in data_description of '{input_path}'")

    errors = VECTOR_2D_CONF_PER_LABEL.validate(array)
    if errors:
        raise ValueError(
            f"Array '{npz_key}' in '{input_path}' is not a valid 2d body_joints array: {'; '.join(errors)}"
        )

    logging.info(
        f"Loaded '{npz_key}' from {input_path}: shape={array.data.shape}, "
        f"subjects={array.axes.subjects}, cameras={array.axes.cameras}, joints={len(array.axes.labels)}"
    )
    return array


def _camera_dataframe(
    cam_arr: np.ndarray,
    subjects: list[str],
    frames: list[str],
    bodyparts: list[str],
    camera: str,
    root_name: str,
) -> pd.DataFrame:
    """Build the wide DLC DataFrame for one camera from a (subjects, frames, joints, 3) slice."""
    n_subjects, n_frames, n_bodyparts, _ = cam_arr.shape

    # (frames, subjects*bodyparts*coords), subject-major to match the column order below.
    flat = cam_arr.transpose(1, 0, 2, 3).reshape(n_frames, n_subjects * n_bodyparts * len(COORDS))

    columns = pd.MultiIndex.from_tuples(
        [(SCORER, subject, bodypart, coord) for subject in subjects for bodypart in bodyparts for coord in COORDS],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    index = pd.MultiIndex.from_tuples([(root_name, camera, f"{frame}{IMAGE_EXT}") for frame in frames])
    return pd.DataFrame(flat, index=index, columns=columns)


def _copy_frames(frames_folder: Path, dest_dir: Path, camera: str, frames: list[str]) -> int:
    """Copy the frames referenced by one camera's annotations next to its annotation file.

    An existing folder for this camera is deleted first, so frames left over from an earlier
    run (a wider frame range, a different NPZ) cannot linger beside the new annotations.
    """
    src_dir = frames_folder / camera / "frames"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Frames folder for camera '{camera}' not found: {src_dir}")

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
        logging.info(f"Removed previous export for camera '{camera}': {dest_dir}")
    dest_dir.mkdir(parents=True)

    missing: list[str] = []
    copied = 0
    for frame in frames:
        name = f"{frame}{IMAGE_EXT}"
        src = src_dir / name
        if not src.is_file():
            missing.append(name)
            continue
        shutil.copyfile(src, dest_dir / name)
        copied += 1

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frame image(s) referenced by the annotations are missing from {src_dir}. "
            f"First missing: {missing[:5]}"
        )

    logging.info(f"Camera '{camera}': copied {copied} frame image(s) to {dest_dir}")
    return copied


def _window_indices(n_frames: int, window: NapariWindowConfig) -> list[int]:
    """Frame positions kept by the sliding window: `size` consecutive, every `stride`.

    A trailing window is truncated at the end of the sequence rather than dropped, so the
    last frames are still reachable for labelling.
    """
    kept = [i for start in range(0, n_frames, window.stride) for i in range(start, min(start + window.size, n_frames))]
    logging.info(
        f"Window sampling: size={window.size}, stride={window.stride} -> "
        f"{len(kept)} of {n_frames} frames in {len(range(0, n_frames, window.stride))} window(s)"
    )
    return kept


def body_joints_npz_to_napari(
    sequence: NapariExportSequenceConfig,
    cfg: NapariExportBodyJointsConfig,
) -> list[Path]:
    """Convert one body_joints NPZ into a napari project, one annotation file per camera.

    Builds this layout under `sequence.output`::

        <output>/<camera>/CollectedData_<scorer>.h5
        <output>/<camera>/<frame>.png

    Each camera's annotation file sits in the same folder as the frames it annotates.
    Row-index paths are "<output name>/<camera>/<frame>.png", i.e. relative to the
    project root's parent.

    When `cfg.window` is set, only the sampled frames are exported. Frame identity lives in
    the filename, so a sampled export imports back correctly with the unexported frames
    left as NaN.
    """
    array = load_body_joints_array(sequence.input, cfg.npz_key)

    subjects = array.axes.subjects
    all_cameras = array.axes.cameras

    # raise_on_unknown: a typo in an export config should fail, not quietly write fewer files.
    selected = resolve_filter(sequence.cameras, all_cameras, raise_on_unknown=True)
    logging.info(f"Exporting cameras: {selected}")

    data = array.data
    frames = array.axes.frames
    if cfg.window is not None:
        kept = _window_indices(len(frames), cfg.window)
        data = data[:, :, kept, :, :]
        frames = [frames[i] for i in kept]

    written: list[Path] = []
    for camera in selected:
        cam_arr = data[:, all_cameras.index(camera), :, :, :]
        df = _camera_dataframe(cam_arr, subjects, frames, array.axes.labels, camera, sequence.output.name)

        camera_dir = sequence.output / camera
        _copy_frames(sequence.frames_folder, camera_dir, camera, frames)

        out_path = camera_dir / H5_FILENAME
        # format="fixed" (the DeepLabCut default): "table" cannot store a MultiIndex on both axes.
        df.to_hdf(out_path, key=H5_KEY, mode="w", format="fixed")

        filled = int(np.count_nonzero(~np.isnan(cam_arr[..., 0])))
        total = cam_arr.shape[0] * cam_arr.shape[1] * cam_arr.shape[2]
        logging.info(
            f"Camera '{camera}': wrote {df.shape[0]} rows x {df.shape[1]} cols "
            f"({filled}/{total} annotated points) to {out_path}"
        )
        written.append(out_path)

    return written
