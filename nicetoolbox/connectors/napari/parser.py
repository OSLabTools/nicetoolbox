"""Parse napari body-joint annotations into Toolbox body_joints NPZ.

The napari (DeepLabCut-style) annotation file is HDF5, with a 4-level column index
(scorer, individuals, bodyparts, coords) and a row index whose last level is the
image filename.

The sequence config's `input` is a root folder holding one subfolder per camera, each
with a single annotation file. The subfolder name is the camera name — it is not read
from inside the file. All cameras must declare the same individuals and bodyparts.

Output shape: 2d (subjects, cameras, frames, joints, 3) — x/y/confidence.
The frame axis holds only frames that were actually annotated, so a sampled export stays
sparse rather than padding the gaps; a frame annotated for one camera but not another is
NaN for the missing camera.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ...configs.models.video_timestamp import timestamp_to_frame_index
from .napari_configs import NapariImportBodyJointsConfig, NapariSequenceConfig

REQUIRED_COORDS = ("x", "y")
LIKELIHOOD_COORD = "likelihood"
MANUAL_CONFIDENCE = 1.0
ANNOTATION_SUFFIXES = {".h5", ".hdf", ".hdf5"}


def load_napari_dataframe(input_path: Path) -> pd.DataFrame:
    """Load a napari HDF5 annotation file as a MultiIndex DataFrame.

    HDF5 only: it round-trips the row index as stored, whereas reading the CSV form requires
    assuming a fixed index depth, which silently folds data columns into the index when the
    file uses a flat image-path index instead.
    """
    suffix = input_path.suffix.lower()
    if suffix not in ANNOTATION_SUFFIXES:
        raise ValueError(
            f"Unsupported napari input extension '{suffix}' ({input_path.name}). "
            f"Expected one of {sorted(ANNOTATION_SUFFIXES)}."
        )

    df = pd.read_hdf(input_path)
    df.columns = pd.MultiIndex.from_tuples(
        [tuple(str(lvl) for lvl in col) for col in df.columns],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    return df


def _frame_number(image_name: str) -> int:
    """Extract integer frame number from a filename like '000003600.png'."""
    return int(Path(image_name).stem)


def discover_camera_files(input_root: Path) -> dict[str, Path]:
    """Find one annotation file per camera under `input_root`.

    Every subfolder is taken to be a camera, named after the folder, and must hold exactly
    one HDF5 annotation file (the filename itself is not read — a napari project names it
    after its own scorer).
    """
    if not input_root.is_dir():
        raise NotADirectoryError(f"Napari input root is not a directory: {input_root}")

    camera_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    if not camera_dirs:
        raise FileNotFoundError(f"No camera subfolders found under: {input_root}")

    files: dict[str, Path] = {}
    for camera_dir in camera_dirs:
        found = sorted(p for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in ANNOTATION_SUFFIXES)
        if not found:
            raise FileNotFoundError(
                f"No annotation file ({', '.join(sorted(ANNOTATION_SUFFIXES))}) in camera folder: {camera_dir}"
            )
        if len(found) > 1:
            raise ValueError(
                f"Expected one annotation file in '{camera_dir}', found {len(found)}: {[p.name for p in found]}"
            )
        files[camera_dir.name] = found[0]

    logging.info(f"Discovered {len(files)} camera(s) under {input_root}: {list(files)}")
    return files


def _validate_columns(frames_by_camera: dict[str, pd.DataFrame]) -> tuple[list[str], list[str], bool]:
    """Check every camera declares the same individuals and bodyparts, and has the x/y/likelihood coords.

    The output array has one shared subject and joint axis, so cameras that disagree cannot be
    stacked into it. Returns the agreed (individuals, bodyparts) in first-camera order, plus
    whether every camera carries a likelihood column.
    """
    individuals: list[str] = []
    bodyparts: list[str] = []
    reference: str = ""
    has_likelihood = True

    for camera, df in frames_by_camera.items():
        cam_individuals = list(df.columns.get_level_values("individuals").unique())
        cam_bodyparts = list(df.columns.get_level_values("bodyparts").unique())
        coords = list(df.columns.get_level_values("coords").unique())

        missing = [c for c in REQUIRED_COORDS if c not in coords]
        if missing:
            raise ValueError(f"Camera '{camera}': expected coords {missing} in napari columns, got: {coords}")

        if LIKELIHOOD_COORD not in coords:
            # napari strips likelihood when a human saves the file; fall back to a fixed confidence.
            logging.info(f"Camera '{camera}': no '{LIKELIHOOD_COORD}' column, using {MANUAL_CONFIDENCE} confidence")
            has_likelihood = False

        if not reference:
            reference, individuals, bodyparts = camera, cam_individuals, cam_bodyparts
            continue

        if cam_individuals != individuals:
            raise ValueError(
                f"Camera '{camera}' individuals {cam_individuals} differ from camera "
                f"'{reference}' {individuals}. All cameras must annotate the same subjects."
            )
        if cam_bodyparts != bodyparts:
            raise ValueError(
                f"Camera '{camera}' bodyparts differ from camera '{reference}'. "
                f"Only in '{camera}': {sorted(set(cam_bodyparts) - set(bodyparts))}; "
                f"only in '{reference}': {sorted(set(bodyparts) - set(cam_bodyparts))}."
            )

    logging.info(f"Napari columns: individuals={individuals}, bodyparts={len(bodyparts)}, likelihood={has_likelihood}")
    return individuals, bodyparts, has_likelihood


def _frame_numbers(df: pd.DataFrame) -> np.ndarray:
    """Frame numbers for every row, taken from the last index level (the image filename)."""
    return np.array([_frame_number(row[-1] if isinstance(row, tuple) else row) for row in df.index])


def napari_to_body_joints_npz(
    sequence: NapariSequenceConfig,
    cfg: NapariImportBodyJointsConfig,
) -> None:
    """Convert a folder of per-camera napari annotations into one multi-camera body_joints NPZ."""
    camera_files = discover_camera_files(sequence.input)

    frames_by_camera: dict[str, pd.DataFrame] = {}
    for camera, input_path in camera_files.items():
        df = load_napari_dataframe(input_path)
        logging.info(f"Camera '{camera}': loaded {input_path} (rows={len(df)})")
        frames_by_camera[camera] = df

    individuals, bodyparts, has_likelihood = _validate_columns(frames_by_camera)

    if cfg.subjects:
        unknown = [v for v in individuals if v not in cfg.subjects]
        if unknown:
            raise ValueError(f"Napari individuals {unknown} not in [subjects] mapping {sorted(cfg.subjects)}")
        subjects = [cfg.subjects[v] for v in individuals]
    else:
        subjects = list(individuals)
    logging.info(f"Subject order: {individuals} -> {subjects}")

    napari_cameras = list(camera_files)
    if cfg.cameras:
        unknown = [v for v in napari_cameras if v not in cfg.cameras]
        if unknown:
            raise ValueError(f"Napari cameras {unknown} not in [cameras] mapping {sorted(cfg.cameras)}")
        cameras = [cfg.cameras[v] for v in napari_cameras]
    else:
        cameras = list(napari_cameras)
    logging.info(f"Camera order: {napari_cameras} -> {cameras}")

    # Frame range spans every camera, so cameras annotated over different ranges stay aligned.
    all_frame_numbers = sorted({int(n) for df in frames_by_camera.values() for n in _frame_numbers(df)})
    first_frame, last_frame = all_frame_numbers[0], all_frame_numbers[-1]

    start_frame = max(first_frame, timestamp_to_frame_index(sequence.start, sequence.fps))
    if sequence.end != -1:
        # Explicit bounds are half-open, so [start, end) — the frame at `end` is excluded.
        end_frame = timestamp_to_frame_index(sequence.end, sequence.fps)
    else:
        # "Until the end" must keep the last annotated frame, which a half-open bound would drop.
        end_frame = last_frame + 1

    # Only annotated frames go into the array. A sampled export (see the export task's window
    # setting) annotates short clips scattered across the recording; emitting the full dense
    # range instead would leave the gaps as NaN, and evaluation would score those as failures
    # rather than skipping them. Keeping the axis sparse lets alignment intersect on frame
    # labels, so metrics only ever see frames a human actually labelled.
    kept_frames = [f for f in all_frame_numbers if start_frame <= f < end_frame]
    if not kept_frames:
        raise ValueError(
            f"No annotated frames in range [{start_frame}, {end_frame}). "
            f"Annotations cover {first_frame}..{last_frame}."
        )
    position_of = {frame: i for i, frame in enumerate(kept_frames)}
    n_frames = len(kept_frames)

    if sequence.reset_frames:
        frame_labels = [f"{i:09d}" for i in range(n_frames)]
    else:
        frame_labels = [f"{f:09d}" for f in kept_frames]
    logging.info(
        f"Frame range: {start_frame}..{end_frame} — keeping {n_frames} annotated frame(s), "
        f"reset_frames={sequence.reset_frames}"
    )

    arr = np.full(
        (len(subjects), len(cameras), n_frames, len(bodyparts), 3),
        fill_value=np.nan,
        dtype=np.float64,
    )

    for cam_idx, napari_cam in enumerate(napari_cameras):
        cam_df = frames_by_camera[napari_cam]
        raw_frames = _frame_numbers(cam_df)
        keep = (raw_frames >= start_frame) & (raw_frames < end_frame)
        cam_df = cam_df.iloc[keep]
        frame_indices = np.array([position_of[f] for f in raw_frames[keep]], dtype=int)
        read_coords = (*REQUIRED_COORDS, LIKELIHOOD_COORD) if has_likelihood else REQUIRED_COORDS
        for ind_idx, individual in enumerate(individuals):
            for joint_idx, joint in enumerate(bodyparts):
                for coord_idx, coord in enumerate(read_coords):
                    col_values = cam_df.xs((individual, joint, coord), level=(1, 2, 3), axis=1).to_numpy().ravel()
                    arr[ind_idx, cam_idx, frame_indices, joint_idx, coord_idx] = col_values
            if not has_likelihood:
                # A keypoint the annotator left unplaced is NaN in x/y and stays NaN in confidence.
                placed = ~np.isnan(arr[ind_idx, cam_idx, frame_indices, :, 0])
                conf = arr[ind_idx, cam_idx, frame_indices, :, 2]
                arr[ind_idx, cam_idx, frame_indices, :, 2] = np.where(placed, MANUAL_CONFIDENCE, conf)
        logging.info(f"Camera '{napari_cam}': filled {int(keep.sum())} of {len(raw_frames)} annotated frames")

    logging.info(f"Filled array of shape {arr.shape}")

    data_description = {
        "2d": {
            "axis0": subjects,
            "axis1": cameras,
            "axis2": frame_labels,
            "axis3": list(bodyparts),
            "axis4": ["coordinate_x", "coordinate_y", "confidence_score"],
        },
    }
    npz_dict = {
        "2d": arr,
        "data_description": np.array(data_description, dtype=object),
    }

    sequence.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(sequence.output, **npz_dict)
    logging.info(f"Saved body_joints NPZ to: {sequence.output}")
