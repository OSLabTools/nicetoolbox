"""
Shared 2d keypoint post-processing for method detectors.

Filtering, gap interpolation, stereo triangulation and reprojection are the same chain for every
detector that emits per-camera 2d keypoints, so they live here rather than being copied into each
detector class. All of them take and return NpzArray, so the axes travel with the data.
"""

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL
from nicetoolbox_core.data.loaded_array import NpzArray

from ...utils import triangulation as tri
from .filters import adaptive_savgol_filter
from .pose_utils import interpolate_data


def filter_keypoints(keypoints: NpzArray, filter_config, is_3d: bool = False) -> NpzArray:
    data = keypoints.data.copy()
    # x/y (or x/y/z); the confidence column is not smoothed
    num_coordinates = 3 if is_3d else 2
    # Frame axis is index 2; each coordinate is smoothed independently per keypoint.
    data[..., :num_coordinates] = adaptive_savgol_filter(
        data[..., :num_coordinates],
        filter_config.window_length,
        filter_config.polyorder,
        axis=2,
    )
    return NpzArray(data, keypoints.axes)


def mask_low_confidence_keypoints(keypoints: NpzArray, min_confidence: float) -> NpzArray:
    data = keypoints.data.copy()
    # confidence is the last dim
    low_conf = data[..., -1] < min_confidence
    data[low_conf, :-1] = np.nan
    return NpzArray(data, keypoints.axes)


def interpolate_keypoints(keypoints: NpzArray, interpolation_config, is_3d: bool = False) -> NpzArray:
    data = keypoints.data.copy()
    interpolate_data(data, is_3d=is_3d, max_empty=interpolation_config.max_gap)
    return NpzArray(data, keypoints.axes)


# TODO: support radial distortion from intrinsics!
def reproject_keypoints_mat(keypoints_3d: NpzArray, projection_matrices: dict[str, np.ndarray]) -> NpzArray:
    """
    Project 3d keypoints into camera images with a 3x4 projection matrix per camera.

    Args:
        keypoints_3d (NpzArray): (subjects, cameras, frames, keypoints, x/y/z/conf).
        projection_matrices (dict[str, np.ndarray]): (3, 4) projection matrix per output camera name.

    Returns:
        NpzArray: (subjects, output cameras, frames, keypoints, x/y/conf) in pixels, same confidence.
    """
    axes = keypoints_3d.axes
    camera_names = list(projection_matrices)
    broadcast = len(axes.cameras) == 1
    if not broadcast and list(axes.cameras) != camera_names:
        raise ValueError(
            f"Cannot project keypoints of cameras {list(axes.cameras)} with projection matrices for "
            f"{camera_names}: pass one input camera to broadcast, or the same cameras in the same order."
        )

    n_subjects, _, n_frames, n_keypoints, _ = keypoints_3d.data.shape
    projected = np.full((n_subjects, len(camera_names), n_frames, n_keypoints, 3), np.nan)
    for cam_idx, camera_name in enumerate(camera_names):
        source = keypoints_3d.data[:, 0 if broadcast else cam_idx]
        projected[:, cam_idx, ..., :2] = tri.project_points_to_camera(source[..., :3], projection_matrices[camera_name])
        # confidence belongs to the 3d origin, copied to the view
        projected[:, cam_idx, ..., 2] = source[..., 3]

    axes_2d = VECTOR_2D_CONF_PER_LABEL.make_axes(axes.subjects, camera_names, axes.frames, labels=axes.labels)
    return NpzArray(projected, axes_2d)


# TODO: support radial distortion from intrinsics!
def reproject_keypoints(keypoints_3d: NpzArray, calibration: dict, camera_names: list[str]) -> NpzArray:
    """
    Project world 3d keypoints into calibrated camera views.

    Args:
        keypoints_3d (NpzArray): (subjects, 1, frames, keypoints, x/y/z/conf) in world coordinates.
        calibration (dict): Calibration per camera name, each with a (3, 4) "projection_matrix".
        camera_names (list[str]): Views to project into; they become the output camera axis.

    Returns:
        NpzArray: (subjects, cameras, frames, keypoints, x/y/conf) in pixels, same confidence.
    """
    projection_matrices = {}
    for camera_name in camera_names:
        projection_matrices[camera_name] = calibration[camera_name]["projection_matrix"]
    return reproject_keypoints_mat(keypoints_3d, projection_matrices)


def extract_key_per_value(input_dict):
    """
    Extracts keys from a dictionary based on the type of their values.

    If all values in the dictionary are integers, it returns a list of keys.
    If any value is a list, it appends an index to the key to create a unique key.

    Args:
        input_dict (dict): The input dictionary to extract keys from.

    Returns:
        return_keys (list): A list of keys extracted from the input dictionary.

    Raises:
        NotImplementedError: If a value in the dictionary is neither an integer nor a
        list.
    """
    if all(isinstance(val, int) for val in list(input_dict.values())):
        return list(input_dict.keys())
    return_keys = []
    for key, value in input_dict.items():
        if isinstance(value, int):
            return_keys.append(value)
        elif isinstance(value, list):
            for idx, _ in enumerate(value):
                return_keys.append(f"{key}_{idx}")
        else:
            raise NotImplementedError
    return return_keys
