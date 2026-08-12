"""
Shared 2d keypoint post-processing for method detectors.

Filtering, gap interpolation, stereo triangulation and reprojection are the same chain for every
detector that emits per-camera 2d keypoints, so they live here rather than being copied into each
detector class. All of them take and return NpzArray, so the axes travel with the data.
"""

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_PER_LABEL
from nicetoolbox_core.data.loaded_array import NpzArray

from ...utils import triangulation as tri
from .filters import adaptive_savgol_filter
from .pose_utils import interpolate_data


def filter_keypoints(keypoints_2d: NpzArray, filter_config) -> NpzArray:
    data = keypoints_2d.data.copy()
    # Frame axis is index 2; x and y are smoothed independently per keypoint.
    data[..., :2] = adaptive_savgol_filter(
        data[..., :2],
        filter_config.window_length,
        filter_config.polyorder,
        axis=2,
    )
    return NpzArray(data, keypoints_2d.axes)


def interpolate_keypoints(keypoints_2d: NpzArray, interpolation_config) -> NpzArray:
    # TODO: create reusable interpolation, not reuse mmpose
    data = keypoints_2d.data.copy()
    interpolate_data(data, is_3d=False, max_empty=interpolation_config.max_gap)
    return NpzArray(data, keypoints_2d.axes)


def reproject_keypoints(keypoints_3d: NpzArray, calibration: dict, camera_names: list[str]) -> NpzArray:
    # camera_names is the only thing not on the input: its camera axis is the "3d" pseudo-camera,
    # so the views to project back into have to be named by the caller.
    axes = keypoints_3d.axes
    # Drop the pseudo-camera axis; the world point is the same for every view.
    world_xyz = keypoints_3d.data[:, 0, :, :, :3]
    n_subjects, n_frames, n_keypoints, _ = world_xyz.shape

    reprojected = np.full((n_subjects, len(camera_names), n_frames, n_keypoints, 2), np.nan)
    for cam_idx, camera_name in enumerate(camera_names):
        projection_matrix = calibration[camera_name]["projection_matrix"]
        reprojected[:, cam_idx] = tri.project_points_to_camera(world_xyz, projection_matrix)

    axes_2d = VECTOR_2D_PER_LABEL.make_axes(axes.subjects, camera_names, axes.frames, labels=axes.labels)
    return NpzArray(reprojected, axes_2d)


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
