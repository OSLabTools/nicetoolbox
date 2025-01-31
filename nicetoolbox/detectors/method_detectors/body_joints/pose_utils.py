"""
Pose estimation utilities. # TODO: Move to a more appropriate location?
"""

import numpy as np
import scipy.interpolate as interp


def interpolate_data(data, is_3d=True, max_empty=10):  # TODO make max_empty 1/3 of FPS
    """
    Interpolates missing data in the given multi-dimensional array using scipy's
    interp1d function.

    Args:
        data (ndarray): The input data array with shape
            (num_persons, num_cameras, num_frames, num_keypoints, _).
        is_3d (bool, optional): Indicates whether the data is 3D or not.
            Defaults to True.
        max_empty (int, optional): The maximum number of consecutive empty frames
            allowed. Defaults to 10.

    Returns:
        ndarray: The interpolated data array with the same shape as the input data.

    """
    num_people, num_cameras, _, num_keypoints, _ = data.shape
    for i in range(num_people):
        for j in range(num_cameras):
            for k in range(num_keypoints):
                x = data[i, j, :, k, 0]
                y = data[i, j, :, k, 1]
                z = None
                if is_3d:
                    z = data[i, j, :, k, 2]

                # Check for NaNs and only proceed if there are any
                if np.isnan(x).any() or np.isnan(y).any():
                    valid = ~np.isnan(x)
                    valid_idx = np.where(valid)[0]

                    # Check gaps in valid indices and filter out large gaps
                    if valid_idx.size > 1:
                        gaps = np.diff(valid_idx)
                        small_gaps_idx = np.where(gaps <= max_empty)[0]
                        small_gaps_valid_idx = valid_idx[small_gaps_idx]

                        if small_gaps_valid_idx.size > 0:
                            small_gaps_valid_idx = np.append(
                                small_gaps_valid_idx, valid_idx[small_gaps_idx[-1] + 1]
                            )

                        if (
                            small_gaps_valid_idx.size > 1
                        ):  # Need at least two points to interpolate
                            # Create interpolation functions for bounded regions
                            f_x = interp.interp1d(
                                small_gaps_valid_idx,
                                x[small_gaps_valid_idx],
                                kind="linear",
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                            f_y = interp.interp1d(
                                small_gaps_valid_idx,
                                y[small_gaps_valid_idx],
                                kind="linear",
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                            f_z = None
                            if is_3d:
                                f_z = interp.interp1d(
                                    small_gaps_valid_idx,
                                    z[small_gaps_valid_idx],
                                    kind="linear",
                                    bounds_error=False,
                                    fill_value=np.nan,
                                )

                            # Apply interpolation only within the gaps
                            for gap_start, gap_end in zip(
                                small_gaps_valid_idx[:-1], small_gaps_valid_idx[1:]
                            ):
                                data[i, j, gap_start : gap_end + 1, k, 0] = f_x(
                                    np.arange(gap_start, gap_end + 1)
                                )
                                data[i, j, gap_start : gap_end + 1, k, 1] = f_y(
                                    np.arange(gap_start, gap_end + 1)
                                )
                                if is_3d:
                                    data[i, j, gap_start : gap_end + 1, k, 2] = f_z(
                                        np.arange(gap_start, gap_end + 1)
                                    )

    return data
