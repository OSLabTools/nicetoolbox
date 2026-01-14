"""
Pose estimation utilities. # TODO: Move to a more appropriate location?
"""

import logging

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
                            small_gaps_valid_idx = np.append(small_gaps_valid_idx, valid_idx[small_gaps_idx[-1] + 1])

                        if small_gaps_valid_idx.size > 1:  # Need at least two points to interpolate
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
                            for gap_start, gap_end in zip(small_gaps_valid_idx[:-1], small_gaps_valid_idx[1:]):
                                data[i, j, gap_start : gap_end + 1, k, 0] = f_x(np.arange(gap_start, gap_end + 1))
                                data[i, j, gap_start : gap_end + 1, k, 1] = f_y(np.arange(gap_start, gap_end + 1))
                                if is_3d:
                                    data[i, j, gap_start : gap_end + 1, k, 2] = f_z(np.arange(gap_start, gap_end + 1))

    return data


def create_iou_all_pairs(data):
    """
    Compute the intersection over union (IoU) of subject bounding boxes across frames and cameras.

    Args:
    data (np.ndarray): Bounding box array with shape
    (#Subject, #Camera, #Frame, 'full_body', BBox), where each BBox
            is represented as [x1, y1, x2, y2, conf].

    Returns:
    iou_array (np.ndarray) :  float32
        IoU laid out as (subject, camera, frame, with_subject).

    """

    if data.shape[-1] != 5:
        logging.error("Last dim must contain bbox values [x1,y1,x2,y2] and confidence score.")
        raise ValueError

    # boxes: (S, C, F, 1, 4) with coords (x1, y1, x2, y2) in the last dim, 4th dimension is "full_body" label
    boxes = data[..., 0, :4].astype(np.float32)
    x1_raw, y1_raw, x2_raw, y2_raw = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = np.minimum(x1_raw, x2_raw)
    x2 = np.maximum(x1_raw, x2_raw)
    y1 = np.minimum(y1_raw, y2_raw)
    y2 = np.maximum(y1_raw, y2_raw)

    w = x2 - x1
    h = y2 - y1
    invalid_mask = (w <= 0) | (h <= 0)
    if np.any(invalid_mask):
        num_invalid = np.sum(invalid_mask)
        logging.error(
            f"Found {num_invalid} boxes with non-positive area." f"The area will be saved as zero for that cases"
        )
    # Areas (safe)
    w = np.maximum(0.0, w)
    h = np.maximum(0.0, h)
    # Area per subject/camera/frame
    area = w * h  # (S,C,F)

    # Pairwise intersections across subjects -> (S,S,C,F)
    # This creates an intersection coordinates matrix for all subjects pairs
    # For two speakers:
    #    |    S1              S2
    # ================================
    # S1 |    S1          max(S1, S2)
    # S2 | max(S2, S1)        S2
    #
    # If we interested in intersection bbox of S1 and S2 speaker
    # It will be (inter_x1, inter_y1) (inter_x2, inter_y2)
    inter_x1 = np.maximum(x1[:, None, :, :], x1[None, :, :, :])
    inter_y1 = np.maximum(y1[:, None, :, :], y1[None, :, :, :])
    inter_x2 = np.minimum(x2[:, None, :, :], x2[None, :, :, :])
    inter_y2 = np.minimum(y2[:, None, :, :], y2[None, :, :, :])

    # Calculate width, height and area of intersection bbox
    # If there is no intersection, difference would be negative
    # We clamp it to 0, to mark no intersection
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h  # (S,S,C,F)

    # We sum pairwise original areas and remove intersections areas from them
    # This result pairwise union areas matrix (S,S,C,F)
    union = area[:, None, :, :] + area[None, :, :, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_sscf = inter / union

    # Reorder to array: (S, C, F, S)
    iou_scfs = np.transpose(iou_sscf, (0, 2, 3, 1)).astype(np.float32)

    return iou_scfs
