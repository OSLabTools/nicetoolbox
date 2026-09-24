import warnings

import numpy as np

from nicetoolbox_core.data.loaded_array import NpzArray

from ..configs.schemas.detectors_config import VideoSubjectAssignmentConfig


def cam_sees_subjects_from_regions(subject_regions: NpzArray) -> dict[str, list[int]]:
    """Per-camera "seen" subject-index lists derived from a loaded subject_regions array.

    A subject counts as seen by a camera when any of its frames on that camera has
    a non-NaN region box.
    """
    data = subject_regions.data  # (n_subjects, n_cameras, n_frames, 4+)
    cameras = subject_regions.axes.cameras
    per_sc = np.any(~np.isnan(data[..., :4]), axis=(2, 3))  # (n_subjects, n_cameras)
    return {cam: np.flatnonzero(per_sc[:, c]).tolist() for c, cam in enumerate(cameras)}


def bboxes_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Axis-aligned [x0, y0, x1, y1] bounding boxes from per-detection 2D keypoints.

    Args:
        landmarks: (n_det, n_kp, 2) or (n_det, n_kp, 3+). Only the first two columns
            of the last axis are used; extra columns (e.g. confidence) are ignored.

    Returns:
        (n_det, 4) float array of [x0, y0, x1, y1].
    """
    landmarks = np.asarray(landmarks, dtype=float)
    xy = landmarks[..., :2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN detection is expected
        x0 = np.nanmin(xy[..., 0], axis=-1)
        y0 = np.nanmin(xy[..., 1], axis=-1)
        x1 = np.nanmax(xy[..., 0], axis=-1)
        y1 = np.nanmax(xy[..., 1], axis=-1)
    return np.stack([x0, y0, x1, y1], axis=-1)


def _iod_matrix(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Compute IoD = |D ∩ R| / |D| for each (detection, region) pair. Vectorized."""
    n_det = boxes.shape[0]
    n_reg = regions.shape[0]
    if n_det == 0 or n_reg == 0:
        return np.zeros((n_det, n_reg), dtype=float)

    d = boxes[:, None, :]  # (n_det, 1, 4)
    r = regions[None, :, :]  # (1, n_reg, 4)

    inter_x0 = np.maximum(d[..., 0], r[..., 0])
    inter_y0 = np.maximum(d[..., 1], r[..., 1])
    inter_x1 = np.minimum(d[..., 2], r[..., 2])
    inter_y1 = np.minimum(d[..., 3], r[..., 3])

    inter_w = np.clip(inter_x1 - inter_x0, 0.0, None)
    inter_h = np.clip(inter_y1 - inter_y0, 0.0, None)
    inter = inter_w * inter_h

    det_area = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    det_area = det_area[:, None]  # (n_det, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        iod = np.where(det_area > 0, inter / det_area, 0.0)

    # NaN region rows mean "subject absent from this camera/frame": zero those columns out.
    region_valid = ~np.any(np.isnan(regions), axis=1)  # (n_reg,)
    iod = iod * region_valid[None, :]
    iod = np.nan_to_num(iod, nan=0.0, posinf=0.0, neginf=0.0)
    return iod


def assign_detections(
    boxes: np.ndarray,
    confidence: np.ndarray | None,
    subject_regions: np.ndarray,
    config: VideoSubjectAssignmentConfig,
) -> np.ndarray:
    """
    Assign each detection to at most one subject slot.

    Args:
        boxes: (n_det, 4) detection boxes as [x0, y0, x1, y1].
        confidence: (n_det,) detection confidences, or None (defaults to 1.0).
        subject_regions: (n_subjects, 4) subject regions; a row of NaN means
            the subject is absent from this camera/frame.
        config: Shared assignment knobs (min_overlap, confidence_weight, ...).

    Returns:
        (n_det,) int array holding the assigned subject index per input
        detection, or -1 where the detection was left unassigned.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n_det = boxes.shape[0]
    subject_regions = np.asarray(subject_regions, dtype=float).reshape(-1, 4)
    n_subj = subject_regions.shape[0]

    assigned = np.full(n_det, -1, dtype=int)
    if n_det == 0 or n_subj == 0:
        return assigned

    if confidence is None:
        confidence = np.ones(n_det, dtype=float)
    else:
        confidence = np.asarray(confidence, dtype=float).reshape(-1)

    # Stable ordering: lexsort so tied cells resolve deterministically.
    order = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
    sorted_boxes = boxes[order]
    sorted_scores = confidence[order]

    iod = _iod_matrix(sorted_boxes, subject_regions)
    iod = np.where(iod >= config.min_overlap, iod, 0.0)

    score_term = config.confidence_weight * sorted_scores[:, None]
    active = iod > 0.0
    ranking = np.where(active, iod + score_term, 0.0)

    # Greedy: repeatedly take the argmax cell, assign, zero its row and column.
    sorted_assign = np.full(n_det, -1, dtype=int)
    while True:
        flat_idx = int(np.argmax(ranking))
        best = ranking.flat[flat_idx]
        if best <= 0.0:
            break
        det_i, subj_j = np.unravel_index(flat_idx, ranking.shape)
        sorted_assign[det_i] = subj_j
        ranking[det_i, :] = 0.0
        ranking[:, subj_j] = 0.0

    # Un-sort back to input order.
    assigned[order] = sorted_assign
    return assigned
