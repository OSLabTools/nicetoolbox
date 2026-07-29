"""Two-view stereo triangulation for SAM 3D Body body_joints exports."""

import logging
from typing import Any

import numpy as np

from ....utils import check_and_exception as check
from ....utils import triangulation as tri
from ..mmpose import pose_utils


def calibration_usable_for_stereo_triangulation(
    calibration: dict[str, Any] | None, triangulation_cameras: list[str]
) -> bool:
    """Intrinsics and projection for both cameras of the configured stereo pair."""
    if not calibration or not triangulation_cameras:
        return False
    for cam in triangulation_cameras:
        c = calibration.get(cam)
        if not c:
            return False
        if "intrinsic_matrix" not in c or "projection_matrix" not in c:
            return False
        K = np.asarray(c["intrinsic_matrix"])
        P = np.asarray(c["projection_matrix"])
        if K.size == 0 or P.size == 0:
            return False
    return True


def _distortion_vec(cal_entry: dict[str, Any]) -> np.ndarray:
    d = cal_entry.get("distortions")
    if d is None:
        return np.zeros(5, dtype=np.float64)
    return np.asarray(d, dtype=np.float64).ravel()


def prepare_2d_for_triangulation(arr_2d: np.ndarray, min_confidence: float) -> np.ndarray:
    """Low-confidence 2D coords → NaN, then temporal interpolation (pose_utils.interpolate_data)."""
    work = np.array(arr_2d, dtype=np.float64, copy=True)
    if work.shape[-1] >= 3:
        low = work[..., 2] < min_confidence
        work[low, 0:2] = np.nan
    return pose_utils.interpolate_data(work, is_3d=False)


def triangulate_stereo_body_joints_from_two_cameras(
    arr_2d_interpolated: np.ndarray,
    *,
    calibration: dict[str, Any],
    camera_names: list[str],
    triangulation_cameras: list[str],
    cam_sees_subjects: dict[str, list[int]],
) -> np.ndarray | None:
    """Stereo 3d column (n_sub, 1, n_frames, n_kp, 4); NaN when subject missing in either camera."""
    c0, c1 = triangulation_cameras
    missing = [cam for cam in triangulation_cameras if cam not in camera_names]
    if missing:
        raise ValueError(
            f"triangulation.triangulation_cameras {missing} are not among the resolved cameras {camera_names}."
        )
    cam0_idx = camera_names.index(c0)
    cam1_idx = camera_names.index(c1)
    common = sorted(set(cam_sees_subjects.get(c0, [])) & set(cam_sees_subjects.get(c1, [])))
    if not common:
        logging.info(
            "SAM 3D Body stereo triangulation: no subjects shared between %s and %s; skipping 3d fill.",
            c0,
            c1,
        )
        return None

    cal0 = calibration[c0]
    cal1 = calibration[c1]
    K0 = np.asarray(cal0["intrinsic_matrix"], dtype=np.float64)
    K1 = np.asarray(cal1["intrinsic_matrix"], dtype=np.float64)
    P0 = np.asarray(cal0["projection_matrix"], dtype=np.float64)
    P1 = np.asarray(cal1["projection_matrix"], dtype=np.float64)
    d0 = _distortion_vec(cal0)
    d1 = _distortion_vec(cal1)

    n_sub, n_cams, n_frames, n_kp, _ = arr_2d_interpolated.shape
    if n_cams < 2:
        return None
    out = np.full((n_sub, 1, n_frames, n_kp, 4), np.nan, dtype=np.float64)

    for subj in common:
        person_cam1 = arr_2d_interpolated[subj, cam0_idx, :, :, :]
        person_cam2 = arr_2d_interpolated[subj, cam1_idx, :, :, :]

        xy_points_cam1 = person_cam1[:, :, :2].reshape(-1, 1, 2)
        xy_points_cam2 = person_cam2[:, :, :2].reshape(-1, 1, 2)
        conf_cam1 = person_cam1[:, :, 2].reshape(-1, 1, 1)
        conf_cam2 = person_cam2[:, :, 2].reshape(-1, 1, 1)

        nan_mask_cam1 = np.isnan(xy_points_cam1).any(axis=2)
        nan_mask_cam2 = np.isnan(xy_points_cam2).any(axis=2)
        combined_nan_mask = nan_mask_cam1 | nan_mask_cam2

        filtered_xy_points_cam1 = xy_points_cam1[~combined_nan_mask]
        filtered_xy_points_cam2 = xy_points_cam2[~combined_nan_mask]
        filtered_confidence_cam1 = conf_cam1[~combined_nan_mask]
        filtered_confidence_cam2 = conf_cam2[~combined_nan_mask]

        if filtered_xy_points_cam1.shape[0] == 0:
            reshaped_3d = np.full((person_cam1.shape[0], person_cam1.shape[1], 4), np.nan)
        else:
            cam1_u = np.squeeze(tri.undistort_points_pinhole(filtered_xy_points_cam1, K0, d0))
            cam2_u = np.squeeze(tri.undistort_points_pinhole(filtered_xy_points_cam2, K1, d1))
            person_data_3d = tri.triangulate_stereo(P0, P1, cam1_u.T, cam2_u.T)
            confidence_combined = np.minimum(filtered_confidence_cam1, filtered_confidence_cam2)
            person_data_3d_with_conf = np.concatenate([person_data_3d.T, confidence_combined], axis=1)
            output_shape = (xy_points_cam1.shape[0], 4)
            output_data_3d = np.full(output_shape, np.nan)
            output_data_3d[~combined_nan_mask.reshape(-1)] = person_data_3d_with_conf
            reshaped_3d = output_data_3d.reshape(person_cam1.shape[0], person_cam1.shape[1], 4)

        check.check_zeros(reshaped_3d)

        out[subj, 0, :, :, :] = reshaped_3d

    if not np.any(np.isfinite(out)):
        return None
    return out


def apply_stereo_triangulation_to_body_joints_payload(
    body_payload: dict[str, Any],
    *,
    calibration: dict[str, Any] | None,
    camera_names: list[str],
    triangulation_cameras: list[str],
    mode: str,
    cam_sees_subjects: dict[str, list[int]],
    video_start_frame_index: int,
    enabled: bool,
    min_confidence: float,
) -> bool:
    """Replace primary 3d with stereo triangulation when enabled. Returns True if applied."""
    if (
        not enabled
        or mode != "multi"
        or not calibration_usable_for_stereo_triangulation(calibration, triangulation_cameras)
    ):
        return False
    assert calibration is not None

    arr_2d = np.asarray(body_payload["2d"], dtype=np.float64)
    work = prepare_2d_for_triangulation(arr_2d, min_confidence)

    arr_3d = triangulate_stereo_body_joints_from_two_cameras(
        work,
        calibration=calibration,
        camera_names=camera_names,
        triangulation_cameras=triangulation_cameras,
        cam_sees_subjects=cam_sees_subjects,
    )
    if arr_3d is None:
        return False

    dd = body_payload["data_description"].item()
    labels = list(dd.get("2d", {}).get("axis3", []))
    if len(labels) != arr_2d.shape[3]:
        labels = [f"mhr_{i}" for i in range(arr_2d.shape[3])]
    frame_indices = [f"{video_start_frame_index + i:09d}" for i in range(arr_2d.shape[2])]
    subj_names = list(dd.get("2d", {}).get("axis0", []))

    axis_3d: dict[str, Any] = {
        "axis0": subj_names,
        "axis1": ["3d"],
        "axis2": frame_indices,
        "axis3": labels,
        "axis4": ["coordinate_x", "coordinate_y", "coordinate_z", "confidence_score"],
        "space": "world_from_two_view_triangulation",
        "note": (
            f"triangulation_cameras {list(triangulation_cameras)} undistorted and passed to "
            "cv2.triangulatePoints. Rows NaN when subject not in both views."
        ),
    }

    body_payload["3d"] = arr_3d
    body_payload["2d_interpolated"] = work
    dd["3d"] = axis_3d
    dd["2d_interpolated"] = dict(dd["2d"])
    body_payload["data_description"] = np.asarray(dd, dtype=object)
    return True
