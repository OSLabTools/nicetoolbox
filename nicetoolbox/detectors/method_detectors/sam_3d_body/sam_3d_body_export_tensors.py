"""
Build MMPose-style dense tensors from SAM 3D Body per_frame_outputs for .npz export.
"""

from typing import Any, Literal

import numpy as np

from ...utils import pose_utils


def frame_person_list(frame_bundle: Any, cam: str, cam_idx: int, mode: str) -> list[dict[str, Any]]:
    if mode == "multi" and isinstance(frame_bundle, dict):
        return list(frame_bundle.get(cam, []))
    if mode == "single" and isinstance(frame_bundle, list) and cam_idx == 0:
        return list(frame_bundle)
    return []


def _max_vertices(per_frame: list[Any], camera_names: list[str], mode: str) -> int:
    m = 0
    for fb in per_frame:
        for ci, cam in enumerate(camera_names):
            for p in frame_person_list(fb, cam, ci, mode):
                v = p.get("pred_vertices")
                if v is None:
                    continue
                m = max(m, int(np.asarray(v).shape[0]))
    return m


def _kp2_to_xy_conf(kp: np.ndarray) -> np.ndarray:
    a = np.asarray(kp, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 2:
        return np.zeros((0, 3), dtype=np.float64)
    if a.shape[1] == 2:
        conf = np.ones((a.shape[0], 1), dtype=np.float64)
        return np.concatenate([a, conf], axis=1)
    return a[:, :3]


def _kp3_to_xyz_conf(kp3: np.ndarray, conf_from_2d: np.ndarray | None) -> np.ndarray:
    a = np.asarray(kp3, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 3:
        return np.zeros((0, 4), dtype=np.float64)
    j = a.shape[0]
    if conf_from_2d is not None and conf_from_2d.shape[0] == j:
        c = np.asarray(conf_from_2d, dtype=np.float64).reshape(-1, 1)[:, :1]
    else:
        c = np.ones((j, 1), dtype=np.float64)
    return np.concatenate([a[:, :3], c], axis=1)


def _number_subject_slots(cam_sees_subjects: dict[str, list[int]], camera_names: list[str]) -> int:
    if not camera_names:
        return 1
    return max(len(cam_sees_subjects.get(c, [])) for c in camera_names)


def _subject_axis_names(number_subjects: int, subjects_descr: list[str]) -> list[str]:
    out: list[str] = []
    for i in range(number_subjects):
        if i < len(subjects_descr):
            out.append(str(subjects_descr[i]))
        else:
            out.append(f"subject_{i}")
    return out


def build_body_joints_npz_payload(
    per_frame: list[Any],
    *,
    joint_indices: list[int],
    joint_names: list[str],
    camera_names: list[str],
    mode: str,
    subjects_descr: list[str],
    cam_sees_subjects: dict[str, list[int]],
    video_start_frame_index: int,
    three_d_primary: Literal["world", "camera"] = "world",
) -> dict[str, Any]:
    """Build joints NPZ payload (MMPose-style keys). three_d_primary: world or camera-native 3d."""
    n_frames = len(per_frame)
    n_cams = len(camera_names)
    n_sub = _number_subject_slots(cam_sees_subjects, camera_names)
    n_kp = len(joint_indices)

    arr_2d = np.full((n_sub, n_cams, n_frames, n_kp, 3), np.nan, dtype=np.float64)
    arr_3d = np.full((n_sub, n_cams, n_frames, n_kp, 4), np.nan, dtype=np.float64)
    arr_3d_w = np.full((n_sub, 1, n_frames, n_kp, 4), np.nan, dtype=np.float64)
    bbox = np.full((n_sub, n_cams, n_frames, 1, 5), np.nan, dtype=np.float64)
    any_world = False

    idx_arr = np.asarray(joint_indices, dtype=np.intp)

    for fi, fb in enumerate(per_frame):
        for ci, cam in enumerate(camera_names):
            plist = frame_person_list(fb, cam, ci, mode)
            slots = cam_sees_subjects.get(cam, [])
            for pid, person in enumerate(plist):
                if pid >= len(slots):
                    continue
                subj = int(slots[pid])
                if subj < 0 or subj >= n_sub:
                    continue
                kp2_full = _kp2_to_xy_conf(person.get("pred_keypoints_2d", np.zeros((0, 3))))
                kp3_full = np.asarray(person.get("pred_keypoints_3d"), dtype=np.float64)
                kp3w_full = person.get("pred_keypoints_3d_world")
                bb = np.asarray(person.get("bbox"), dtype=np.float64).ravel()
                if n_kp > 0 and kp2_full.shape[0] > 0:
                    valid = idx_arr[idx_arr < kp2_full.shape[0]]
                    arr_2d[subj, ci, fi, : len(valid), :] = kp2_full[valid]
                if n_kp > 0 and kp3_full.size > 0 and kp3_full.ndim == 2 and kp3_full.shape[1] >= 3:
                    valid3 = idx_arr[idx_arr < kp3_full.shape[0]]
                    conf2 = kp2_full[valid3, 2] if kp2_full.shape[0] > 0 else None
                    arr_3d[subj, ci, fi, : len(valid3), :] = _kp3_to_xyz_conf(kp3_full[valid3], conf2)
                if n_kp > 0 and kp3w_full is not None:
                    w = np.asarray(kp3w_full, dtype=np.float64)
                    if w.ndim == 2 and w.shape[1] >= 3:
                        any_world = True
                        validw = idx_arr[idx_arr < w.shape[0]]
                        conf2 = kp2_full[validw, 2] if kp2_full.shape[0] > 0 else None
                        arr_3d_w[subj, 0, fi, : len(validw), :] = _kp3_to_xyz_conf(w[validw], conf2)
                if bb.size >= 4:
                    bbox[subj, ci, fi, 0, :4] = bb[:4]
                    bbox[subj, ci, fi, 0, 4] = 1.0

    frame_indices = [f"{video_start_frame_index + i:09d}" for i in range(n_frames)]
    subj_names = _subject_axis_names(n_sub, subjects_descr)
    joint_labels = list(joint_names)

    axis_2d = {
        "axis0": subj_names,
        "axis1": list(camera_names),
        "axis2": frame_indices,
        "axis3": joint_labels,
        "axis4": ["coordinate_x", "coordinate_y", "confidence_score"],
    }
    axis_3d_cam = {
        "axis0": subj_names,
        "axis1": list(camera_names),
        "axis2": frame_indices,
        "axis3": joint_labels,
        "axis4": ["coordinate_x", "coordinate_y", "coordinate_z", "confidence_score"],
        "space": "sam_camera_or_model",
    }
    axis_3d_world_slot = {
        "axis0": subj_names,
        "axis1": ["3d"],
        "axis2": frame_indices,
        "axis3": joint_labels,
        "axis4": ["coordinate_x", "coordinate_y", "coordinate_z", "confidence_score"],
        "space": "world_from_calibration",
    }
    axis_3d_world_broadcast = {
        "axis0": subj_names,
        "axis1": list(camera_names),
        "axis2": frame_indices,
        "axis3": joint_labels,
        "axis4": ["coordinate_x", "coordinate_y", "coordinate_z", "confidence_score"],
        "space": "world_from_calibration",
        "note": "Same world coords repeated per camera view for tensor shape alignment.",
    }

    data_description: dict[str, Any] = {
        "2d": axis_2d,
        "2d_interpolated": dict(axis_2d),
        "2d_filtered": dict(axis_2d),
        "bbox_2d": {
            "axis0": subj_names,
            "axis1": list(camera_names),
            "axis2": frame_indices,
            "axis3": ["full_body"],
            "axis4": ["top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y", "confidence_score"],
        },
        "bbox_overlap": {
            "axis0": subj_names,
            "axis1": list(camera_names),
            "axis2": frame_indices,
            "axis3": [f"with_{s}" for s in subj_names],
        },
        "sam_3d_body": {
            "description": (
                "SAM 3D Body (MHR) export aligned with body_joints / hand_joints / "
                "facial_landmarks tensor conventions (Subject, Camera, Frame, Labels, Data)."
            ),
            "mhr": "Keypoint rows follow native MHR vertex order (axis3 labels mhr_0 …).",
            "three_d_primary": three_d_primary,
        },
    }

    iou = pose_utils.create_iou_all_pairs(bbox)

    out: dict[str, Any] = {
        "2d": arr_2d,
        "2d_interpolated": np.array(arr_2d, copy=True),
        "2d_filtered": np.array(arr_2d, copy=True),
        "bbox_2d": bbox,
        "bbox_overlap": iou,
    }

    if three_d_primary == "world":
        if any_world and np.any(np.isfinite(arr_3d_w)):
            wb = np.repeat(arr_3d_w, n_cams, axis=1) if n_cams else arr_3d_w
            out["3d"] = wb
            data_description["3d"] = axis_3d_world_broadcast
            data_description["3d_world"] = axis_3d_world_slot
            out["3d_world"] = arr_3d_w
            data_description["3d_camera"] = axis_3d_cam
            out["3d_camera"] = arr_3d
        elif np.any(np.isfinite(arr_3d)):
            out["3d"] = arr_3d
            data_description["3d"] = axis_3d_cam
    else:
        if np.any(np.isfinite(arr_3d)):
            out["3d"] = arr_3d
            data_description["3d"] = axis_3d_cam
        if any_world and np.any(np.isfinite(arr_3d_w)):
            data_description["3d_world"] = axis_3d_world_slot
            out["3d_world"] = arr_3d_w

    out["data_description"] = np.asarray(data_description, dtype=object)
    return out


def build_body_mesh_npz_payload(
    per_frame: list[Any],
    *,
    camera_names: list[str],
    mode: str,
    faces: np.ndarray,
    subjects_descr: list[str],
    cam_sees_subjects: dict[str, list[int]],
    video_start_frame_index: int,
    save_vertices: bool,
) -> dict[str, Any]:
    n_frames = len(per_frame)
    n_cams = len(camera_names)
    n_sub = _number_subject_slots(cam_sees_subjects, camera_names)
    n_v = _max_vertices(per_frame, camera_names, mode) if save_vertices else 0

    verts = (
        np.full((n_sub, n_cams, n_frames, max(n_v, 1), 3), np.nan, dtype=np.float64)
        if save_vertices and n_v > 0
        else np.zeros((n_sub, n_cams, n_frames, 0, 3), dtype=np.float64)
    )
    verts_w = (
        np.full((n_sub, n_cams, n_frames, max(n_v, 1), 3), np.nan, dtype=np.float64)
        if save_vertices and n_v > 0
        else np.zeros((n_sub, n_cams, n_frames, 0, 3), dtype=np.float64)
    )
    any_w = False

    if save_vertices and n_v > 0:
        for fi, fb in enumerate(per_frame):
            for ci, cam in enumerate(camera_names):
                plist = frame_person_list(fb, cam, ci, mode)
                slots = cam_sees_subjects.get(cam, [])
                for pid, person in enumerate(plist):
                    if pid >= len(slots):
                        continue
                    subj = int(slots[pid])
                    if subj < 0 or subj >= n_sub:
                        continue
                    v = np.asarray(person.get("pred_vertices"), dtype=np.float64)
                    if v.size == 0 or v.ndim != 2 or v.shape[1] < 3:
                        continue
                    take = min(n_v, v.shape[0])
                    verts[subj, ci, fi, :take, :] = v[:take, :3]
                    vw = person.get("pred_vertices_world")
                    if vw is not None:
                        w = np.asarray(vw, dtype=np.float64)
                        if w.ndim == 2 and w.shape[1] >= 3:
                            any_w = True
                            take_w = min(n_v, w.shape[0])
                            verts_w[subj, ci, fi, :take_w, :] = w[:take_w, :3]

    frame_indices = [f"{video_start_frame_index + i:09d}" for i in range(n_frames)]
    subj_names = _subject_axis_names(n_sub, subjects_descr)
    mesh_vertex_axis3 = [f"mesh_vertex_{i}" for i in range(max(n_v, 1))] if n_v > 0 else ["mesh_vertex_0"]
    data_description: dict[str, Any] = {
        "faces": {
            "axis0": "triangle_row",
            "axis1": ["vertex_i", "vertex_j", "vertex_k"],
            "note": "MHR mesh faces (int32 indices into vertices axis3).",
        },
        "vertices": {
            "axis0": subj_names,
            "axis1": list(camera_names),
            "axis2": frame_indices,
            "axis3": mesh_vertex_axis3,
            "axis4": ["coordinate_x", "coordinate_y", "coordinate_z"],
            "note": "Last dimension indexes mesh template vertices 0 … V-1.",
            "csv_skip": True,
        },
        "sam_3d_body": {
            "description": "SAM 3D Body dense mesh vertices per subject, camera, and frame.",
        },
    }
    if any_w and verts_w.size > 0:
        data_description["vertices_world"] = {
            "axis0": subj_names,
            "axis1": list(camera_names),
            "axis2": frame_indices,
            "axis3": mesh_vertex_axis3,
            "axis4": ["coordinate_x", "coordinate_y", "coordinate_z"],
            "csv_skip": True,
        }
    out_mesh: dict[str, Any] = {
        "faces": np.asarray(faces, dtype=np.int32),
        "vertices": verts,
        "data_description": np.asarray(data_description, dtype=object),
    }
    if any_w and verts_w.size > 0:
        out_mesh["vertices_world"] = verts_w
    return out_mesh
