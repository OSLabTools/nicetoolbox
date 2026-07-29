"""
Method detector: SAM 3D Body (Hugging Face). Subprocess runs GPU inference;
post-processing and visualization run in the main environment.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nicetoolbox_core.video_loaders import ImagePathsByCameraLoader

from ....configs.schemas.detectors_instances_configs import MethodDetectorRuntime, Sam3dBodyConfig
from ....configs.schemas.predictions_mapping import Sam3dBodyMhr
from ....utils import video as vd
from ..base_method import BaseMethod
from ..filters import adaptive_savgol_filter
from . import sam_3d_body_mesh_viz as mesh_viz
from .sam_3d_body_export_tensors import build_body_joints_npz_payload, build_body_mesh_npz_payload, frame_person_list
from .sam_3d_body_paths import RAW_INFERENCE_NPZ_NAME, SAM3D_BODY_LOCAL_NPZ_STEM, SAM3D_BODY_OUTPUT_NPZ_STEM
from .sam_3d_body_stereo_triangulation import apply_stereo_triangulation_to_body_joints_payload


def _calibration_usable_for_world_alignment(calibration: dict[str, Any] | None, camera_names: list[str]) -> bool:
    """True when every requested camera has intrinsics + projection matrix (world alignment path)."""
    if not calibration or not camera_names:
        return False
    for cam in camera_names:
        if cam not in calibration:
            return False
        c = calibration[cam]
        if "intrinsic_matrix" not in c or "projection_matrix" not in c:
            return False
        K = np.asarray(c["intrinsic_matrix"])
        P = np.asarray(c["projection_matrix"])
        if K.size == 0 or P.size == 0:
            return False
    return True


def _bundle_camera_params(
    calibration: dict[str, Any] | None, camera_names: list[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    notes = {
        "convention": "Per camera: intrinsic_matrix 3x3, distortions, projection_matrix 3x4 as in NICE calibration.",
        "source": "Sequence calibration from VideoDataHandler.",
    }
    if not calibration:
        return {}, notes
    out: dict[str, Any] = {}
    for cam in camera_names:
        if cam not in calibration:
            continue
        c = calibration[cam]
        entry: dict[str, Any] = {}
        for key in ("intrinsic_matrix", "distortions", "projection_matrix", "image_size"):
            if key in c and c[key] is not None:
                entry[key] = np.asarray(c[key], dtype=np.float64)
        if entry:
            out[cam] = entry
    return out, notes


def _projection_to_world_from_cam_rt(K: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K_inv = np.linalg.inv(K)
    Rt = K_inv @ P
    R = Rt[:, :3].astype(np.float64)
    t = Rt[:, 3].astype(np.float64)
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1.0
        R = u @ vt
    return R, t


def _camera_points_to_world(X_cam: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    Xc = np.asarray(X_cam, dtype=np.float64)
    R = np.asarray(R_w2c, dtype=np.float64)
    t = np.asarray(t_w2c, dtype=np.float64).reshape(3)
    return (R.T @ (Xc.reshape(-1, 3).T - t.reshape(3, 1))).T.reshape(Xc.shape)


def _smooth_time_series(
    data: np.ndarray,
    window_length: int,
    polyorder: int,
    axis: int = 0,
) -> np.ndarray:
    return adaptive_savgol_filter(data, window_length, polyorder, axis=axis, mode="interp")


def _validate_sam3d_npz(raw: Any) -> None:
    required = ("faces", "frames_meta", "per_frame_outputs", "camera_names_order")
    for k in required:
        if k not in raw.files:
            raise ValueError(f"SAM 3D raw npz missing key '{k}'")


def _sort_persons_by_bbox_x(person_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(person_dicts) <= 1:
        return person_dicts
    bboxes = [np.asarray(p["bbox"], dtype=np.float64) for p in person_dicts]
    centers = [(b[0] + b[2]) / 2.0 for b in bboxes]
    order = np.argsort(np.array(centers))
    return [person_dicts[int(i)] for i in order]


def _apply_temporal_smoothing_to_packed(
    per_frame: list[Any],
    window_length: int,
    polyorder: int,
    param_keys: tuple[str, ...] = (
        "global_rot",
        "body_pose_params",
        "hand_pose_params",
        "shape_params",
        "scale_params",
        "expr_params",
    ),
) -> list[Any]:
    if not per_frame:
        return per_frame
    n_frames = len(per_frame)
    n_persons = max((len(pf) for pf in per_frame if isinstance(pf, list)), default=0)
    if n_persons == 0:
        return per_frame
    out_frames: list[list[dict[str, Any]]] = []
    for fi in range(n_frames):
        plist = per_frame[fi]
        if not isinstance(plist, list):
            out_frames.append([])
            continue
        out_frames.append([dict(p) for p in plist])

    for pid in range(n_persons):
        for key in param_keys:
            series = []
            for fi in range(n_frames):
                plist = per_frame[fi]
                if pid >= len(plist):
                    series.append(None)
                    continue
                v = plist[pid].get(key)
                series.append(np.asarray(v, dtype=np.float64) if v is not None else None)
            valid = [s for s in series if s is not None]
            if len(valid) < 3:
                continue
            ref_shape = valid[0].shape
            try:
                stack = np.stack(
                    [series[i] if series[i] is not None else np.full(ref_shape, np.nan) for i in range(n_frames)],
                    axis=0,
                )
            except ValueError:
                continue
            smoothed = _smooth_time_series(stack, window_length, polyorder, axis=0)
            for fi in range(n_frames):
                if pid < len(out_frames[fi]) and series[fi] is not None:
                    out_frames[fi][pid][key] = smoothed[fi].astype(np.float32)
    return out_frames


def _world_align_packed_frame(
    person: dict[str, Any],
    K: np.ndarray,
    P: np.ndarray,
) -> dict[str, Any]:
    out = dict(person)
    try:
        R, t = _projection_to_world_from_cam_rt(K, P)
    except Exception as e:
        logging.debug("World align skip (decompose failed): %s", e)
        return out
    for src, dst in (("pred_keypoints_3d", "pred_keypoints_3d_world"), ("pred_vertices", "pred_vertices_world")):
        if src not in person:
            continue
        arr = np.asarray(person[src], dtype=np.float64)
        out[dst] = _camera_points_to_world(arr, R, t).astype(np.float32)
    return out


def _cross_view_mean_distance(p0: dict[str, Any], p1: dict[str, Any], key: str = "pred_keypoints_3d_world") -> float:
    if key not in p0 or key not in p1:
        return float("nan")
    a = np.asarray(p0[key], dtype=np.float64)
    b = np.asarray(p1[key], dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    d = np.linalg.norm(a - b, axis=-1)
    return float(np.nanmean(d))


def _strip_3d_from_body_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop every 3D array (and its data_description entry), keeping 2D only.

    Used for the body_joints export when no calibration is available: world-space
    3D can't be produced, and camera-space 3D belongs in body_joints_local, so the
    calibration-free body_joints NPZ carries 2D keypoints only.
    """
    out = dict(payload)
    dropped: list[str] = []
    for key in list(out.keys()):
        if key == "data_description":
            continue
        kl = key.lower()
        if kl.startswith("3d") or kl.endswith("_3d") or "_3d_" in kl:
            out.pop(key, None)
            dropped.append(key)
    dd_obj = out.get("data_description")
    if dd_obj is not None:
        try:
            dd = dd_obj.item() if hasattr(dd_obj, "item") else dd_obj
            if isinstance(dd, dict):
                for key in dropped:
                    dd.pop(key, None)
                out["data_description"] = np.asarray(dd, dtype=object)
        except Exception as e:
            logging.debug("strip_3d: data_description cleanup skipped: %s", e)
    if dropped:
        logging.info("SAM 3D Body: body_joints 2D-only export, dropped 3D keys %s", dropped)
    return out


def _attach_body_meta(
    body_payload: dict[str, Any],
    *,
    mhr_mapping: Sam3dBodyMhr,
    bundle_notes: dict[str, str],
    cross_view: bool,
    cross_view_residuals: list[float],
    raw: Any,
    export_policy: dict[str, Any],
) -> dict[str, Any]:
    dd = body_payload["data_description"].item()
    sam_block = dd.setdefault("sam_3d_body", {})
    sam_block.update(
        {
            "camera_params_bundle": bundle_notes,
            "sam_3d_body_mhr_mapping": {
                "config_section": "human_pose.sam_3d_body_mhr",
                "coco_body_17_joint_names": list(mhr_mapping.keypoints_index.body.keys()),
                "coco_body_17_mhr70_index": list(mhr_mapping.keypoints_index.body.values()),
            },
            "cross_view": {
                "enabled": cross_view,
                "mean_keypoint_world_residual": float(np.nanmean(cross_view_residuals))
                if cross_view_residuals
                else float("nan"),
                "note": "Cross-camera person index; world coords from calibration decomposition.",
            },
            "export_policy": export_policy,
        }
    )
    if "data_description" in raw.files:
        try:
            prev = raw["data_description"].item()
            if isinstance(prev, dict):
                sam_block["inference_stage"] = prev
        except Exception:
            pass
    body_payload["data_description"] = np.asarray(dd, dtype=object)
    return body_payload


def _run_sam3d_post_process(
    raw_npz_path: Path,
    *,
    out_body_joints_npz: Path | None,
    out_body_joints_local_npz: Path,
    out_hand_joints_npz: Path | None,
    out_hand_joints_local_npz: Path,
    out_body_mesh_npz: Path,
    calibration: dict[str, Any] | None,
    camera_names: list[str],
    temporal_smooth: bool,
    smooth_window_length: int,
    smooth_polyorder: int,
    world_align: bool,
    cross_view: bool,
    bbox_sort: bool,
    mhr_mapping: Sam3dBodyMhr,
    subjects_descr: list[str],
    cam_sees_subjects: dict[str, list[int]],
    video_start_frame_index: int,
    save_vertices: bool,
    write_body_joints_world: bool,
    triangulate: bool,
    triangulation_cameras: list[str],
    triangulation_min_detection_confidence: float,
) -> None:
    raw = np.load(raw_npz_path, allow_pickle=True)
    _validate_sam3d_npz(raw)

    faces = raw["faces"]
    per_frame = [f if isinstance(f, dict) else list(f) for f in raw["per_frame_outputs"]]
    mode = str(np.asarray(raw["mode"]).item()) if "mode" in raw.files else "unknown"
    _, bundle_notes = _bundle_camera_params(calibration, camera_names)

    if bbox_sort:
        if mode == "multi":
            for _fi, bundle in enumerate(per_frame):
                if not isinstance(bundle, dict):
                    continue
                for cam in camera_names:
                    if cam in bundle and isinstance(bundle[cam], list):
                        bundle[cam] = _sort_persons_by_bbox_x(bundle[cam])
        else:
            for fi in range(len(per_frame)):
                if isinstance(per_frame[fi], list):
                    per_frame[fi] = _sort_persons_by_bbox_x(per_frame[fi])

    if temporal_smooth:
        if mode == "multi":
            for cam in camera_names:
                cam_frames: list[list[dict[str, Any]]] = []
                for _fi, bundle in enumerate(per_frame):
                    if isinstance(bundle, dict):
                        cam_frames.append(bundle.get(cam, []))
                    else:
                        cam_frames.append([])
                smoothed_lists = _apply_temporal_smoothing_to_packed(cam_frames, smooth_window_length, smooth_polyorder)
                for fi, bundle in enumerate(per_frame):
                    if isinstance(bundle, dict):
                        bundle[cam] = smoothed_lists[fi]
        else:
            per_frame = _apply_temporal_smoothing_to_packed(per_frame, smooth_window_length, smooth_polyorder)

    cross_view_residuals: list[float] = []
    if world_align and calibration:
        if mode == "multi":
            for _fi, bundle in enumerate(per_frame):
                if not isinstance(bundle, dict):
                    continue
                for cam in camera_names:
                    if cam not in bundle or cam not in calibration:
                        continue
                    cal = calibration[cam]
                    if "intrinsic_matrix" not in cal or "projection_matrix" not in cal:
                        continue
                    K = np.asarray(cal["intrinsic_matrix"], dtype=np.float64)
                    P = np.asarray(cal["projection_matrix"], dtype=np.float64)
                    bundle[cam] = [_world_align_packed_frame(p, K, P) for p in bundle[cam]]
            if cross_view and len(camera_names) >= 2:
                c0, c1 = camera_names[0], camera_names[1]
                for _fi, bundle in enumerate(per_frame):
                    if not isinstance(bundle, dict):
                        continue
                    p0s, p1s = bundle.get(c0, []), bundle.get(c1, [])
                    for i in range(min(len(p0s), len(p1s))):
                        cross_view_residuals.append(_cross_view_mean_distance(p0s[i], p1s[i]))
        else:
            cam = camera_names[0]
            if cam in calibration:
                cal = calibration[cam]
                if "intrinsic_matrix" in cal and "projection_matrix" in cal:
                    K = np.asarray(cal["intrinsic_matrix"], dtype=np.float64)
                    P = np.asarray(cal["projection_matrix"], dtype=np.float64)
                    for fi in range(len(per_frame)):
                        per_frame[fi] = [_world_align_packed_frame(p, K, P) for p in per_frame[fi]]

    body_indices = (
        list(mhr_mapping.keypoints_index.body.values())
        + list(mhr_mapping.keypoints_index.foot.values())
        + list(mhr_mapping.keypoints_index.extra.values())
    )
    body_names = (
        list(mhr_mapping.keypoints_index.body.keys())
        + list(mhr_mapping.keypoints_index.foot.keys())
        + list(mhr_mapping.keypoints_index.extra.keys())
    )
    hand_indices = list(mhr_mapping.keypoints_index.hand.values())
    hand_names = list(mhr_mapping.keypoints_index.hand.keys())

    body_world = None
    if write_body_joints_world:
        body_world = build_body_joints_npz_payload(
            per_frame,
            joint_indices=body_indices,
            joint_names=body_names,
            camera_names=camera_names,
            mode=mode,
            subjects_descr=subjects_descr,
            cam_sees_subjects=cam_sees_subjects,
            video_start_frame_index=video_start_frame_index,
            three_d_primary="world",
        )
        stereo_applied = apply_stereo_triangulation_to_body_joints_payload(
            body_world,
            calibration=calibration,
            camera_names=camera_names,
            triangulation_cameras=triangulation_cameras,
            mode=mode,
            cam_sees_subjects=cam_sees_subjects,
            video_start_frame_index=video_start_frame_index,
            enabled=triangulate,
            min_confidence=triangulation_min_detection_confidence,
        )
        if stereo_applied:
            logging.info(
                "SAM 3D Body: body_joints 3d from two-view triangulation (%s).",
                " & ".join(triangulation_cameras),
            )
        policy_world = {
            "npz_role": "body_joints",
            "filename_stem": SAM3D_BODY_OUTPUT_NPZ_STEM,
            "alignment_method": (
                "two_view_opencv_triangulation" if stereo_applied else "calibration_projection_decomposition"
            ),
            "not_multi_view_triangulation": not stereo_applied,
        }
        if stereo_applied:
            policy_world["triangulation_cameras"] = list(triangulation_cameras)
        body_world = _attach_body_meta(
            body_world,
            mhr_mapping=mhr_mapping,
            bundle_notes=bundle_notes,
            cross_view=cross_view,
            cross_view_residuals=cross_view_residuals,
            raw=raw,
            export_policy=policy_world,
        )
        if out_body_joints_npz is None:
            raise ValueError("out_body_joints_npz required when write_body_joints_world is True")
        out_body_joints_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_body_joints_npz, **body_world)
        logging.info("SAM 3D Body wrote body_joints export %s", out_body_joints_npz)
    else:
        # No usable calibration: still write body_joints, but 2D only.
        # World 3D is impossible without calibration, and camera-space 3D is the
        # job of body_joints_local — so this file carries 2D keypoints only.
        body_2d_only = build_body_joints_npz_payload(
            per_frame,
            joint_indices=body_indices,
            joint_names=body_names,
            camera_names=camera_names,
            mode=mode,
            subjects_descr=subjects_descr,
            cam_sees_subjects=cam_sees_subjects,
            video_start_frame_index=video_start_frame_index,
            three_d_primary="camera",
        )
        body_2d_only = _strip_3d_from_body_payload(body_2d_only)
        policy_2d = {
            "npz_role": "body_joints",
            "filename_stem": SAM3D_BODY_OUTPUT_NPZ_STEM,
            "alignment_method": "none_no_calibration",
            "not_multi_view_triangulation": True,
            "three_d_primary": "none_2d_only",
            "note": "No usable calibration; 3D omitted (camera-space 3D is in body_joints_local).",
        }
        body_2d_only = _attach_body_meta(
            body_2d_only,
            mhr_mapping=mhr_mapping,
            bundle_notes=bundle_notes,
            cross_view=cross_view,
            cross_view_residuals=cross_view_residuals,
            raw=raw,
            export_policy=policy_2d,
        )
        if out_body_joints_npz is None:
            raise ValueError("out_body_joints_npz required to write body_joints")
        out_body_joints_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_body_joints_npz, **body_2d_only)
        logging.info("SAM 3D Body wrote 2D-only body_joints export (no calibration) %s", out_body_joints_npz)

    body_local = build_body_joints_npz_payload(
        per_frame,
        joint_indices=body_indices,
        joint_names=body_names,
        camera_names=camera_names,
        mode=mode,
        subjects_descr=subjects_descr,
        cam_sees_subjects=cam_sees_subjects,
        video_start_frame_index=video_start_frame_index,
        three_d_primary="camera",
    )
    policy_local = {
        "npz_role": "body_joints_local",
        "filename_stem": SAM3D_BODY_LOCAL_NPZ_STEM,
        "alignment_method": "calibration_projection_decomposition",
        "not_multi_view_triangulation": True,
        "three_d_primary": "camera_native_sam_output",
        "body_joints_world_npz_written": write_body_joints_world,
    }
    body_local = _attach_body_meta(
        body_local,
        mhr_mapping=mhr_mapping,
        bundle_notes=bundle_notes,
        cross_view=cross_view,
        cross_view_residuals=cross_view_residuals,
        raw=raw,
        export_policy=policy_local,
    )
    out_body_joints_local_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_body_joints_local_npz, **body_local)
    logging.info("SAM 3D Body wrote body_joints_local export %s", out_body_joints_local_npz)

    for component, indices, names, out_world, out_local, role_local in [
        ("hand_joints", hand_indices, hand_names, out_hand_joints_npz, out_hand_joints_local_npz, "hand_joints_local"),
    ]:
        if write_body_joints_world and out_world is not None:
            payload_world = build_body_joints_npz_payload(
                per_frame,
                joint_indices=indices,
                joint_names=names,
                camera_names=camera_names,
                mode=mode,
                subjects_descr=subjects_descr,
                cam_sees_subjects=cam_sees_subjects,
                video_start_frame_index=video_start_frame_index,
                three_d_primary="world",
            )
            payload_world = _attach_body_meta(
                payload_world,
                mhr_mapping=mhr_mapping,
                bundle_notes=bundle_notes,
                cross_view=cross_view,
                cross_view_residuals=cross_view_residuals,
                raw=raw,
                export_policy={"npz_role": component, "filename_stem": SAM3D_BODY_OUTPUT_NPZ_STEM},
            )
            out_world.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_world, **payload_world)
            logging.info("SAM 3D Body wrote %s export %s", component, out_world)
        payload_local = build_body_joints_npz_payload(
            per_frame,
            joint_indices=indices,
            joint_names=names,
            camera_names=camera_names,
            mode=mode,
            subjects_descr=subjects_descr,
            cam_sees_subjects=cam_sees_subjects,
            video_start_frame_index=video_start_frame_index,
            three_d_primary="camera",
        )
        payload_local = _attach_body_meta(
            payload_local,
            mhr_mapping=mhr_mapping,
            bundle_notes=bundle_notes,
            cross_view=cross_view,
            cross_view_residuals=cross_view_residuals,
            raw=raw,
            export_policy={"npz_role": role_local, "filename_stem": SAM3D_BODY_LOCAL_NPZ_STEM},
        )
        out_local.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_local, **payload_local)
        logging.info("SAM 3D Body wrote %s export %s", role_local, out_local)

    mesh_payload = build_body_mesh_npz_payload(
        per_frame,
        camera_names=camera_names,
        mode=mode,
        faces=faces,
        subjects_descr=subjects_descr,
        cam_sees_subjects=cam_sees_subjects,
        video_start_frame_index=video_start_frame_index,
        save_vertices=save_vertices,
    )
    mdd = mesh_payload["data_description"].item()
    m_sam = mdd.setdefault("sam_3d_body", {})
    m_sam.update(
        {
            "faces_shape": list(faces.shape),
            "camera_params_bundle": bundle_notes,
            "sam_3d_body_mhr_mapping": {
                "config_section": "human_pose.sam_3d_body_mhr",
                "coco_body_17_joint_names": list(mhr_mapping.keypoints_index.body.keys()),
                "coco_body_17_mhr70_index": list(mhr_mapping.keypoints_index.body.values()),
            },
            "export_policy": {
                "npz_role": "body_mesh",
                "filename_stem": SAM3D_BODY_OUTPUT_NPZ_STEM,
                "alignment_method": "calibration_projection_decomposition",
                "not_multi_view_triangulation": True,
                "body_joints_world_npz_written": write_body_joints_world,
            },
        }
    )
    if "data_description" in raw.files:
        try:
            prev = raw["data_description"].item()
            if isinstance(prev, dict):
                m_sam["inference_stage"] = prev
        except Exception:
            pass
    mesh_payload["data_description"] = np.asarray(mdd, dtype=object)

    out_body_mesh_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_body_mesh_npz, **mesh_payload)
    logging.info("SAM 3D Body wrote body_mesh export %s", out_body_mesh_npz)


def _per_frame_lists_from_body_mesh_npz(
    body_npz: Any,
    mesh_npz: Any | None,
    *,
    camera_names: list[str],
    cam_sees_subjects: dict[str, list[int]],
    mode: str,
) -> tuple[list[Any], np.ndarray]:
    """
    Rebuild per-frame person dict lists for visualization from dense body_joints /
    body_mesh exports.
    """
    two_d = body_npz["2d_interpolated"] if "2d_interpolated" in body_npz.files else body_npz["2d"]
    if "3d_camera" in body_npz.files:
        three_d = body_npz["3d_camera"]
    elif "3d" in body_npz.files:
        three_d = body_npz["3d"]
    else:
        three_d = None
    verts_all = None
    if mesh_npz is not None and "vertices" in mesh_npz.files:
        verts_all = np.asarray(mesh_npz["vertices"], dtype=np.float64)
        if verts_all.shape[-2] == 0:
            verts_all = None
    n_frames = int(two_d.shape[2])
    per_frame: list[Any] = []

    for fi in range(n_frames):
        if mode == "multi":
            bundle: dict[str, list[dict[str, Any]]] = {}
            for cam in camera_names:
                plist: list[dict[str, Any]] = []
                ci = camera_names.index(cam)
                for _pid, subj in enumerate(cam_sees_subjects.get(cam, [])):
                    kp2 = np.asarray(two_d[subj, ci, fi], dtype=np.float64)
                    d: dict[str, Any] = {"pred_keypoints_2d": kp2.astype(np.float32)}
                    if three_d is not None:
                        ci_3d = 0 if int(three_d.shape[1]) == 1 else ci
                        d["pred_keypoints_3d"] = np.asarray(three_d[subj, ci_3d, fi, :, :3], dtype=np.float32)
                    if verts_all is not None:
                        v = np.asarray(verts_all[subj, ci, fi], dtype=np.float64)
                        mask = np.any(np.isfinite(v), axis=1)
                        d["pred_vertices"] = v[mask].astype(np.float32)
                    plist.append(d)
                bundle[cam] = plist
            per_frame.append(bundle)
        else:
            cam = camera_names[0]
            ci = 0
            plist_single: list[dict[str, Any]] = []
            for _pid, subj in enumerate(cam_sees_subjects.get(cam, [])):
                kp2 = np.asarray(two_d[subj, ci, fi], dtype=np.float64)
                d2: dict[str, Any] = {"pred_keypoints_2d": kp2.astype(np.float32)}
                if three_d is not None:
                    ci_3d = 0 if int(three_d.shape[1]) == 1 else ci
                    d2["pred_keypoints_3d"] = np.asarray(three_d[subj, ci_3d, fi, :, :3], dtype=np.float32)
                if verts_all is not None:
                    v = np.asarray(verts_all[subj, ci, fi], dtype=np.float64)
                    mask = np.any(np.isfinite(v), axis=1)
                    d2["pred_vertices"] = v[mask].astype(np.float32)
                plist_single.append(d2)
            per_frame.append(plist_single)

    if mesh_npz is not None and "faces" in mesh_npz.files:
        faces = np.asarray(mesh_npz["faces"], dtype=np.int64)
    else:
        faces = np.zeros((0, 3), dtype=np.int64)
    return per_frame, faces


def _draw_mhr_on_image(
    image: np.ndarray,
    kp_2d: np.ndarray,
    edges: list[tuple[int, int]],
    joint_color: tuple[int, int, int],
    skel_color: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    n = kp_2d.shape[0]
    for i, j in edges:
        if i >= n or j >= n:
            continue
        p0 = kp_2d[i, :2]
        p1 = kp_2d[j, :2]
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
            continue
        cv2.line(
            image,
            tuple(int(x) for x in p0),
            tuple(int(x) for x in p1),
            color=skel_color,
            thickness=thickness,
        )
    for idx in range(n):
        pt = kp_2d[idx, :2]
        if np.any(np.isnan(pt)):
            continue
        cv2.circle(image, tuple(int(x) for x in pt), radius=radius, color=joint_color, thickness=-1)


def _draw_component_overlay(
    image: np.ndarray,
    kp_2d: np.ndarray,
    chains: list[list[str]],
    name_to_pos: dict[str, int],
    joint_color: tuple[int, int, int],
    skel_color: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    for chain in chains:
        for a, b in zip(chain[:-1], chain[1:]):
            pa, pb = name_to_pos.get(a), name_to_pos.get(b)
            if pa is None or pb is None or pa >= kp_2d.shape[0] or pb >= kp_2d.shape[0]:
                continue
            p0, p1 = kp_2d[pa, :2], kp_2d[pb, :2]
            if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
                continue
            cv2.line(image, tuple(int(x) for x in p0), tuple(int(x) for x in p1), color=skel_color, thickness=thickness)
    for i in range(kp_2d.shape[0]):
        pt = kp_2d[i, :2]
        if np.any(np.isnan(pt)):
            continue
        cv2.circle(image, tuple(int(x) for x in pt), radius=radius, color=joint_color, thickness=-1)


class Sam3dBody(BaseMethod):
    """SAM 3D Body (Hugging Face). GPU inference in sam_3d_body venv; post-process in main env."""

    inference_package_name = "sam_3d_body"
    components = [
        "body_joints",
        "body_joints_local",
        "hand_joints",
        "hand_joints_local",
        "body_mesh",
    ]
    algorithm_type = "sam_3d_body"
    inference_config = Sam3dBodyConfig

    def _initialize_detector(self) -> MethodDetectorRuntime:
        """
        Initialize the Sam3d method detector.
        """
        self.viz_folder = self.viz_folders["body_joints"]
        self.out_folder = self.out_folders["body_joints_local"]

        return super()._initialize_detector()

    def post_inference(self) -> None:
        raw_path = (
            Path(self.io.get_detector_output_folder("body_joints_local", self.algorithm_instance, "output"))
            / RAW_INFERENCE_NPZ_NAME
        )
        if not raw_path.is_file():
            logging.error("SAM 3D Body raw inference result missing: %s", raw_path)
            return

        cfg = self.detector_config
        calib_ok = _calibration_usable_for_world_alignment(self.data.calibration, list(cfg.camera_names))

        out_body = Path(self.result_folders["body_joints"]) / f"{SAM3D_BODY_OUTPUT_NPZ_STEM}.npz"
        out_body_local = Path(self.result_folders["body_joints_local"]) / f"{SAM3D_BODY_LOCAL_NPZ_STEM}.npz"
        out_hand = Path(self.result_folders["hand_joints"]) / f"{SAM3D_BODY_OUTPUT_NPZ_STEM}.npz" if calib_ok else None
        out_hand_local = Path(self.result_folders["hand_joints_local"]) / f"{SAM3D_BODY_LOCAL_NPZ_STEM}.npz"
        out_mesh = Path(self.result_folders["body_mesh"]) / f"{SAM3D_BODY_OUTPUT_NPZ_STEM}.npz"

        _run_sam3d_post_process(
            raw_path,
            out_body_joints_npz=out_body,
            out_body_joints_local_npz=out_body_local,
            out_hand_joints_npz=out_hand,
            out_hand_joints_local_npz=out_hand_local,
            out_body_mesh_npz=out_mesh,
            calibration=self.data.calibration,
            camera_names=list(cfg.camera_names),
            temporal_smooth=cfg.temporal_smooth,
            smooth_window_length=cfg.smooth_window_length,
            smooth_polyorder=cfg.smooth_polyorder,
            world_align=cfg.world_align_keypoints_3d,
            cross_view=cfg.cross_view_consistency,
            bbox_sort=cfg.bbox_sort_left_to_right,
            mhr_mapping=self.predictions_mapping.human_pose.sam_3d_body_mhr,
            subjects_descr=list(self.data.subjects_descr),
            cam_sees_subjects=dict(self.data.cam_sees_subjects),
            video_start_frame_index=int(self.data.video_start_frame_index),
            save_vertices=cfg.save_vertices,
            write_body_joints_world=calib_ok,
            triangulate=cfg.triangulation.triangulate,
            triangulation_cameras=cfg.triangulation.triangulation_cameras,
            triangulation_min_detection_confidence=cfg.triangulation_min_detection_confidence,
        )

    def visualization(self, _data) -> None:
        logging.info(
            "VISUALIZING the method detector output of %s and %s.",
            self.components,
            self.algorithm_instance,
        )
        cfg = self.detector_config
        fps = int(self.data.fps)
        video_start = int(self.data.video_start_frame_index)
        dataloader = ImagePathsByCameraLoader(
            config=self.data.get_input_recipes(), expected_cameras=list(cfg.camera_names)
        )

        RADIUS = 4
        THICKNESS = 2
        JOINT_COLOR = (187, 197, 254)
        SKELETON_COLOR = (255, 144, 30)
        person_colors = [
            (JOINT_COLOR, SKELETON_COLOR),
            ((200, 230, 200), (100, 180, 100)),
            ((230, 200, 200), (180, 100, 100)),
            ((220, 220, 150), (150, 150, 80)),
        ]
        mhr = self.predictions_mapping.human_pose.sam_3d_body_mhr
        ki = mhr.keypoints_index
        body_joint_names = list(ki.body.keys()) + list(ki.foot.keys()) + list(ki.extra.keys())
        name_to_pos = {name: pos for pos, name in enumerate(body_joint_names)}
        vis_edges: list[tuple[int, int]] = []
        for chain in mhr.connections.body_joints:
            for a, b in zip(chain[:-1], chain[1:]):
                pa, pb = name_to_pos.get(a), name_to_pos.get(b)
                if pa is not None and pb is not None:
                    vis_edges.append((pa, pb))
        mesh_colors_bgr = [
            (80, 180, 80),
            (60, 140, 200),
            (180, 100, 100),
            (200, 180, 60),
        ]

        if not self.viz_folder:
            logging.warning("SAM 3D Body visualization: viz_folder unset (visualize off?).")
            return
        viz_dir = str(Path(self.viz_folder))
        os.makedirs(viz_dir, exist_ok=True)
        viz_3d_dir: str | None = None
        if cfg.visualize_mesh:
            viz_3d_dir = str(Path(self.result_folders["body_mesh"]) / "visualization_3d")
            os.makedirs(viz_3d_dir, exist_ok=True)

        viz_dir_local = str(
            Path(self.io.get_detector_output_folder("body_joints_local", self.algorithm_instance, "visualization"))
        )
        os.makedirs(viz_dir_local, exist_ok=True)

        def _try_world_body_npz() -> Path | None:
            p = Path(self.result_folders["body_joints"]) / f"{SAM3D_BODY_OUTPUT_NPZ_STEM}.npz"
            return p if p.is_file() else None

        def _try_local_body_npz() -> Path | None:
            for stem in (SAM3D_BODY_LOCAL_NPZ_STEM, SAM3D_BODY_OUTPUT_NPZ_STEM):
                p = Path(self.result_folders["body_joints_local"]) / f"{stem}.npz"
                if p.is_file():
                    return p
            return None

        world_body_path = _try_world_body_npz()
        local_body_path = _try_local_body_npz()
        if local_body_path is None:
            logging.warning(
                "SAM 3D Body visualization: missing camera-primary body NPZ (expected %s under body_joints_local).",
                SAM3D_BODY_LOCAL_NPZ_STEM,
            )
            return

        mesh_pred = Path(self.result_folders["body_mesh"]) / f"{SAM3D_BODY_OUTPUT_NPZ_STEM}.npz"

        local_body_npz = np.load(local_body_path, allow_pickle=True)
        world_body_npz = np.load(world_body_path, allow_pickle=True) if world_body_path is not None else local_body_npz
        mesh_npz = np.load(mesh_pred, allow_pickle=True) if mesh_pred.is_file() else None
        mode = "multi" if len(cfg.camera_names) > 1 else "single"
        per_frame_world, faces = _per_frame_lists_from_body_mesh_npz(
            world_body_npz,
            mesh_npz,
            camera_names=list(cfg.camera_names),
            cam_sees_subjects=dict(self.data.cam_sees_subjects),
            mode=mode,
        )
        per_frame_local, faces_local = _per_frame_lists_from_body_mesh_npz(
            local_body_npz,
            mesh_npz,
            camera_names=list(cfg.camera_names),
            cam_sees_subjects=dict(self.data.cam_sees_subjects),
            mode=mode,
        )
        viz_targets: list[tuple[str, str, str | None, list[Any], np.ndarray]] = [
            ("body_joints_world_primary_npz", viz_dir, viz_3d_dir, per_frame_world, faces),
            ("body_joints_local_camera_npz", viz_dir_local, None, per_frame_local, faces_local),
        ]

        calibration = self.data.calibration or {}

        mesh_logged_missing = False
        success = True
        for camera_name, frames_list in dataloader:
            if camera_name not in cfg.camera_names:
                continue
            cam_idx = cfg.camera_names.index(camera_name)
            cal_cam = calibration.get(camera_name) if isinstance(calibration, dict) else None

            for _tag, vroot, v3root, per_frame, face_arr in viz_targets:
                cam_out = os.path.join(vroot, camera_name)
                os.makedirs(cam_out, exist_ok=True)
                cam_out_3d = os.path.join(v3root, camera_name) if v3root is not None else None
                if cam_out_3d is not None:
                    os.makedirs(cam_out_3d, exist_ok=True)

                for frame_idx, image_file in enumerate(frames_list):
                    if frame_idx >= len(per_frame):
                        break
                    image = cv2.imread(image_file)
                    if image is None:
                        logging.warning("Could not read image %s", image_file)
                        continue
                    plist = frame_person_list(per_frame[frame_idx], camera_name, cam_idx, mode)

                    viz_2d = image.copy()
                    for pid, person in enumerate(plist):
                        kp = np.asarray(person.get("pred_keypoints_2d"), dtype=np.float64)
                        if kp.size == 0:
                            continue
                        jc, sc = person_colors[pid % len(person_colors)]
                        _draw_mhr_on_image(viz_2d, kp, vis_edges, jc, sc, RADIUS, THICKNESS)

                    cv2.imwrite(
                        os.path.join(cam_out, f"{frame_idx + video_start:09d}.jpg"),
                        viz_2d,
                    )

                    if cam_out_3d is not None:
                        viz_3d = image.copy()
                        h, w = image.shape[:2]
                        K, dist = mesh_viz._intrinsics_from_calibration_or_image(cal_cam, w, h)
                        for pid, person in enumerate(plist):
                            verts = person.get("pred_vertices")
                            if verts is None:
                                if not mesh_logged_missing:
                                    logging.warning(
                                        "SAM 3D Body visualization_3d: no pred_vertices in npz. "
                                        "Set save_vertices = true in [algorithms.sam_3d_body] and re-run inference."
                                    )
                                    mesh_logged_missing = True
                                continue
                            kp3 = np.asarray(person.get("pred_keypoints_3d"), dtype=np.float64)
                            kp2 = np.asarray(person.get("pred_keypoints_2d"), dtype=np.float64)
                            if kp3.size == 0 or kp2.size == 0:
                                continue
                            mc = mesh_colors_bgr[pid % len(mesh_colors_bgr)]
                            mesh_viz.draw_mesh_wireframe_on_image(
                                viz_3d,
                                verts,
                                face_arr,
                                kp3,
                                kp2,
                                K,
                                dist,
                                mc,
                                line_thickness=1,
                            )
                        cv2.imwrite(
                            os.path.join(cam_out_3d, f"{frame_idx + video_start:09d}.jpg"),
                            viz_3d,
                        )

                success *= vd.frames_to_video(
                    cam_out,
                    os.path.join(vroot, f"{camera_name}.mp4"),
                    fps,
                    start_frame=video_start,
                )
                if cam_out_3d is not None:
                    success *= vd.frames_to_video(
                        cam_out_3d,
                        os.path.join(v3root, f"{camera_name}.mp4"),
                        fps,
                        start_frame=video_start,
                    )

        mhr = self.predictions_mapping.human_pose.sam_3d_body_mhr
        for comp, npz_stem, chains in [
            ("hand_joints_local", SAM3D_BODY_LOCAL_NPZ_STEM, mhr.connections.hand_joints),
        ]:
            comp_path = Path(self.result_folders[comp]) / f"{npz_stem}.npz"
            if not comp_path.is_file():
                continue
            comp_npz = np.load(comp_path, allow_pickle=True)
            kp_data = comp_npz["2d_interpolated"] if "2d_interpolated" in comp_npz.files else comp_npz["2d"]
            labels: list[str] = list(comp_npz["data_description"].item()["2d_interpolated"]["axis3"])
            name_to_pos = {name: pos for pos, name in enumerate(labels)}
            n_sub = kp_data.shape[0]
            viz_comp_dir = self.viz_folders[comp]
            os.makedirs(viz_comp_dir, exist_ok=True)

            for camera_name, frames_list in dataloader:
                if camera_name not in cfg.camera_names:
                    continue
                cam_idx = cfg.camera_names.index(camera_name)
                cam_out = os.path.join(viz_comp_dir, camera_name)
                os.makedirs(cam_out, exist_ok=True)
                for frame_idx, image_file in enumerate(frames_list):
                    if frame_idx >= kp_data.shape[2]:
                        break
                    image = cv2.imread(image_file)
                    if image is None:
                        continue
                    viz = image.copy()
                    for subj in range(n_sub):
                        jc, sc = person_colors[subj % len(person_colors)]
                        kp = kp_data[subj, cam_idx, frame_idx].astype(np.float64)
                        _draw_component_overlay(viz, kp, chains, name_to_pos, jc, sc, radius=1, thickness=1)
                    cv2.imwrite(os.path.join(cam_out, f"{frame_idx + video_start:09d}.jpg"), viz)
                vd.frames_to_video(
                    cam_out, os.path.join(viz_comp_dir, f"{camera_name}.mp4"), fps, start_frame=video_start
                )

        if cfg.visualize_mesh_interactive and mesh_pred.is_file():
            out_inter = Path(self.result_folders["body_mesh"]) / "visualization_3d_interactive"
            script = Path(__file__).resolve().parent / "sam_3d_body_interactive_mesh_html.py"
            cmd = [
                sys.executable,
                str(script),
                "--npz",
                str(mesh_pred),
                "--out-dir",
                str(out_inter),
                "--fps",
                str(fps),
                "--camera-index",
                str(int(cfg.interactive_mesh_camera_index)),
                "--stride",
                str(max(1, int(cfg.interactive_mesh_frame_stride))),
            ]
            if not cfg.interactive_mesh_prefer_world:
                cmd.append("--no-prefer-world")
            if cfg.interactive_mesh_subject_spacing is not None and float(cfg.interactive_mesh_subject_spacing) > 0:
                cmd.extend(["--subject-spacing", str(float(cfg.interactive_mesh_subject_spacing))])
            cmd.extend(
                [
                    "--line-width",
                    str(max(1, min(6, int(cfg.interactive_mesh_line_width)))),
                    "--max-edges",
                    str(max(500, int(cfg.interactive_mesh_max_edges))),
                    "--max-frames",
                    str(max(2, int(cfg.interactive_mesh_max_frames))),
                ]
            )
            if not cfg.interactive_mesh_plotlyjs_cdn:
                cmd.append("--embed-plotlyjs")
            try:
                r = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=7200,
                )
                if r.returncode != 0:
                    tail = (r.stderr or r.stdout or "").strip()
                    logging.warning(
                        "SAM 3D Body interactive mesh HTML script failed (exit %s). %s",
                        r.returncode,
                        tail[-800:] if tail else "",
                    )
                else:
                    logging.info(
                        "SAM 3D Body wrote interactive mesh HTML under %s",
                        out_inter,
                    )
            except subprocess.TimeoutExpired:
                logging.warning("SAM 3D Body interactive mesh HTML script timed out.")
            except Exception as e:
                logging.warning("SAM 3D Body interactive mesh HTML: %s", e)

        logging.info("Detector %s: visualization finished with code %s.", self.components, success)
