"""
Run SAM 3D Body inference (facebookresearch/sam-3d-body + Hugging Face weights).
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

if os.name == "nt":
    # upstream notebook/utils.py prints Unicode (✓) which cp1252 can't encode
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    # renderer.py sets PYOPENGL_PLATFORM="egl" at import time if not already set;
    # EGL is unavailable on Windows, pyglet uses standard WGL
    os.environ.setdefault("PYOPENGL_PLATFORM", "pyglet")

import cv2
import numpy as np
import torch

from nicetoolbox_core.entrypoint import run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByCameraLoader

# Launched as ``python …/sam_3d_body_inference.py`` (no package context). Load sibling
# ``sam_3d_body_paths.py`` by path so we do not import ``nicetoolbox…sam_3d_body`` (that
# would run ``__init__.py`` and pull the full detector stack into the subprocess).
_paths_file = Path(__file__).resolve().parent / "sam_3d_body_paths.py"
_spec = importlib.util.spec_from_file_location("_sam_3d_body_paths", _paths_file)
_sam3d_paths = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_sam3d_paths)
RAW_INFERENCE_NPZ_NAME = _sam3d_paths.RAW_INFERENCE_NPZ_NAME
ensure_hf_hub_cache_env = _sam3d_paths.ensure_hf_hub_cache_env
ensure_sam3d_repo = _sam3d_paths.ensure_sam3d_repo


def _bundle_calib_for_npz(calibration, camera_names):
    if not calibration:
        return {}
    out = {}
    for cam in camera_names:
        if cam not in calibration:
            continue
        c = calibration[cam]
        entry = {}
        for key in ("intrinsic_matrix", "distortions", "projection_matrix", "image_size"):
            if key in c and c[key] is not None:
                entry[key] = np.asarray(c[key], dtype=np.float64)
        if entry:
            out[cam] = entry
    return out


def tensor_K(K: np.ndarray, device: str):
    """Build camera intrinsics tensor for SAM 3D Body (must be B x 3 x 3, not 3 x 3)."""
    t = torch.as_tensor(K, dtype=torch.float32, device=device)
    # prepare_batch default cam_int is shape (1, 3, 3); get_ray_condition indexes batch dim 0.
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t


def pack_person_output(person: dict, save_vertices: bool) -> dict:
    d = {
        "bbox": np.asarray(person["bbox"], np.float32),
        "focal_length": np.float32(person["focal_length"]),
        "pred_keypoints_2d": np.asarray(person["pred_keypoints_2d"], np.float32),
        "pred_keypoints_3d": np.asarray(person["pred_keypoints_3d"], np.float32),
        "pred_cam_t": np.asarray(person["pred_cam_t"], np.float32),
        "global_rot": np.asarray(person["global_rot"], np.float32),
        "body_pose_params": np.asarray(person["body_pose_params"], np.float32),
        "hand_pose_params": np.asarray(person["hand_pose_params"], np.float32),
        "shape_params": np.asarray(person["shape_params"], np.float32),
        "scale_params": np.asarray(person["scale_params"], np.float32),
        "expr_params": np.asarray(person["expr_params"], np.float32),
        "pred_pose_raw": np.asarray(person["pred_pose_raw"], np.float32),
        "mhr_model_params": np.asarray(person["mhr_model_params"], np.float32),
    }
    if save_vertices:
        d["pred_vertices"] = np.asarray(person["pred_vertices"], np.float32)
    return d


@run_inference_entrypoint
def main(config: dict) -> None:
    logging.info("SAM 3D Body inference starting (algorithm=%s)", config.get("algorithm"))

    # TODO: unify loading non-pip installable submodules
    nicetoolbox_root = Path(config["nicetoolbox_root"])

    ensure_hf_hub_cache_env(nicetoolbox_root)
    repo_cfg = (config.get("sam3d_repo_path") or os.environ.get("SAM3D_BODY_REPO", "")).strip()
    repo_root = ensure_sam3d_repo(repo_cfg, nicetoolbox_root)

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from notebook.utils import setup_sam_3d_body

    camera_names = config["camera_names"]
    device = config["device"]
    estimator = setup_sam_3d_body(
        hf_repo_id=config.get("hf_repo_id", "facebook/sam-3d-body-dinov3"),
        detector_name=config.get("detector_name", "vitdet"),
        segmentor_name=config.get("segmentor_name", "sam2"),
        fov_name=config.get("fov_name", "moge2"),
        device=device,
    )

    calibration = config.get("calibration") or {}
    cam_int_map = {}
    for cam in camera_names:
        if cam in calibration and "intrinsic_matrix" in calibration[cam]:
            K = np.asarray(calibration[cam]["intrinsic_matrix"], dtype=np.float64)
            if K.shape == (3, 3):
                cam_int_map[cam] = tensor_K(K, device)

    dataloader = ImagePathsByCameraLoader(config=config, expected_cameras=camera_names)
    paths_by_cam: dict[str, list] = {}
    for cam, plist in dataloader:
        paths_by_cam[cam] = plist

    lengths = [len(paths_by_cam[c]) for c in camera_names if c in paths_by_cam]
    if not lengths:
        raise RuntimeError("No image paths found for SAM 3D Body.")
    n_frames = min(lengths)
    if min(lengths) != max(lengths):
        logging.warning(
            "Camera frame counts differ (min=%s max=%s); using first %s frames.",
            min(lengths),
            max(lengths),
            n_frames,
        )

    mode = "multi" if len(camera_names) > 1 else "single"
    faces_once = None
    frames_meta = []
    per_frame: list = []

    inference_type = config.get("inference_type", "body")
    save_vertices = bool(config.get("save_vertices", False))

    for fi in range(n_frames):
        if mode == "multi":
            frame_bundle: dict = {}
            for cam in camera_names:
                path = paths_by_cam[cam][fi]
                bgr = cv2.imread(path)
                if bgr is None:
                    raise RuntimeError(f"Failed to read image: {path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cam_int = cam_int_map.get(cam)
                out = estimator.process_one_image(rgb, cam_int=cam_int, inference_type=inference_type)
                frame_bundle[cam] = [pack_person_output(p, save_vertices) for p in out]
                if faces_once is None:
                    faces_once = np.asarray(estimator.faces, dtype=np.int32)
            per_frame.append(frame_bundle)
            frames_meta.append({"frame_index": fi, "cameras": list(camera_names)})
        else:
            cam = camera_names[0]
            path = paths_by_cam[cam][fi]
            bgr = cv2.imread(path)
            if bgr is None:
                raise RuntimeError(f"Failed to read image: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cam_int = cam_int_map.get(cam)
            out = estimator.process_one_image(rgb, cam_int=cam_int, inference_type=inference_type)
            per_frame.append([pack_person_output(p, save_vertices) for p in out])
            frames_meta.append({"frame_index": fi, "num_persons": len(out)})
            if faces_once is None:
                faces_once = np.asarray(estimator.faces, dtype=np.int32)

    assert faces_once is not None
    camera_bundle = _bundle_calib_for_npz(calibration, camera_names)
    out_path = os.path.join(config["out_folders"]["body_joints_local"], RAW_INFERENCE_NPZ_NAME)

    # Object-array pickle is fine when sam_3d_body and main nicetoolbox both use NumPy 1.x
    # (see install script / sam_3d_body_pip_requirements numpy pin).
    np.savez_compressed(
        out_path,
        faces=faces_once,
        frames_meta=np.asarray(frames_meta, dtype=object),
        per_frame_outputs=np.asarray(per_frame, dtype=object),
        mode=np.array(str(mode)),
        camera_names_order=np.asarray([str(c) for c in camera_names], dtype=object),
        hf_repo=np.array(str(config.get("hf_repo_id", ""))),
        camera_params_bundle=np.asarray(camera_bundle, dtype=object),
        data_description=np.asarray(
            {
                "note": "Raw SAM 3D Body pack; run Sam3dBody.post_inference for alignment/smoothing.",
                "mhr_params": "Native outputs are MHR (Momentum Human Rig), not vanilla SMPL.",
            },
            dtype=object,
        ),
    )
    logging.info("SAM 3D Body wrote %s", out_path)
