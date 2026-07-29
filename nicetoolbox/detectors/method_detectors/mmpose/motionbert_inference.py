"""
3D body pose lifting from an existing 2D body_joints NPZ (e.g. MotionBERT).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import numpy as np
from mmengine.structures import InstanceData
from mmpose.apis import convert_keypoint_definition, extract_pose_sequence, inference_pose_lifter_model, init_model
from mmpose.apis.inferencers.pose3d_inferencer import Pose3DInferencer
from mmpose.structures import PoseDataSample, merge_data_samples, split_instances
from mmpose_pose3d_visualizer_patch import apply_pose3d_visualizer_rgb_reshape_fix

from nicetoolbox_core.entrypoint import run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByCameraLoader


def _mmpose_visualization_dir(config: dict[str, Any], camera_name: str) -> str:
    """
    Returns the visualization directory for a given camera.
    """
    result_folders = config["result_folders"]
    if not result_folders:
        pred_dir = config["prediction_folders"][camera_name]
        out_root = os.path.normpath(os.path.join(pred_dir, "..", ".."))
        return os.path.join(out_root, "visualization", camera_name)
    comp = next(iter(result_folders.keys()))
    base = result_folders[comp]
    return os.path.join(base, config["algorithm"], "visualization", camera_name)


def _validate_subject_aligned_frame_batch(data, num_persons: int):
    """NPZ lift path: one prediction per NICE subject slot — no bbox resorting."""
    out = []
    for i, frame in enumerate(data):
        preds = frame["predictions"][0]
        if len(preds) != num_persons:
            raise ValueError(
                f"Frame {i}: expected {num_persons} subject-aligned predictions, got {len(preds)} "
                "(check cam_sees_subjects vs upstream 2D NPZ)."
            )
        out.append(preds)
    return out


def convert_output_to_numpy(
    data,
    camera_subjects,
    number_subjects,
    fallback_num_keypoints,
    is_3d_model,
):
    num_frames = len(data)
    num_persons = len(camera_subjects)
    num_keypoints = fallback_num_keypoints
    num_estimations = 3
    logging.info(f"frames: {num_frames}, keypoints: {num_keypoints}, estimations: {num_estimations}")

    # Subject order and de-duplication come from upstream 2D (NPZ); one list entry per NICE subject slot.
    sorted_frame_predictions = _validate_subject_aligned_frame_batch(data, num_persons)

    keypoints_array = np.full((number_subjects, num_frames, num_keypoints, num_estimations), np.nan)
    bbox_array = np.full((number_subjects, num_frames, 1, 5), np.nan)

    keypoints_3d_array = None
    if is_3d_model:
        keypoints_3d_array = np.full((number_subjects, num_frames, num_keypoints, 4), np.nan)

    for frame_index, frame in enumerate(sorted_frame_predictions):
        for detection_index, person in enumerate(frame):
            person_index = camera_subjects[detection_index]

            for kp_index, (kp, score) in enumerate(zip(person["keypoints"], person["keypoint_scores"])):
                keypoints_array[person_index, frame_index, kp_index] = [kp[0], kp[1], score]

            if is_3d_model and keypoints_3d_array is not None:
                k3 = person.get("keypoints_3d")
                if k3 is not None:
                    # MMPose often omits ``keypoint_3d_scores``; reuse 2D instance scores (same length).
                    scores_3d = (
                        person["keypoint_3d_scores"]
                        if person.get("keypoint_3d_scores") is not None
                        else person["keypoint_scores"]
                    )
                    for kp_index, (kp_3d, score) in enumerate(zip(k3, scores_3d)):
                        if kp_index >= num_keypoints:
                            break
                        keypoints_3d_array[person_index, frame_index, kp_index] = [
                            kp_3d[0],
                            kp_3d[1],
                            kp_3d[2],
                            score,
                        ]
                else:
                    raw_kpts = person.get("keypoints")
                    raw_scores = person.get("keypoint_scores")
                    if raw_kpts and len(raw_kpts) > 0:
                        probe = np.asarray(raw_kpts[0], dtype=np.float64).ravel()
                        if probe.size >= 3:
                            for kp_index, kp in enumerate(raw_kpts):
                                if kp_index >= num_keypoints:
                                    break
                                kp_arr = np.asarray(kp, dtype=np.float64).ravel()
                                if kp_arr.size < 3:
                                    continue
                                sc = raw_scores[kp_index] if kp_index < len(raw_scores) else 1.0
                                keypoints_3d_array[person_index, frame_index, kp_index] = [
                                    float(kp_arr[0]),
                                    float(kp_arr[1]),
                                    float(kp_arr[2]),
                                    float(sc),
                                ]

            bbox = person.get("bbox", person.get("track_bbox", [[0, 0, 0, 0]]))[0]
            bbox_score = person.get("bbox_score", 1.0)
            bbox_array[person_index, frame_index, 0] = [
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                bbox_score,
            ]

    data_description = {
        "2d": ["coordinate_x", "coordinate_y", "confidence_score"],
        "bbox_2d": [
            "top_left_x",
            "top_left_y",
            "bottom_right_x",
            "bottom_right_y",
            "confidence_score",
        ],
    }
    if is_3d_model:
        data_description["3d"] = ["coordinate_x", "coordinate_y", "coordinate_z", "confidence_score"]

    return keypoints_array, keypoints_3d_array, bbox_array, data_description


def _read_image_size_from_dataloader(config: dict[str, Any]) -> tuple[int, int]:
    dataloader = ImagePathsByCameraLoader(config=config, expected_cameras=config["camera_names"])
    import cv2

    for _cam, paths in dataloader:
        for p in paths:
            im = cv2.imread(p)
            if im is not None:
                h, w = im.shape[:2]
                return h, w
    raise RuntimeError(
        "Could not read any frame from image_folders / input_recipes to determine (height, width) for the lifter."
    )


def _postprocess_pose_lift_results(
    pose_lift_results: list[PoseDataSample],
    pose_est_results: list[PoseDataSample],
    disable_rebase_keypoint: bool,
) -> None:
    for idx, pose_lift_res in enumerate(pose_lift_results):
        pose_lift_res.track_id = pose_est_results[idx].get("track_id")

        pred_instances = pose_lift_res.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        if keypoint_scores.ndim == 3:
            pose_lift_res.pred_instances.keypoint_scores = np.squeeze(keypoint_scores, axis=1)
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        keypoints = keypoints[..., [0, 2, 1]]
        keypoints[..., 0] = -keypoints[..., 0]
        keypoints[..., 2] = -keypoints[..., 2]

        if not disable_rebase_keypoint:
            keypoints[..., 2] -= np.min(keypoints[..., 2], axis=-1, keepdims=True)

        pose_lift_res.pred_instances.keypoints = keypoints

    pose_lift_results.sort(key=lambda x: x.get("track_id", 1e4))


def _one_person_sample(
    kpts_coco: np.ndarray,
    scores: np.ndarray,
    bbox_xyxy: np.ndarray,
    bbox_score: float,
    track_id: int,
    pose_det_dataset: str,
    pose_lift_dataset: str,
) -> PoseDataSample:
    k = kpts_coco.astype(np.float32)
    s = scores.astype(np.float32)
    if k.shape[0] != 17:
        raise ValueError(f"Expected 17 COCO body keypoints, got shape {k.shape}")

    k_xy = k[:, :2]
    k_h36m = convert_keypoint_definition(k_xy[None, ...], pose_det_dataset, pose_lift_dataset)[0]

    ds = PoseDataSample()
    pred = InstanceData()
    xy = k_h36m[:, :2].astype(np.float32)
    pred.set_field(xy[np.newaxis, ...], "keypoints")
    pred.set_field(s[np.newaxis, ...], "keypoint_scores")
    pred.set_field(bbox_xyxy.astype(np.float32)[np.newaxis, ...], "bboxes")
    pred.set_field(np.array([float(bbox_score)], dtype=np.float32), "bbox_scores")
    ds.pred_instances = pred
    ds.gt_instances = InstanceData()
    ds.set_field(track_id, "track_id")
    return ds


def _build_frame_persons_from_npz(
    k_rows: np.ndarray,
    bbox_rows: np.ndarray,
    pose_det_dataset: str,
    pose_lift_dataset: str,
    coco_indices: Sequence[int],
) -> list[PoseDataSample]:
    frame_samples: list[PoseDataSample] = []
    for det_i in range(k_rows.shape[0]):
        raw = k_rows[det_i].astype(np.float32)
        sl = raw[list(coco_indices)]
        xy = sl[:, :2]
        conf = sl[:, 2]
        bb = bbox_rows[det_i, 0, :4]
        bb_score = float(bbox_rows[det_i, 0, 4])
        if not np.isfinite(xy).all():
            xy = np.where(np.isfinite(xy), xy, 0.0)
        if not np.isfinite(bb).all():
            bb = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        frame_samples.append(_one_person_sample(xy, conf, bb, bb_score, det_i, pose_det_dataset, pose_lift_dataset))
    return frame_samples


def _merged_coco_2d_for_vis(
    k_rows: np.ndarray,
    b_rows: np.ndarray,
    coco_indices: Sequence[int],
    img_path: str,
    ori_h: int,
    ori_w: int,
) -> PoseDataSample:
    parts: list[PoseDataSample] = []
    for i in range(k_rows.shape[0]):
        ds = PoseDataSample()
        pred = InstanceData()
        sl = k_rows[i][list(coco_indices)]
        xy = sl[:, :2].astype(np.float32)
        sc = sl[:, 2].astype(np.float32)
        bb = b_rows[i, 0, :4].astype(np.float32)
        bs = float(b_rows[i, 0, 4])
        pred.set_field(xy[np.newaxis, ...], "keypoints")
        pred.set_field(sc[np.newaxis, ...], "keypoint_scores")
        pred.set_field(bb[np.newaxis, ...], "bboxes")
        pred.set_field(np.array([bs], dtype=np.float32), "bbox_scores")
        ds.pred_instances = pred
        ds.gt_instances = InstanceData()
        ds.set_field(i, "track_id")
        ds.metainfo["img_path"] = img_path
        ds.metainfo["ori_shape"] = (ori_h, ori_w)
        parts.append(ds)
    return merge_data_samples(parts)


def _maybe_visualize_frame(
    pose3d_vis: Pose3DInferencer,
    disk_img_path: str,
    global_frame_index: int,
    pose_lift_results: list[PoseDataSample],
    k_rows: np.ndarray,
    b_rows: np.ndarray,
    coco_indices: Sequence[int],
    vis_out_dir: str,
    viz_cfg: dict[str, Any],
) -> None:
    import mmcv

    img_rgb = mmcv.imread(disk_img_path, channel_order="rgb")
    if img_rgb is None:
        logging.warning("3D vis: skip missing image %s", disk_img_path)
        return
    ori_h, ori_w = img_rgb.shape[:2]
    det_merged = _merged_coco_2d_for_vis(k_rows, b_rows, coco_indices, disk_img_path, ori_h, ori_w)

    merged_3d = merge_data_samples(pose_lift_results)
    merged_3d.metainfo["ori_shape"] = (ori_h, ori_w)
    merged_3d.metainfo["img_path"] = disk_img_path

    out_path = os.path.join(vis_out_dir, f"{int(global_frame_index):09d}.png")
    os.makedirs(vis_out_dir, exist_ok=True)

    vis = pose3d_vis.visualizer
    vis.radius = viz_cfg["radius"]
    vis.line_width = viz_cfg["thickness"]
    pv = pose3d_vis.pose2d_model.visualizer
    vis.det_kpt_color = pv.kpt_color
    vis.det_dataset_skeleton = pv.skeleton
    vis.det_dataset_link_color = pv.link_color

    vis.add_datasample(
        "pose_lift_3d",
        img_rgb,
        data_sample=merged_3d,
        det_data_sample=det_merged,
        draw_gt=False,
        draw_pred=True,
        draw_2d=False,
        draw_bbox=False,
        show=False,
        dataset_2d=pose3d_vis.pose2d_model.model.dataset_meta["dataset_name"],
        dataset_3d=pose3d_vis.model.dataset_meta["dataset_name"],
        kpt_thr=viz_cfg.get("kpt_thr", 0.3),
        num_instances=viz_cfg["num_instances"],
        out_file=out_path,
    )


@run_inference_entrypoint
def main(config: dict[str, Any]) -> None:
    logging.info(f'RUNNING 3D pose lift from 2D NPZ — {config["algorithm"]}!')

    npz_path = config["motionbert_2d_keypoints_npz"]
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"motionbert_2d_keypoints_npz not found: {npz_path!r}. "
            "Run the upstream body_joints detector declared in input_detector_names first."
        )

    pose_det_dataset = config["motionbert_2d_pose_det_dataset"]
    required_assets = config["required_assets"]
    pose3d_cfg = required_assets["pose3d_config_file"]
    pose3d_ckpt = required_assets["pose3d_checkpoint"]
    device = config["device"]

    coco_indices = config["motionbert_2d_coco_body_indices"]

    bundle = np.load(npz_path, allow_pickle=True)
    kpts_all = bundle["2d"]
    bbox_all = bundle["bbox_2d"]
    # The camera axis of the source arrays follows the upstream detector's own camera
    # order, which is independent of this detector's camera_names.
    source_camera_names = list(bundle["data_description"].item()["2d"]["axis1"])

    k_src = kpts_all.shape[-2]
    if max(coco_indices) >= k_src:
        raise ValueError(
            f"motionbert_2d_coco_body_indices max {max(coco_indices)} >= source keypoints {k_src} in {npz_path}."
        )

    lift_model = init_model(pose3d_cfg, pose3d_ckpt, device=device)
    lift_model.eval()
    pose_lift_ds = lift_model.cfg.test_dataloader.dataset
    causal = pose_lift_ds.get("causal", False)
    seq_len = pose_lift_ds.get("seq_len", 1)
    seq_step = pose_lift_ds.get("seq_step", 1)
    pose_lift_dataset = lift_model.dataset_meta["dataset_name"]

    height, width = _read_image_size_from_dataloader(config)
    dataloader = ImagePathsByCameraLoader(config=config, expected_cameras=config["camera_names"])
    start_frame, end_frame = dataloader.get_frames_range()
    camera_paths = list(dataloader)

    subjects_in_any_camera_view = [
        s
        for i, s in enumerate(config["subjects_descr"])
        if any(i in config["cam_sees_subjects"].get(cam) for cam in config["camera_names"])
    ]

    number_subjects = max(len(config["cam_sees_subjects"][c]) for c in config["camera_names"])
    fallback_k = config["keypoints_indices"]["body_joints_local"]
    if not len(fallback_k):
        raise ValueError("keypoints_indices.body_joints_local must be non-empty for 3D lift export.")
    fallback_num_keypoints = max(fallback_k) + 1

    camera_keypoints_output: list[np.ndarray] = []
    camera_keypoints_3d_output: list[np.ndarray] = []
    camera_bbox_output: list[np.ndarray] = []

    want_mmpose_frames = bool(config["save_detector_images"]) or bool(config["visualize"])
    disable_rebase = False

    pose3d_vis_inferencer: Pose3DInferencer | None = None
    if want_mmpose_frames:
        apply_pose3d_visualizer_rgb_reshape_fix()
        if "pose_config_file" in required_assets:
            pose2d_path = required_assets["pose_config_file"]
        else:
            pose2d_path = config["pose_config"]
        pose3d_vis_inferencer = Pose3DInferencer(
            pose3d_cfg,
            pose3d_ckpt,
            pose2d_model=pose2d_path,
            pose2d_weights=required_assets["pose_checkpoint"],
            det_model=required_assets["detection_config"],
            det_weights=required_assets["detection_checkpoint"],
            det_cat_ids=[0],
            device=device,
        )
        pose3d_vis_inferencer._video_input = False
        vtmp = getattr(pose3d_vis_inferencer, "visualizer", None)
        if vtmp is None:
            logging.warning("MMPose 3D visualizer is None; skipping MMPose frame export.")
            pose3d_vis_inferencer = None
        else:
            vtmp._nicetoolbox_multi_3d_layout = bool(config["mmpose_3d_nice_multi_layout"])
            vtmp._nicetoolbox_pelvis_sep_scale = float(config["pelvis_sep_scale"])
            logging.info("3D lifter: Pose3dLocalVisualizer loaded for frame export (inferencer not used for lifting).")

    viz_kwargs = {
        "num_instances": -1,
        "radius": 4,
        "thickness": 2,
        "kpt_thr": 0.3,
    }

    for camera_name, image_paths in camera_paths:
        logging.info("Camera - %s (3D lift from 2D NPZ)", camera_name)
        if camera_name not in source_camera_names:
            raise ValueError(
                f"Camera {camera_name!r} is not present in the source 2D NPZ {npz_path!r}, which contains "
                f"{source_camera_names}. Align camera_names between motionbert and its upstream "
                "body_joints detector."
            )
        cam_idx = source_camera_names.index(camera_name)
        vis_cam_dir = ""
        cam_subjects = config["cam_sees_subjects"][camera_name]
        f_count = kpts_all.shape[2]
        if len(image_paths) != f_count:
            logging.warning(
                "Frame count mismatch: NPZ has %s frames, loader has %s for %s — "
                "lifting uses NPZ length; vis skips missing paths.",
                f_count,
                len(image_paths),
                camera_name,
            )

        pose_est_results_list: list[list[PoseDataSample]] = []
        for f in range(f_count):
            k_cam = kpts_all[:, cam_idx, f, :, :]
            b_cam = bbox_all[:, cam_idx, f, :, :]
            k_rows = np.stack([k_cam[s] for s in cam_subjects], axis=0)
            b_rows = np.stack([b_cam[s] for s in cam_subjects], axis=0)
            pose_est_results_list.append(
                _build_frame_persons_from_npz(k_rows, b_rows, pose_det_dataset, pose_lift_dataset, coco_indices)
            )

        if want_mmpose_frames and pose3d_vis_inferencer is not None:
            vis_cam_dir = _mmpose_visualization_dir(config, camera_name)
            os.makedirs(vis_cam_dir, exist_ok=True)
            logging.info("MMPose 3D vis (from 2D NPZ) -> %s", vis_cam_dir)

        frame_results: list[dict[str, Any]] = []
        for f in range(f_count):
            k_cam = kpts_all[:, cam_idx, f, :, :]
            b_cam = bbox_all[:, cam_idx, f, :, :]
            k_rows = np.stack([k_cam[s] for s in cam_subjects], axis=0)
            b_rows = np.stack([b_cam[s] for s in cam_subjects], axis=0)

            pose_seq_2d = extract_pose_sequence(
                pose_est_results_list, frame_idx=f, causal=causal, seq_len=seq_len, step=seq_step
            )
            pose_lift_results = inference_pose_lifter_model(
                lift_model,
                pose_seq_2d,
                with_track_id=True,
                image_size=(width, height),
                norm_pose_2d=True,
            )
            current_2d = pose_est_results_list[f]
            if not pose_lift_results:
                frame_results.append({"predictions": [[]]})
                continue
            _postprocess_pose_lift_results(pose_lift_results, current_2d, disable_rebase)

            if want_mmpose_frames and pose3d_vis_inferencer is not None and vis_cam_dir and f < len(image_paths):
                _maybe_visualize_frame(
                    pose3d_vis_inferencer,
                    image_paths[f],
                    int(start_frame) + f,
                    pose_lift_results,
                    k_rows,
                    b_rows,
                    coco_indices,
                    vis_cam_dir,
                    viz_kwargs,
                )

            merged = merge_data_samples(pose_lift_results)
            preds = split_instances(merged.pred_instances)
            for pi, pred_dict in enumerate(preds):
                lifted = pred_dict["keypoints"]
                pred_dict["keypoints_3d"] = lifted
                coco_sl = k_rows[pi][list(coco_indices)]
                pred_dict["keypoints"] = coco_sl[:, :2].tolist()
                pred_dict["bbox"] = [b_rows[pi, 0, :4].astype(float).tolist()]
                pred_dict["bbox_score"] = float(b_rows[pi, 0, 4])
            frame_results.append({"predictions": [preds]})

        keypoints_array, keypoints_3d_array, bbox_array, estimations_data_descr = convert_output_to_numpy(
            frame_results,
            cam_subjects,
            number_subjects,
            fallback_num_keypoints,
            True,
        )
        camera_keypoints_output.append(keypoints_array)
        camera_bbox_output.append(bbox_array)
        if keypoints_3d_array is not None:
            camera_keypoints_3d_output.append(keypoints_3d_array)

        if want_mmpose_frames and vis_cam_dir:
            logging.info("MMPose 3D visualization frames written -> %s", vis_cam_dir)

    frame_indices = [f"{idx:09d}" for idx in range(start_frame, end_frame)]

    for component, result_folder in config["result_folders"].items():
        indices = config["keypoints_indices"][component]
        data_desc = {
            "2d": {
                "axis0": subjects_in_any_camera_view,
                "axis1": config["camera_names"],
                "axis2": frame_indices,
                "axis3": config["keypoints_description"][component],
                "axis4": estimations_data_descr["2d"],
            },
            "bbox_2d": {
                "axis0": subjects_in_any_camera_view,
                "axis1": config["camera_names"],
                "axis2": frame_indices,
                "axis3": ["full_body"],
                "axis4": estimations_data_descr["bbox_2d"],
            },
        }
        out_dict = {
            "2d": np.stack(camera_keypoints_output, axis=1)[:, :, :, indices],
            "bbox_2d": np.stack(camera_bbox_output, axis=1),
            "data_description": data_desc,
        }
        if camera_keypoints_3d_output:
            out_dict["3d"] = np.stack(camera_keypoints_3d_output, axis=1)[:, :, :, indices]
            data_desc["3d"] = {
                "axis0": subjects_in_any_camera_view,
                "axis1": config["camera_names"],
                "axis2": frame_indices,
                "axis3": config["keypoints_description"][component],
                "axis4": estimations_data_descr["3d"],
            }
        save_file_name = os.path.join(result_folder, f"{config['algorithm']}.npz")
        np.savez_compressed(save_file_name, **out_dict)

    logging.info(f'3D pose lift from 2D NPZ — {config["algorithm"]} COMPLETED!\n')
