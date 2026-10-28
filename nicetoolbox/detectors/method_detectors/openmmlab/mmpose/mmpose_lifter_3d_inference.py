# mmpose API doesn't have proper way to uplift provided 2d data
# and prefer to run inference again
# so we have to do a bit of hacking around it

import logging
import os
import pickle
from typing import Dict, List

import mmcv
import numpy as np
from mmengine.infer.infer import BaseInferencer
from mmengine.structures import InstanceData
from mmpose.apis import convert_keypoint_definition, extract_pose_sequence, inference_pose_lifter_model, init_model
from mmpose.structures import PoseDataSample
from mmpose.visualization import Pose3dLocalVisualizer

from nicetoolbox_core.data.loaded_array import load_array_from_path, select_array
from nicetoolbox_core.entrypoint import SubprocessResult, run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByCameraLoader

RAW_INFERENCE_PICKLE_NAME = "mmpose_lifter_3d_inference_raw.pkl"
COCO_BODY_INDICES = list(range(17))


# native mmpose visualizer
# it has a bug and can't visualize multiple people - so we visualize only one
def _visualize_person(visualizer, frame_path, pose_lift_result, out_path):
    image_rgb = mmcv.imread(frame_path, channel_order="rgb")
    # convert 3d coodrdinates to matplotlib format
    vis_sample = PoseDataSample()
    vis_sample.pred_instances = pose_lift_result.pred_instances.clone()
    kpts = np.asarray(vis_sample.pred_instances.keypoints).copy()
    kpts = kpts[..., [0, 2, 1]]
    kpts[..., 0] *= -1
    kpts[..., 2] *= -1
    vis_sample.pred_instances.keypoints = kpts
    vis_sample.metainfo["ori_shape"] = image_rgb.shape[:2]
    vis_sample.metainfo["img_path"] = frame_path

    # send to native mmpose visualizer
    visualizer.add_datasample(
        "pose3d",
        image_rgb,
        data_sample=vis_sample,
        det_data_sample=None,
        draw_gt=False,
        draw_pred=True,
        draw_2d=True,
        draw_bbox=False,
        show=False,
        dataset_2d="coco",
        dataset_3d="h36m",
        kpt_thr=0.3,
        num_instances=1,
        out_file=out_path,
    )


def _one_sample(kpts_h36m: np.ndarray, scores: np.ndarray, bbox_xyxy: np.ndarray, track_id: int) -> PoseDataSample:
    # repackage npz single sample to mmpose
    ds = PoseDataSample()
    pred = InstanceData()
    pred.set_field(kpts_h36m[np.newaxis, ...].astype(np.float32), "keypoints")
    pred.set_field(scores[np.newaxis, ...].astype(np.float32), "keypoint_scores")
    pred.set_field(bbox_xyxy[np.newaxis, ...].astype(np.float32), "bboxes")
    pred.set_field(np.array([1.0], dtype=np.float32), "bbox_scores")
    ds.pred_instances = pred
    ds.gt_instances = InstanceData()
    ds.set_field(track_id, "track_id")
    return ds


@run_inference_entrypoint
def main(config: dict) -> None:
    logging.info("RUNNING 3D pose lift (mmpose_lifter_3d)!")

    out_folder = config["algorithm_out_folder"]
    camera_names = config["camera_names"]
    required_assets = config["required_assets"]
    pose_2d_npz = config["pose_2d_npz"]
    pose_2d_npz_key = config["pose_2d_npz_key"]
    person_bbox_npz = config["person_bbox_npz"]
    person_bbox_npz_key = config["person_bbox_npz_key"]
    native_viz_folder = config["native_viz_folder"]
    visualize_native = config["visualize_native"]

    # ! we hardcode only motionbert cfg to simplify path resolution
    # other lifter models aren't supported
    pose3d_cfg = config["pose_config"]
    mmpose_mim = BaseInferencer._get_repo_or_mim_dir("mmpose")
    cgf_folder = os.path.join(mmpose_mim, "configs", "body_3d_keypoint")
    pose3d_cfg_path = os.path.join(cgf_folder, "motionbert", "h36m", f"{pose3d_cfg}.py")
    pose3d_ckpt = required_assets["pose_checkpoint"]

    # upstream detectors may run on a different camera set/order, align both arrays to ours
    # load upstream 2D pose bundle
    pose_2d_raw = load_array_from_path(pose_2d_npz, pose_2d_npz_key)
    pose_2d = select_array(pose_2d_raw, cameras=camera_names)
    kpts_all = pose_2d.data  # (S, C, F, K, 3)
    source_frames = list(pose_2d.axes.frames)

    # load upstream person bboxes: (S, C, F, 5) - xyxy + score, per subject slot
    person_bboxes_raw = load_array_from_path(person_bbox_npz, person_bbox_npz_key)
    person_bboxes = select_array(person_bboxes_raw, cameras=camera_names)
    bboxes_all = person_bboxes.data

    # init lifter model
    lift_model = init_model(pose3d_cfg_path, pose3d_ckpt, device="cuda:0")
    lift_model.eval()

    # ! copy-pasted from mmpose 3d inferencer
    # init lifter model parameters
    ds_cfg = lift_model.cfg.test_dataloader.dataset
    causal = ds_cfg.get("causal", False)
    seq_len = ds_cfg.get("seq_len", 1)
    seq_step = ds_cfg.get("seq_step", 1)

    n_subjects, _, n_frames = kpts_all.shape[:3]

    # loader gives us the cameras this detector should process + their probed [w, h]
    dataloader = ImagePathsByCameraLoader(config=config, expected_cameras=camera_names)

    # native visualizer — one instance for the whole subprocess
    if visualize_native:
        visualizer = Pose3dLocalVisualizer(name="motionbert_native_vis")
        visualizer.set_dataset_meta(lift_model.dataset_meta)

    # for each camera
    per_camera_outputs: Dict[str, List[list]] = {}
    for cam_name, frame_paths in dataloader.with_progress(desc="MotionBERT"):
        cam_idx = camera_names.index(cam_name)
        logging.info(f"Lifting camera '{cam_name}'...")

        # prepare folder for native visualization
        if visualize_native:
            vis_camera_dir = os.path.join(native_viz_folder, cam_name)
            os.makedirs(vis_camera_dir, exist_ok=True)

        # MotionBERT normalizes 2D coords by the source frame's dimensions
        image_size = tuple(dataloader.resolutions[cam_name])

        # Pass 1: wrap every (frame, subject) into a PoseDataSample. track_id = subject slot.
        # Skip subjects with no valid 2D keypoints in a frame — avoids lifting garbage.
        # Track subject_id ordering per frame so we can map results[i] back to a subject slot.
        pose_est_list: List[List[PoseDataSample]] = []
        subject_ids_per_frame: List[List[int]] = []
        for f in range(n_frames):
            frame_samples = []
            frame_subject_ids = []
            for s in range(n_subjects):
                kpts = kpts_all[s, cam_idx, f, COCO_BODY_INDICES, :]  # (17, 3)
                if np.all(np.isnan(kpts[:, :2])):  # subject absent this frame
                    continue
                # real bbox from upstream mmdet (xyxy). Fall back to keypoint hull if bbox missing.
                bbox_row = bboxes_all[s, cam_idx, f, :4]
                if np.any(np.isnan(bbox_row)):
                    xy_valid = kpts[:, :2][~np.isnan(kpts[:, :2]).any(axis=-1)]
                    x1, y1 = xy_valid.min(axis=0)
                    x2, y2 = xy_valid.max(axis=0)
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                else:
                    bbox = bbox_row.astype(np.float32)
                # nan will crash mmpose, replace with 0
                xy = np.where(np.isnan(kpts[:, :2]), 0.0, kpts[:, :2])
                sc = np.where(np.isnan(kpts[:, 2]), 0.0, kpts[:, 2])
                # convert from coco to motionbert h36m
                xy_h36m = convert_keypoint_definition(xy[None, ...], "coco", "h36m")[0]
                frame_samples.append(_one_sample(xy_h36m, sc, bbox, track_id=s))
                frame_subject_ids.append(s)
            pose_est_list.append(frame_samples)
            subject_ids_per_frame.append(frame_subject_ids)

        # Pass 2: for each frame, build the temporal window and run the lift
        camera_frames: List[list] = []
        for f in range(n_frames):
            # skip frames where no subjects are present — nothing to lift, no PNGs
            if not pose_est_list[f]:
                camera_frames.append([])
                continue

            # get temporal pose sequence
            pose_seq_2d = extract_pose_sequence(
                pose_est_list, frame_idx=f, causal=causal, seq_len=seq_len, step=seq_step
            )
            # run inference
            results = inference_pose_lifter_model(
                lift_model,
                pose_seq_2d,
                with_track_id=True,
                image_size=image_size,
                norm_pose_2d=True,
            )

            # normalize shape once: lifter emits keypoints (1, T, K, 3), scores (1, T, K);
            # both viz and result-serialization want the T axis gone.
            for res in results:
                kpts_3d = np.asarray(res.pred_instances.keypoints)
                scores = np.asarray(res.pred_instances.keypoint_scores)
                if kpts_3d.ndim == 4:
                    kpts_3d = np.squeeze(kpts_3d, axis=1)
                if scores.ndim == 3:
                    scores = np.squeeze(scores, axis=1)
                res.pred_instances.keypoints = kpts_3d
                res.pred_instances.keypoint_scores = scores

            # results[i] corresponds to the i-th subject present in the target frame
            persons = []
            for i, res in enumerate(results):
                subj_idx = subject_ids_per_frame[f][i]
                # native visualization: one PNG per subject per frame
                if visualize_native:
                    subj_dir = os.path.join(vis_camera_dir, f"subject_{subj_idx}")
                    os.makedirs(subj_dir, exist_ok=True)
                    out_path = os.path.join(subj_dir, f"{source_frames[f]}.png")
                    _visualize_person(visualizer, frame_paths[f], res, out_path)
                # save final results
                persons.append(
                    {
                        "subject_idx": subj_idx,
                        "keypoints_3d": np.asarray(res.pred_instances.keypoints)[0].tolist(),
                        "score": np.asarray(res.pred_instances.keypoint_scores)[0].tolist(),
                    }
                )
            camera_frames.append(persons)
        per_camera_outputs[cam_name] = camera_frames

    save_path = os.path.join(out_folder, RAW_INFERENCE_PICKLE_NAME)
    with open(save_path, "wb") as raw_file:
        pickle.dump(per_camera_outputs, raw_file)
    logging.info(f"Pose lift 'mmpose_lifter_3d' raw pack COMPLETED! Wrote '{save_path}'.\n")

    return SubprocessResult.from_artifacts(raw_output=save_path)
