import logging
import os
import pickle
from typing import Dict, List

import numpy as np
from mmpose.apis import Pose2DInferencer, inference_topdown
from mmpose.structures import merge_data_samples

from nicetoolbox_core.data.loaded_array import load_array_from_path, select_array
from nicetoolbox_core.entrypoint import SubprocessResult, run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByCameraLoader

RAW_INFERENCE_PICKLE_NAME = "mmpose_inference_raw.pkl"


@run_inference_entrypoint
def main(config: dict) -> None:
    logging.info("RUNNING pose estimation mmpose!")

    out_folder = config["algorithm_out_folder"]
    camera_names = config["camera_names"]
    required_assets = config["required_assets"]
    person_bbox_npz = config["person_bbox_npz"]
    person_bbox_npz_key = config["person_bbox_npz_key"]
    save_raw_json_predictions = bool(config["save_raw_json_predictions"])
    native_viz_folder = config["native_viz_folder"]
    visualize_native = bool(config["visualize_native"])
    decode_visibility = bool(config["decode_visibility"])

    dataloader = ImagePathsByCameraLoader(config=config, expected_cameras=camera_names)

    # initialize mmpose model using Pose2DInferencer
    # we use it only to load model and resolve configs
    pose_inferencer = Pose2DInferencer(
        model=config["pose_config"],  # TODO: file/alias config resolution like in mmdet?
        weights=required_assets["pose_checkpoint"],
        device="cuda:0",
        det_model="whole_image",  # need this one to keep bounding boxes
    )
    # unfortunately, existing API discard pre-calculated bboxes
    # so we run models directly
    model = pose_inferencer.model

    # visibility is a decode-time option of the SimCC codec (RTMPose/RTMW); heatmap codecs
    # (ViTPose, HRNet) don't have it, their keypoint scores are already roughly in [0, 1]
    decoder = getattr(model.head, "decoder", None)
    if hasattr(decoder, "decode_visibility"):
        decoder.decode_visibility = decode_visibility
    elif decode_visibility:
        logging.warning(
            f"decode_visibility is not supported by the '{type(decoder).__name__}' codec, "
            "visibility will be exported as None."
        )
        decode_visibility = False

    # nice bboxes: (subjects, cameras, frames, 5) [x1, y1, x2, y2, conf]
    # bbox detector may run on a different camera set/order, align it to ours
    person_bboxes = load_array_from_path(person_bbox_npz, person_bbox_npz_key)
    bboxes_all = select_array(person_bboxes, cameras=camera_names).data

    # for each camera
    per_camera_outputs: Dict[str, List[list]] = {}
    for camera_name, frame_paths in dataloader.with_progress(desc="MMPose 2D"):
        cam_idx = camera_names.index(camera_name)
        camera_out_dir = os.path.join(out_folder, camera_name)
        vis_out_dir = os.path.join(native_viz_folder, camera_name)

        camera_frames: List[list] = []
        for frame_idx, frame_path in enumerate(frame_paths):
            # per-subject bboxes for this (camera, frame); drop rows with NaN.
            frame_bboxes = bboxes_all[:, cam_idx, frame_idx, :4]
            valid_mask = ~np.isnan(frame_bboxes).any(axis=1)
            if not valid_mask.any():
                camera_frames.append([])
                continue  # TODO: we break native visualization and json continuity (skip empty frames)
            bboxes_ndarray = frame_bboxes[valid_mask].astype(np.float32)  # (N, 4)
            valid_subject_indices = np.where(valid_mask)[0]  # canonical subject slot per bbox row

            # TODO: cross-frame batching? would require some mmpose hacking
            # inference_topdown stacks the N person crops
            results = inference_topdown(model, frame_path, bboxes_ndarray)

            # save predictions results
            persons = []
            for pi, res in enumerate(results):
                pred = res.pred_instances
                person = {
                    "subject_idx": int(valid_subject_indices[pi]),
                    "keypoints": pred.keypoints[0].tolist(),
                    "score": pred.keypoint_scores[0].tolist(),
                }
                # check if optional "visibility" information is available
                if decode_visibility:
                    person["visibility"] = pred.keypoints_visible[0].tolist()
                else:
                    person["visibility"] = None
                persons.append(person)
            camera_frames.append(persons)

            # run native visualization and export
            if save_raw_json_predictions or visualize_native:
                merged = merge_data_samples(results)
                # native visualization
                if visualize_native:
                    pose_inferencer.visualize(
                        inputs=[frame_path],
                        preds=[merged],
                        draw_bbox=True,
                        vis_out_dir=vis_out_dir,
                    )
                # native json export
                if save_raw_json_predictions:
                    pose_inferencer.postprocess(
                        preds=[merged],
                        visualization=[],
                        pred_out_dir=camera_out_dir,
                    )
        per_camera_outputs[camera_name] = camera_frames

    # save to raw pickle
    save_file_name = os.path.join(out_folder, RAW_INFERENCE_PICKLE_NAME)
    with open(save_file_name, "wb") as raw_file:
        pickle.dump(per_camera_outputs, raw_file)
    logging.info(f"Pose estimation 'mmpose' raw pack COMPLETED! Wrote '{save_file_name}'.\n")

    return SubprocessResult.from_artifacts(raw_output=save_file_name)
