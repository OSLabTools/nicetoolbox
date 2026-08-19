"""
Run gaze detection on the provided data.
"""

import logging
import os

import cv2
import multiview_eth_xgaze.landmarks as lm
import numpy as np
from multiview_eth_xgaze.eth_xgaze.utils import vector_to_pitchyaw
from multiview_eth_xgaze.gaze_estimator import GazeEstimator
from multiview_eth_xgaze.xgaze_utils import draw_gaze, get_cam_para_studio

from nicetoolbox_core.entrypoint import run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

# Filename of the raw, native inference pack (distinct from the structured {algorithm}.npz
# that post_inference will produce). Kept as a module constant so the detector can import it.
RAW_INFERENCE_NPZ_NAME = "eth_xgaze_inference_raw.npz"


def _visualize_frame(config, camera_names, images, frame_bundle, real_frame_idx, debug):
    """Draw each detection's camera-local gaze arrow on its camera image (debug only).

    Kept self-contained in the inference script; the camera-local gaze is projected to a 2D
    pitch/yaw arrow via the same camera rotation used elsewhere.
    """
    for camera_name in camera_names:
        if camera_name not in images:
            continue
        image = images[camera_name].copy()

        for detection in frame_bundle.get(camera_name, []):
            landmarks = detection["landmarks_2d"]
            gaze_cam = detection["gaze_cam"]
            if np.isnan(landmarks).all() or np.isnan(gaze_cam).any():
                continue

            # gaze_cam is already in the camera frame; pitch/yaw give the 2D arrow direction.
            gaze_2d_direction = vector_to_pitchyaw(gaze_cam[None]).reshape(-1)
            face_center = np.nanmean(landmarks, axis=0)
            draw_gaze(image, gaze_2d_direction, thickness=2, color=(0, 255, 0), position=face_center.astype(int))

        if debug:
            cv2.imshow("img_show", image)
            cv2.waitKey(0)

        # Match the method-detector convention: detector_output/images/<camera_name>/.
        out_dir = os.path.join(config["out_folder"], "images", camera_name)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{real_frame_idx:09d}.jpg"), image)


@run_inference_entrypoint
def eth_xgaze_inference(config, debug=False):
    """Run ETH-XGaze and save the raw, native per-detection pack (see module docstring)."""
    logging.info("RUNNING gaze detection 'ETH-XGaze' (raw native save)!")

    # (1) Access config parameters
    camera_names = config["camera_names"]
    calibration = config["calibration"]
    # Patch TORCH_HOME OS env variable
    # face_alignment s3fd and 2DFAN4 models are used implicitly by torch
    # they should be already downloaded by the asset manager
    face_alignment_cache_dir = config["face_alignment_cache_dir"]
    os.environ["TORCH_HOME"] = face_alignment_cache_dir

    # (2) Prepare data loader
    dataloader = ImagePathsByFrameIndexLoader(config=config, expected_cameras=camera_names)

    # (3) Initialize gaze estimator and face detector
    logging.info("Load gaze estimator and start detection.")
    req_assets = config["required_assets"]

    gaze_estimator = GazeEstimator(req_assets["face_model_filename"], req_assets["pretrained_model_filename"])
    face_detector = lm.get_face_detector(None, None)  # TODO: both parameters never used

    # (4) Inference loop — accumulate ragged per-detection records (no dense toolbox arrays).
    per_frame_outputs = []

    for real_frame_idx, frame_paths_per_camera in dataloader:
        images = {}  # {camera_name: image}, cached for gaze estimation + viz

        # (A) Face + landmark detection per camera
        # {camera_name: (landmarks (n_faces, 6, 2), scores (n_faces, 6))}
        landmarks_by_cam = {}
        for camera_name, frame_path in frame_paths_per_camera.items():
            image = cv2.imread(frame_path)
            images[camera_name] = image

            landmark_predictions, score = lm.get_landmarks(image, face_detector, debug)
            if landmark_predictions is None:
                continue
            landmarks_by_cam[camera_name] = (landmark_predictions, score)

        # (B) Gaze estimation per (camera, detected face).
        frame_bundle = {}
        for camera_name, (landmark_predictions, scores) in landmarks_by_cam.items():
            image = images[camera_name]
            cam_matrix, cam_distor, _, _ = get_cam_para_studio(calibration, camera_name, image)

            detections = []
            for face_index in range(landmark_predictions.shape[0]):
                landmarks = landmark_predictions[face_index, :, :2]
                if np.isnan(landmarks).all():
                    continue

                # Gaze in the original camera coordinate frame (before any world rotation).
                pred_gaze = gaze_estimator.gaze_estimation(image, landmarks, cam_matrix, cam_distor)
                if not isinstance(pred_gaze, np.ndarray):
                    continue
                gaze_cam = np.asarray(pred_gaze, dtype=np.float32).reshape(3)

                detections.append(
                    {
                        "face_index": face_index,
                        "gaze_cam": gaze_cam,
                        "landmarks_2d": np.asarray(landmarks, dtype=np.float32),
                        "landmark_scores": np.asarray(scores[face_index], dtype=np.float32),
                    }
                )
            frame_bundle[camera_name] = detections

        per_frame_outputs.append(frame_bundle)
        if config["visualize_native"]:
            _visualize_frame(config, camera_names, images, frame_bundle, real_frame_idx, debug)

    # (5) Save the raw native pack. Object-array pickle is fine within the eth_xgaze env.
    out_dict = {"per_frame_outputs": np.asarray(per_frame_outputs, dtype=object)}

    save_file_name = os.path.join(config["out_folders"]["gaze_individual"], RAW_INFERENCE_NPZ_NAME)
    np.savez_compressed(save_file_name, **out_dict)
    logging.info(f"Gaze detection 'ETH-XGaze' raw pack COMPLETED! Wrote '{save_file_name}'.\n")
