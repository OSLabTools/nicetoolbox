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

from nicetoolbox_core.dataloader import ImagePathsByFrameIndexLoader
from nicetoolbox_core.entrypoint import run_inference_entrypoint


@run_inference_entrypoint
def eth_xgaze_inference(config, debug=False):
    """
    Run multiview-eth-xgaze gaze detection on the provided data.
    code Code

    The function uses the 'multiview-eth-xgaze gaze' library to estimate gaze vectors
    for each camera. The estimated gaze vectors are converted to pitch and yaw angles
    using a simple linear transformation. The resulting angles are saved in a .npz
    file with the following structure:
        - 3d: Numpy array of shape (n_frames, n_subjects, 3)
        - data_description: A dictionary containing the description of the data.

    Args:
        config (dict): The configuration dictionary containing parameters for gaze
            detection.
        debug (bool, optional): A flag indicating whether to print debug information.
            Defaults to False.
    """

    logging.info("RUNNING gaze detection 'Multiview ETH-Xgaze'!")

    # (1) Access config parameters
    camera_names = config["camera_names"]
    subjects_descr = config["subjects_descr"]
    cam_sees_subjects = config["cam_sees_subjects"]
    calibration = config["calibration"]

    # (2) Prepare data loader
    dataloader = ImagePathsByFrameIndexLoader(config=config, expected_cameras=camera_names)
    n_frames = len(dataloader)
    n_subjects = len(subjects_descr)
    n_cams = len(camera_names)

    # (3) Initialize gaze estimator and face detector
    logging.info("Load gaze estimator and start detection.")
    gaze_estimator = GazeEstimator(config["face_model_filename"], config["pretrained_model_filename"])

    face_detector = lm.get_face_detector(config["shape_predictor_filename"], config["face_detector_filename"])

    # (4) Prepare to store results
    results = np.zeros((n_frames, n_subjects, 3))  # Back compatibility
    results_2d = np.zeros((n_frames, n_cams, n_subjects, 2))  # for single camera setup
    results_per_camera = np.full((n_subjects, n_cams, n_frames, 3), np.nan)  # raw
    landmarks_2d = np.full((n_frames, n_cams, n_subjects, 6, 3), np.nan)  # with scores

    # (5) Inference loop
    frame_indices = []
    for frame_idx, (real_frame_idx, frame_paths_per_camera) in enumerate(dataloader):
        # Store frame indices for saving later
        frame_indices.append(f"{real_frame_idx:09d}")

        images = {}  # Cache loaded images per camera: {camera_name: image}

        # (A) Face and Landmark Detection per camera
        for cam_idx, camera_name in enumerate(camera_names):
            if camera_name not in frame_paths_per_camera:
                logging.error(f"Camera '{camera_name}' not found in frame paths for frame " f"index {real_frame_idx}.")
                continue

            subjects_by_cam = cam_sees_subjects[camera_name]

            frame_file = frame_paths_per_camera[camera_name]
            image = cv2.imread(frame_file)
            images[camera_name] = image

            landmark_predictions, score = lm.get_landmarks(image, face_detector, debug)

            if landmark_predictions is not None:
                if landmark_predictions.shape[0] > len(subjects_by_cam):
                    # Too many faces detected, select the ones with highest scores
                    scores = score.mean(axis=1)
                    max_value_indices = sorted(scores.argsort()[-len(subjects_by_cam) :])
                    landmark_predictions = landmark_predictions[max_value_indices]
                    # Filter scores to match the selected faces
                    score = score[max_value_indices]

                elif landmark_predictions.shape[0] < len(subjects_by_cam):
                    logging.error(
                        f"Gaze landmark detection: Detected "
                        f"{landmark_predictions.shape[0]} subjects instead of "
                        f"{len(subjects_by_cam)} in frame file {frame_file}."
                    )
                    # When the landmark detection is missing for one subject.
                    # We assign missing for all subjects,
                    # bec. not possible to know which subject is missing
                    landmark_predictions = np.full(
                        (
                            len(subjects_by_cam),
                            landmark_predictions.shape[1],
                            landmark_predictions.shape[2],
                        ),
                        np.nan,
                    )
                    # Also pad scores
                    score = np.full((len(subjects_by_cam), 6), np.nan)

                # Add confidence scores to the landmarks
                # 1. Expand scores: (N, 6) -> (N, 6, 1)
                scores_expanded = score[:, :, np.newaxis]
                # 2. Concatenate: (N, 6, 2) + (N, 6, 1) -> (N, 6, 3)
                landmarks_and_scores = np.concatenate((landmark_predictions, scores_expanded), axis=2)

                # Store landmarks with scores for the subjects seen by this camera
                landmarks_2d[frame_idx, cam_idx, subjects_by_cam] = landmarks_and_scores

        # (B) OK, now let's do gaze estimation - per subject
        for sub_id in range(n_subjects):
            gaze_world = []  # back compatibility

            # loop over the cameras and predict the gaze for each in 2d
            for camera_idx, camera_name in enumerate(camera_names):
                # Camera image loaded?
                if camera_name not in images:
                    continue

                # Subject visible in the camera?
                subjects_by_cam = cam_sees_subjects[camera_name]
                if sub_id not in subjects_by_cam:
                    continue

                landmarks = landmarks_2d[frame_idx, camera_idx, sub_id, :, :2]
                # TODO: Use confidence scores? Check if low confidence -> skip

                # Landmarks predicted?
                if np.isnan(landmarks).all():
                    continue

                # get camera parameters given the calibration
                image = images[camera_name]
                cam_matrix, cam_distor, cam_rotation, _ = get_cam_para_studio(calibration, camera_name, image)

                # Predict gaze dir in the camera coordinate system (Run Neural Net)
                pred_gaze = gaze_estimator.gaze_estimation(image, landmarks, cam_matrix, cam_distor)

                # Convert the gaze direction to the world coordinate system
                if pred_gaze != "":
                    pred_gaze_world = np.dot(np.linalg.inv(cam_rotation), pred_gaze).reshape((1, 3))
                    pred_gaze_world = pred_gaze_world / np.linalg.norm(pred_gaze_world)
                    # SAVE per camera results (RAW)
                    results_per_camera[sub_id, camera_idx, frame_idx] = pred_gaze_world

                    # Below: back compatibility
                    gaze_world.append(pred_gaze_world.reshape((1, 3)))
                else:
                    gaze_world.append(np.full((1, 3), np.nan))

            # Multiview fusion - TODO: Cut from here
            gaze_world = np.asarray(gaze_world).reshape((-1, 3))
            gaze_world = np.nanmean(gaze_world, axis=0).reshape((-1, 3))
            results[frame_idx][sub_id] = gaze_world.flatten()

        # (C) 2D Projection for single camera setup
        if len(camera_names) == 1:
            gaze_camera = np.matmul(cam_rotation, results[frame_idx].T).T
            # vector to pitchyaw
            pitchyaw = np.empty((n_subjects, 2))
            gaze_norm = gaze_camera / np.linalg.norm(gaze_camera, axis=1, keepdims=True)
            pitchyaw[:, 0] = np.arcsin(-1 * gaze_norm[:, 1])  # theta
            pitchyaw[:, 1] = np.arctan2(-1 * gaze_norm[:, 0], -1 * gaze_norm[:, 2])  # phi

            length_ratio = 5.0
            length = image.shape[1] / length_ratio
            dx = (-length * np.sin(pitchyaw[:, 1]) * np.cos(pitchyaw[:, 0])).astype(int)
            dy = (-length * np.sin(pitchyaw[:, 0])).astype(int)

            # shape (n_subjects, 2)
            results_2d[frame_idx, 0] = np.stack((dx, dy), axis=-1)

        # (D) Visualization
        if config["visualize"]:
            # create camera folders
            for camera_name in camera_names:
                image = images[camera_name].copy()
                camera_idx = camera_names.index(camera_name)

                out_dir = os.path.join(config["out_folder"], camera_name)
                os.makedirs(out_dir, exist_ok=True)

                _, _, cam_rotation, _ = get_cam_para_studio(calibration, camera_name, image)

                for sub_id in range(n_subjects):
                    if sub_id not in cam_sees_subjects[camera_name]:
                        continue

                    landmarks = landmarks_2d[frame_idx, camera_idx, sub_id, :, :2]
                    gaze_result = results[frame_idx][sub_id]

                    if not np.isnan(landmarks).all() and not np.isnan(gaze_result).all():
                        # project 3D gaze to 2D
                        gaze_cam = np.dot(cam_rotation, gaze_result.T)
                        gaze_2d_direction = vector_to_pitchyaw(gaze_cam[None]).reshape(-1)
                        face_center = np.nanmean(landmarks, axis=0)
                        draw_gaze(
                            image,
                            gaze_2d_direction,
                            thickness=2,
                            color=(0, 255, 0),
                            position=face_center.astype(int),
                        )

                if debug:
                    cv2.imshow("img_show", image)
                    cv2.waitKey(0)

                # Save the image with gaze direction
                save_file_name = os.path.join(out_dir, f"{real_frame_idx:09d}.jpg")
                cv2.imwrite(save_file_name, image)

        if frame_idx % config["log_frame_idx_interval"] == 0:
            logging.info(f"Finished frame {frame_idx} / {n_frames}.")

    if not config["visualize"]:
        logging.info("Visualization of images turned off.")

    #  save as npz file
    out_dict = {
        "3d_multiview": results[None].transpose(2, 0, 1, 3),
        "3d": results_per_camera,
        "landmarks_2d": np.array(landmarks_2d, dtype=float).transpose(2, 1, 0, 3, 4),
        "data_description": {
            "3d_multiview": dict(
                axis0=config["subjects_descr"],
                axis1=["3d"],
                axis2=frame_indices,
                axis3=["coordinate_x", "coordinate_y", "coordinate_z"],
            ),
            "3d": dict(
                axis0=config["subjects_descr"],  # Subjects
                axis1=camera_names,  # Cameras
                axis2=frame_indices,  # Frames
                axis3=["coordinate_x", "coordinate_y", "coordinate_z"],  # Data
            ),
            "landmarks_2d": dict(
                axis0=config["subjects_descr"],
                axis1=camera_names,
                axis2=frame_indices,
                axis3=[
                    "right_eye_0",
                    "right_eye_1",
                    "left_eye_0",
                    "left_eye_1",
                    "mouth_0",
                    "mouth_1",
                ],
                axis4=["coordinate_u", "coordinate_v", "confidence_score"],
            ),
        },
    }
    if len(camera_names) == 1:
        out_dict.update({"2d": results_2d.transpose(2, 1, 0, 3)})
        out_dict["data_description"].update(
            {
                "2d": dict(
                    axis0=config["subjects_descr"],
                    axis1=camera_names,
                    axis2=frame_indices,
                    axis3=["coordinate_x", "coordinate_y"],
                )
            }
        )

    save_file_name = os.path.join(config["result_folders"]["gaze_individual"], f"{config['algorithm']}.npz")
    np.savez_compressed(save_file_name, **out_dict)

    logging.info("Gaze detection 'Multiview ETH-XGaze' COMPLETED!\n")
