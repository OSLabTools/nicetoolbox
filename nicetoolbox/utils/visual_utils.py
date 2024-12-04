"""
Helper functions for the visualizer module.
"""

import numpy as np


def load_calibration(calibration_file, video_input_config, camera_names="all"):
    calib = None

    calib_details = "__".join(
        [
            word
            for word in [
                video_input_config["session_ID"],
                video_input_config["sequence_ID"],
            ]
            if word
        ]
    )
    loaded_calib = np.load(calibration_file, allow_pickle=True)[calib_details].item()
    if camera_names == "all":
        calib = dict((key, value) for key, value in loaded_calib.items())
    else:
        calib = dict(
            (key, value) for key, value in loaded_calib.items() if key in camera_names
        )
    return calib


def get_cam_para_studio(content, cam):
    cam_matrix = (
        content[cam]["intrinsic_matrix"]
        if "intrinsic_matrix" in content[cam].keys()  # noqa: SIM118
        else None
    )
    cam_matrix = np.vstack(cam_matrix)
    cam_distor = (
        content[cam]["distortions"] if "distortions" in content[cam].keys() else None  # noqa: SIM118
    )
    cam_distor = np.hstack(cam_distor)
    cam_rotation = (
        content[cam]["rotation_matrix"]
        if "rotation_matrix" in content[cam].keys()  # noqa: SIM118
        else None
    )
    cam_extrinsic = (
        content[cam]["extrinsics_matrix"]
        if "extrinsics_matrix" in content[cam].keys()  # noqa: SIM118
        else None
    )
    if type(cam_rotation) is not list:
        cam_rotation = cam_rotation.tolist()
    if type(cam_extrinsic) is not list:
        cam_extrinsic = cam_extrinsic.tolist()

    if isinstance(cam_matrix, np.ndarray):
        cam_matrix = cam_matrix.tolist()
        cam_distor = cam_distor.tolist()
        cam_rotation = cam_rotation
        cam_extrinsic = cam_extrinsic
    return cam_matrix, cam_distor, cam_rotation, cam_extrinsic


def return_2d_vector(image_width, pitchyaw, length_ratio=5.0):
    # (h, w) = image_in.shape[:2]
    length = image_width / length_ratio
    dx = (-length * np.sin(pitchyaw[:, 1]) * np.cos(pitchyaw[:, 0])).astype(int)
    dy = (-length * np.sin(pitchyaw[:, 0])).astype(int)
    return dx, dy


def vector_to_pitchyaw(vector):
    # Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`)
    # angles.Ensure vectors are in the shape (num_frames, 3)
    assert vector.shape[1] == 3, "Input vectors must have shape (num_vectors, 3)"

    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    unit_vector = vector / norm
    pitch = np.arcsin(-1 * unit_vector[:, 1])  # theta
    yaw = np.arctan2(-1 * unit_vector[:, 0], -1 * unit_vector[:, 2])  # phi
    return np.vstack((pitch, yaw)).T


def reproject_gaze_to_camera_view_vectorized(cam_rotation, gaze_vectors, image_width):
    # Convert the gaze to current camera coordinate system for multiple vectors
    gaze_cam = np.dot(
        cam_rotation, gaze_vectors.T
    ).T  # Apply rotation to all gaze vectors
    draw_gaze_dir = vector_to_pitchyaw(gaze_cam)
    dx, dy = return_2d_vector(image_width, draw_gaze_dir)
    return dx, dy
