import numpy as np
import os
import scipy.signal as signal
import json
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

### ToDo apply_savgol_filter should be implemented in ISA-Tool gaze
def apply_savgol_filter(data, windows_length=11, polyorder=2):
    """
    Takes data as an input and apply filter to each dimension of the data separately.

    Parameters
    ----------
    data: for two people list - np array, shape - [#Frames, XYZ]
    windows_length: int
    polyorder: int - the degree of polynomial, should be less than windows_length

    Returns:
    smoothed data(np array, shape - [#Frames, XYZ]
    """
    smooth_data = np.zeros(data.shape)
    for dim in range(data.shape[-1]):
        smooth_data[:, dim] = signal.savgol_filter(data[:, dim], window_length=windows_length,
                                                                 polyorder=polyorder)
    return smooth_data

def load_calibration(calibration_file, video_input_config, camera_names='all'):
    calib = None
    if video_input_config['sequence_ID'] == '':
        loaded_calib = np.load(calibration_file, allow_pickle=True)[video_input_config['session_ID']].item()
        if camera_names == 'all':
            calib = dict((key, value) for key, value in loaded_calib.items())
        else:
            calib = dict((key, value) for key, value in loaded_calib.items()
                         if key in camera_names)
    else:
        # TODO - test with other calibrations
        loaded_calib = np.load(calibration_file, allow_pickle=True)[video_input_config['session_ID']].item()[video_input_config['sequence_ID']]
        if camera_names == 'all':
            calib = dict((f"video_{key}", value) for key, value in loaded_calib.items())
        else:
            calib = dict((f"video_{key}", value) for key, value in loaded_calib.items()
                         if f"video_{key}" in camera_names)
    return calib


def project_points_using_cv2(points, cam_int, cam_ext, dist):
    """
    Project points using cv2 projectPoints function.
    :param points: (N, 3) points (world space)
    :param cam_int: (3,3) camera intrinsic matrix
    :param cam_ext: (4,4) camera extrinsic matrix
    :return: (N, 2) projected points
    """
    t_vec = cam_ext[:3, 3]
    r_mat = cam_ext[:3, :3]
    r_vec = R.from_matrix(r_mat).as_rotvec()
    projected_points, _ = cv2.projectPoints(points, r_vec, t_vec, cam_int, dist)
    return projected_points.squeeze()


def get_cam_para_studio(content, cam):
    cam_matrix = content[cam]['intrinsic_matrix'] if 'intrinsic_matrix' in content[cam].keys() else None
    cam_matrix = np.vstack(cam_matrix)
    cam_distor = content[cam]['distortions'] if 'distortions' in content[cam].keys() else None
    cam_distor = np.hstack(cam_distor)
    cam_rotation = content[cam]['rotation_matrix'] if 'rotation_matrix' in content[cam].keys() else None
    cam_extrinsic = content[cam]['extrinsics_matrix'] if 'extrinsics_matrix' in content[cam].keys() else None
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
    #(h, w) = image_in.shape[:2]
    length = image_width / length_ratio
    dx = (-length * np.sin(pitchyaw[:,1]) * np.cos(pitchyaw[:,0])).astype(int)
    dy = (-length * np.sin(pitchyaw[:,0])).astype(int)
    return dx, dy

def vector_to_pitchyaw(vector):
    ##Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.
    # Ensure vectors are in the shape (num_frames, 3)
    assert vector.shape[1] == 3, "Input vectors must have shape (num_vectors, 3)"

    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    unit_vector = vector/norm
    pitch = np.arcsin(-1 * unit_vector[:,1])  # theta
    yaw = np.arctan2(-1 * unit_vector[:,0], -1 * unit_vector[:,2])  # phi
    return np.vstack((pitch, yaw)).T


def reproject_gaze_to_camera_view_vectorized(cam_rotation, gaze_vectors, image_width):
    # Convert the gaze to current camera coordinate system for multiple vectors
    gaze_cam = np.dot(cam_rotation, gaze_vectors.T).T # Apply rotation to all gaze vectors
    draw_gaze_dir = vector_to_pitchyaw(gaze_cam)
    dx, dy = return_2d_vector(image_width, draw_gaze_dir)
    return dx, dy









