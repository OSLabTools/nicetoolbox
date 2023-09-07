import os
import cv2
import numpy as np
import json
import re


def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, landmarks_warped, R, face_center


def vector_to_pitchyaw(vector):
    ##Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.
    out = np.empty((1, 2))
    vector = vector / np.linalg.norm(vector)
    out[:, 0] = np.arcsin(-1 * vector[1])  # theta
    out[:, 1] = np.arctan2(-1 * vector[0], -1 * vector[2])  # phi
    return out


def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """

    pred_theta = pitchyaws[0, 0]
    pred_phi = pitchyaws[0, 1]
    data_x = -1 * np.cos(pred_theta) * np.sin(pred_phi)
    data_y = -1 * np.sin(pred_theta)
    data_z = -1 * np.cos(pred_theta) * np.cos(pred_phi)
    out = np.empty((1, 3))
    out[0, 0] = data_x
    out[0, 1] = data_y
    out[0, 2] = data_z

    return out


def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255), position=None, length_ratio=5.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / length_ratio
    if position is None:
        position = (int(w / 2.0), int(h / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = int(-length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0]))
    dy = int(-length * np.sin(pitchyaw[0]))
    cv2.arrowedLine(image_out, tuple(np.round(position).astype(np.int32)),
                   tuple(np.round([position[0] + dx, position[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out


def get_cam_para_studio(content, cam_id, image):
    cam = 'cam' + str(cam_id)
    cam_matrix = content['2020-08-11'][cam]['intrinsic_matrix']
    cam_matrix = np.vstack(cam_matrix)
    cam_distor = content['2020-08-11'][cam]['distortions']
    cam_distor = np.hstack(cam_distor)
    cam_rotation = content['2020-08-11'][cam]['rvec']
    cam_rotation = np.hstack(cam_rotation)
    cam_rotation = cv2.Rodrigues(cam_rotation)[0]  # change to be rotation matrix
    # cam_translation = content['2020-08-11']['cam'+str(cam_id)]['tvec']  # we do not need translation
    # cam_translation = np.vstack(cam_translation)

    cam_resolution = sorted(content['2020-08-11'][cam]['image_size'])
    img_resolution = sorted(image.shape[:2])
    if cam_resolution != img_resolution:
        print(f"WARNING: video resolution {img_resolution} does not match "
              f"calibration resolution {cam_resolution}. Scaling intrinsics.")
        resize_factor = np.array(img_resolution) / np.array(cam_resolution)
        if not np.isclose(resize_factor[0], resize_factor[1], atol=1e-3):
            print(f"WARNING: The resize factor {resize_factor} differs for "
                  f"x and y!")
        cam_matrix[0] *= resize_factor[1]  # x-dimension
        cam_matrix[1] *= resize_factor[0]  # y-dimension
    return cam_matrix, cam_distor, cam_rotation


def get_cam_para_capture_hall(cam_cal_file_path):
    fid = open(cam_cal_file_path)
    cam_file_content = json.load(fid)
    fid.close()
    folder_name = os.path.basename(os.path.dirname(cam_cal_file_path))
    indices = [i.start() for i in re.finditer("_", folder_name)]
    cam_index = int(folder_name[indices[0]+1:indices[1]])  # get the camera id from the file name

    cam_matrix = cam_file_content['Cam_'+str(cam_index).zfill(2)]['intrinsics']
    cam_matrix = np.vstack(cam_matrix)
    cam_matrix = cam_matrix.reshape((3, 3))
    cam_distor = np.zeros((1, 5))  # I did not find the parameters for distortion
    cam_rotation = cam_file_content['Cam_'+str(cam_index).zfill(2)]['extrinsics']
    cam_rotation = np.vstack(cam_rotation)
    cam_rotation = cam_rotation[0:3, 0:3]
    cam_rotation = cam_rotation.reshape((3, 3))
    # cam_translation = content['2020-08-11']['cam'+str(cam_id)]['tvec']  # we do not need translation
    # cam_translation = np.vstack(cam_translation)
    return cam_matrix, cam_distor, cam_rotation