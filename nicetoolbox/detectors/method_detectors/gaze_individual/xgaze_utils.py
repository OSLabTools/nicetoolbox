import cv2
import numpy as np
import logging


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


def get_cam_para_studio(content, cam, image):
    #cam_matrix = content['2020-08-11'][cam]['intrinsic_matrix']
    cam_matrix = content[cam]['intrinsic_matrix']
    cam_matrix = np.vstack(cam_matrix)
    #cam_distor = content['2020-08-11'][cam]['distortions']
    cam_distor = content[cam]['distortions']
    cam_distor = np.hstack(cam_distor)
    #cam_rotation = content[cam]['rvec']
    #cam_rotation = np.hstack(cam_rotation)
    #cam_rotation = cv2.Rodrigues(cam_rotation)[0]  # change to be rotation matrix
    cam_rotation = np.array(content[cam]['rotation_matrix'])
    # cam_translation = content['2020-08-11']['cam'+str(cam_id)]['tvec']  # we do not need translation
    # cam_translation = np.vstack(cam_translation)
    cam_translation = np.array(content[cam]['translation'])

    #cam_resolution = sorted(content['2020-08-11'][cam]['image_size'])
    cam_resolution = np.array(content[cam]['image_size']).flatten()[::-1]
    img_resolution = np.array(image.shape[:2]).flatten()
    if any(cam_resolution != img_resolution):
        logging.warning(f"Camera {cam}: video resolution {img_resolution} does not match "
              f"calibration resolution {cam_resolution}. Scaling intrinsics.")
        resize_factor = img_resolution / cam_resolution
        if not np.isclose(resize_factor[0], resize_factor[1], atol=1e-3):
            logging.warning(f"The resize factor {resize_factor} differs for "
                  f"x and y!")
        cam_matrix[0] *= resize_factor[1]  # x-dimension
        cam_matrix[1] *= resize_factor[0]  # y-dimension
    return cam_matrix, cam_distor, cam_rotation, cam_translation

