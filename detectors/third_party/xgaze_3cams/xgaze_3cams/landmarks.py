import cv2
import face_alignment
import numpy as np


def get_face_detector(shape_predictor_filename, face_detector_filename):
    # prepare head pose estimation
    # predictor = dlib.shape_predictor(shape_predictor_filename)
    # face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful
    # face_detector = dlib.cnn_face_detection_model_v1(face_detector_filename)
    face_detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')

    return face_detector


def get_landmarks(input_image, face_detector, debug=False):
    """
    return:
    landmarks_subset_ordered : numpy array of shape (n_subjects, 6, 2)
    """
    landmarks_img, scores, _ = face_detector.get_landmarks(input_image, return_landmark_score=True)

    if landmarks_img is None:
        return None, None

    # extract the subset of the landmarks that show ...
    landmarks_subset = [l_img[[36, 39, 42, 45, 48, 54]]
                        for l_img in landmarks_img]
    scores_subset = [s[[36, 39, 42, 45, 48, 54]] for s in scores]

    # order predictions such that the smallest list index corresponds to the
    # person on the very left. Increase indices from left to right
    landmarks_order = np.argsort([l[0, 0] for l in landmarks_subset])
    landmarks_subset_ordered = np.take(landmarks_subset, landmarks_order, axis=0)
    scores_subset_ordered = np.take(scores_subset, landmarks_order, axis=0)

    if debug:
        for lms in landmarks_subset_ordered[0]:
            input_img = cv2.circle(input_image,
                                   (int(lms[0]), int(lms[1])),
                                   radius=10, color=(0, 0, 255), thickness=2)
        cv2.imshow("img_show", input_img)
        cv2.waitKey(0)
    return landmarks_subset_ordered[..., :2], scores_subset_ordered


def get_landmarks_1(input_image, cam_id, shape_predictor_filename,
                    face_detector_filename):
    is_debug = False
    face_detector = get_face_detector(shape_predictor_filename,
                                      face_detector_filename)

    landmarks_img = face_detector.get_landmarks(input_image)
    if cam_id == 1:
        landmarks = landmarks_img[0]
        landmarks_sub = landmarks[[36, 39, 42, 45, 48, 54], 0:2]
        sub_1_landmark = landmarks_sub
        sub_2_landmark = None
    elif cam_id == 2:
        landmarks = landmarks_img[0]
        landmarks_sub = landmarks[[36, 39, 42, 45, 48, 54], 0:2]
        sub_1_landmark = None
        sub_2_landmark = landmarks_sub
    elif cam_id == 3 or cam_id == 4:
        if len(landmarks_img) != 2:
            print(f"WARNING: Face detector does not detect 2 subjects for "
                  f"camera {cam_id}. Skipping.")
            sub_1_landmark = None
            sub_2_landmark = None
        elif landmarks_img[0][0, 0] < landmarks_img[1][0, 0]:  #  subject 1 is the one on the left
            landmarks = landmarks_img[0]
            landmarks_sub = landmarks[[36, 39, 42, 45, 48, 54], 0:2]
            sub_1_landmark = landmarks_sub
            landmarks = landmarks_img[1]
            landmarks_sub = landmarks[[36, 39, 42, 45, 48, 54], 0:2]
            sub_2_landmark = landmarks_sub
        else:
            landmarks = landmarks_img[1]
            landmarks_sub = landmarks[[36, 39, 42, 45, 48, 54], 0:2]
            sub_1_landmark = landmarks_sub
            landmarks = landmarks_img[0]
            landmarks_sub = landmarks[[36, 39, 42, 45, 48, 54], 0:2]
            sub_2_landmark = landmarks_sub
    else:
        print('The camera id is wrong, it must be from 1 to 4')

    if is_debug:
        for num_dd in range(0, landmarks_sub.shape[0]):
            input_img = cv2.circle(input_image, (
                int(landmarks_sub[num_dd, 0]), int(landmarks_sub[num_dd, 1])), radius=10,
                                   color=(0, 0, 255), thickness=2)
        cv2.imshow("img_show", input_img)
        cv2.waitKey(0)
    return sub_1_landmark, sub_2_landmark


def get_landmarks_2(image, resize_factor):
    face_detector = get_face_detector()

    landmarks_img = face_detector.get_landmarks(image)
    landmarks_img = landmarks_img[0]  # we assume there is only one person
    landmarks_sub = landmarks_img[[36, 39, 42, 45, 48, 54], 0:2] / resize_factor
    return landmarks_sub

