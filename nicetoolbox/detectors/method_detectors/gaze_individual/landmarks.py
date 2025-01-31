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
