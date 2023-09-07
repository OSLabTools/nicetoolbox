"""
    This code is by XuCong
    taken from /ps/project/pis/GazeInterpersonalSynchrony/code_from_XuCong
"""

import os
import cv2
import numpy as np
import json
from glob import glob

from detectors.gaze.xgaze_3cams.gaze_estimator import GazeEstimator
from detectors.gaze.xgaze_3cams.utils import vector_to_pitchyaw, draw_gaze, \
    get_cam_para_studio, get_cam_para_capture_hall
from detectors.gaze.xgaze_3cams.landmarks import get_landmarks_1, get_landmarks_2

from detectors.base_detector import BaseDetector


class XGaze3cams(BaseDetector):
    """Class to setup and run existing computer vision research code.
    """
    name = 'XGaze-3cameras'
    behavior = 'gaze'

    def __init__(self, config) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        config : dict
            some configurations/settings dictionary, here it must contain the key 'image_file'
        """
        super().__init__(config)
        self.config = config['xgaze_3cam']

    def inference(self, data):
        """Run inference of the method on the pre-loaded image

        Returns
        -------
        dict
            dict(method_name:str, values:list)
            the values list contains entries of the form
                dict(feature:str, start:int, end:int, label:str)
        """
        # check that the data was created
        assert data.frames_list is not None, \
            f"{self.name}: Please specify 'input_data_format: frames' in " \
            f"config. Currently it is {self.config['input_data_format']}."

        label_names = dict(AatB='AlookatB',
                           BatA='BlookatA',
                           mutual='mutual_gaze',
                           averted='gaze_averted')

        detections = dict(name=self.name, values=[])

        gaze_estimator = GazeEstimator(
                self.config['face_model_filename'],
                self.config['pretrained_model_filename'])

        ## read images
        for frame_files, frame_id in zip(data.frames_list, data.frame_indices_list):
            images = []  # all images for that frame
            file_paths = []  # all image file path for that frame
            sub_1_landmarks = []  # all landmarks for that frame
            sub_2_landmarks = []  # all landmarks for that frame
            # get the landmarks
            for frame_file, cam_id in zip(frame_files, data.all_camera_ids):
                image = cv2.imread(frame_file)
                images.append(image)

                file_paths.append(frame_file)
                landmark_1, landmark_2 = get_landmarks_1(
                        image,
                        cam_id,
                        self.config['shape_predictor_filename'],
                        self.config['face_detector_filename']
                )
                sub_1_landmarks.append(landmark_1)
                sub_2_landmarks.append(landmark_2)

            # OK, now let's do gaze estimation
            for sub_id in range(1, 3):  # loop it for each subject
                gaze_world = []  # the final gaze direction in the world coordinate system
                if sub_id == 1:
                    landmarks = sub_1_landmarks
                else:
                    landmarks = sub_2_landmarks
                for cam_id in data.all_camera_ids:
                    if landmarks[cam_id - 1] is None:
                        continue
                    image = images[cam_id - 1]
                    cam_matrix, cam_distor, cam_rotation = get_cam_para_studio(
                        data.calibration, cam_id, image)
                    pred_gaze = gaze_estimator.gaze_estimation(
                            image,
                            landmarks[cam_id - 1],
                            cam_matrix,
                            cam_distor
                    )

                    # convert the gaze direction to cam4, the world coordinate system
                    pred_gaze_cam4 = np.dot(np.linalg.inv(cam_rotation),
                                            pred_gaze)  # to the cam4 coordinate system
                    pred_gaze_cam4 = pred_gaze_cam4.reshape((1, 3))
                    pred_gaze_cam4 = pred_gaze_cam4 / np.linalg.norm(
                        pred_gaze_cam4)
                    gaze_world.append(pred_gaze_cam4.reshape((1, 3)))
                # get the final gaze direction by averaging them
                gaze_world = np.asarray(gaze_world).reshape((-1, 3))
                gaze_world = np.mean(gaze_world, axis=0).reshape((-1, 3))
                if sub_id == 1:
                    gaze_world_sub_1 = gaze_world
                else:
                    gaze_world_sub_2 = gaze_world

            # visualization on each image, project the gaze back to each cam to draw the gaze direction
            is_debug = False
            for cam_id in data.all_camera_ids:
                image = images[cam_id - 1]
                cam_matrix, cam_distor, cam_rotation = get_cam_para_studio(
                    data.calibration, cam_id, image)
                for sub_id in range(1, 3):
                    if sub_id == 1:
                        landmarks = sub_1_landmarks
                        gaze_world = gaze_world_sub_1
                    else:
                        landmarks = sub_2_landmarks
                        gaze_world = gaze_world_sub_2
                    if landmarks[cam_id - 1] is None:
                        continue

                    # convert the gaze to current camera coordinate system
                    gaze_cam = np.dot(cam_rotation, gaze_world.T)
                    draw_gaze_dir = vector_to_pitchyaw(gaze_cam).reshape(-1)
                    face_center = np.mean(landmarks[cam_id - 1], axis=0)
                    draw_gaze(image,
                              draw_gaze_dir,
                              thickness=2,
                              color=(0, 0, 255),
                              position=face_center.astype(int)
                              )
                if is_debug:
                    cv2.imshow("img_show", image)
                    cv2.waitKey(0)

                # save the image with gaze direction
                _, file_name = os.path.split(file_paths[cam_id - 1])
                save_file_name = os.path.join(
                        self.out_folder, f'cam{cam_id}_{file_name}')
                cv2.imwrite(save_file_name, image)

        return gaze_world_sub_1, gaze_world_sub_2
