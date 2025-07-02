"""
SPIGA method detector class.
"""

import logging
import os

import cv2
import numpy as np

from ....utils import video as vd
from ..base_detector import BaseDetector


class Spiga(BaseDetector):
    """
    SPIGA is a method detector that computes the head_orientation component.

    Component: head_orientation

    Attributes:
        components (list): A list containing the name of the component: head_orientation
        algorithm (str): Algorithm name used to compute the head_orientation component.
        camera_names (list): List of camera names used to capture original input data.
    """

    components = ["head_orientation"]
    algorithm = "spiga"

    def __init__(self, config, io, data) -> None:
        """
        Initialize the SPIGA method detector with all inference preparation.

        Args:
            config (dict): Configuration settings for SPIGA.
            io (class): IO class instance for input/output operations.
            data (class): Data class instance for frame and subject data.
        """
        logging.info(
            f"Prepare Inference for '{self.algorithm}' and component {self.components}."
        )

        self.frames_list = data.frames_list
        config["frames_list"] = self.frames_list
        config["frame_indices_list"] = data.frame_indices_list
        self.video_start = data.video_start

        # Call the base class constructor
        super().__init__(config, io, data, requires_out_folder=config["visualize"])

        self.camera_names = [n for n in config["camera_names"] if n]
        self.cam_sees_subjects = config["cam_sees_subjects"]
        self.results_folder = config["result_folders"][self.components[0]]

        logging.info("SPIGA Inference Preparation complete.\n")

    def post_inference(self):
        """
        Optional post-processing after SPIGA inference.
        """
        pass

    def visualization(self, data):
        n_subj = len(self.subjects_descr)

        prediction_file = os.path.join(self.results_folder, f"{self.algorithm}.npz")
        predictions = np.load(prediction_file, allow_pickle=True)
        head_data = predictions["head_orientation"]
        data_decr_arr = predictions["data_description"]
        camera_order = data_decr_arr.item()["head_orientation"]["axis1"]

        # per camera and frame, visualize each subject's gaze
        success = True
        for camera_name in camera_order:
            cam_idx = camera_order.index(camera_name)
            os.makedirs(os.path.join(self.viz_folder, camera_name), exist_ok=True)

            for frame_idx in range(head_data.shape[2]):
                # load the original input image
                image_file = [
                    file for file in self.frames_list[frame_idx] if camera_name in file
                ][0]
                image = cv2.imread(image_file)

                colors = [(255, 204, 204), (204, 255, 204)]

                for subject_idx in range(n_subj):
                    if subject_idx not in self.cam_sees_subjects[camera_name]:
                        continue

                    head_orientation = head_data[subject_idx, cam_idx, frame_idx]
                    start = (int(head_orientation[0]), int(head_orientation[1]))
                    end = (int(head_orientation[2]), int(head_orientation[3]))

                    cv2.arrowedLine(
                        image,
                        start,
                        end,
                        colors[subject_idx],
                        thickness=3,
                        tipLength=0.1,
                    )

                cv2.imwrite(
                    os.path.join(
                        self.viz_folder,
                        camera_name,
                        f"{frame_idx + int(self.video_start):05d}.jpg",
                    ),
                    image,
                )

            # create and save video
            success *= vd.frames_to_video(
                os.path.join(self.viz_folder, camera_name),
                os.path.join(self.viz_folder, f"{camera_name}.mp4"),
                fps=data.fps,
                start_frame=int(self.video_start),
            )

        logging.info(
            f"Detector {self.components}: visualization finished with code "
            f"{success}."
        )
