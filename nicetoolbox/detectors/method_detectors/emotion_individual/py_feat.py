"""
Py-feat method detector class.
"""

import logging
import os

import cv2
import numpy as np

from ....utils import video as vd
from ..base_detector import BaseDetector


class PyFeat(BaseDetector):
    """
    The Python - Facial Expression Analysis Toolbox (Py-feat) is a method
    detector that computes emotion_individual component

    Component emotion_individual

    Attributes:
    components (list): A list containing the name of the component: emotion_individual
    algorithm (st): Algorithm name used to compute the emotion_individual component.
    camera_names (list): A list of camera names used to capture original input data.
    """

    components = ["emotion_individual"]
    algorithm = "py_feat"

    def __init__(self, config, io, data) -> None:
        """
        Initialize the PyFeat method detector with all inference preparation.

        Args:
            config (dict): A dictionary containing the configuration settings for
            the method detector.
            io (class): An instance of the IO class for input-output operations.
            data (class): An instance of the Data class for accessing data.
        """

        logging.info(
            f"Prepare Inference for '{self.algorithm}' and component {self.components}."
        )

        self.frames_list = data.frames_list
        config["frames_list"] = self.frames_list
        config["frame_indices_list"] = data.frame_indices_list
        self.video_start = data.video_start

        # call the base class __init__
        super().__init__(config, io, data, requires_out_folder=config["visualize"])

        self.camera_names = [n for n in config["camera_names"] if n != ""]
        while "" in self.camera_names:
            self.camera_names.remove("")
        self.cam_sees_subjects = config["cam_sees_subjects"]

        self.results_folder = config["result_folders"][self.components[0]]

        logging.info("Inference Preparation complete.\n")

    def post_inference(self):
        """
        Post processing after inference.
        """
        pass

    def visualization(self, data):
        """
        Visualizes the processed frames of the pyfeat algorithm as a video.

        Returns:
            None

        Raises:
            AssertionError: If no frames are found for at least one camera or if
            the number of frames per camera is not consistent.
        """
        n_subj = len(self.subjects_descr)

        prediction_file = os.path.join(self.results_folder, f"{self.algorithm}.npz")
        predictions = np.load(prediction_file, allow_pickle=True)

        algorithm_labels = predictions["data_description"].item()["emotions"]["axis3"]
        # format:xywh
        facebbox_data = predictions["faceboxes"]
        emotion_data = predictions["emotions"]
        emotion_colors = [
            [255, 0, 0],
            [85, 170, 47],
            [72, 61, 139],
            [255, 215, 0],
            [70, 130, 180],
            [255, 140, 0],
            [128, 128, 128],
        ]

        # per camera and frame, visualize each subject's emotion
        success = True
        for camera_name in self.camera_names:
            cam_idx = self.camera_names.index(camera_name)
            os.makedirs(os.path.join(self.viz_folder, camera_name), exist_ok=True)

            for frame_idx in range(emotion_data.shape[2]):
                # load the original input image
                image_file = [
                    file for file in self.frames_list[frame_idx] if camera_name in file
                ][0]
                image = cv2.imread(image_file)

                for subject_idx in range(n_subj):
                    if subject_idx not in self.cam_sees_subjects[camera_name]:
                        continue

                    # Draw the bounding box
                    sbj_facebbox = facebbox_data[subject_idx, cam_idx, frame_idx]
                    subject_emotion_probability = emotion_data[
                        subject_idx, cam_idx, frame_idx
                    ]
                    max_probability_idx = np.argmax(subject_emotion_probability)
                    x = int(sbj_facebbox[0])
                    y = int(sbj_facebbox[1])
                    width = int(sbj_facebbox[2])
                    height = int(sbj_facebbox[3])

                    cv2.rectangle(
                        image,
                        (x, y),
                        (x + width, y + height),
                        tuple(emotion_colors[max_probability_idx]),
                        2,
                    )

                    label = algorithm_labels[max_probability_idx]

                    _, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                    # Put label text
                    cv2.putText(
                        image,
                        label,
                        (x, y - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        tuple(emotion_colors[max_probability_idx]),
                        2,
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
