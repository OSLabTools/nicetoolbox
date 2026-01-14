"""
Py-feat method detector class.
"""

import logging
import os

import cv2
import numpy as np

from nicetoolbox_core.dataloader import ImagePathsByFrameIndexLoader

from ....utils import video as vd
from ..base_detector import BaseDetector


class PyFeat(BaseDetector):
    """
    The Python - Facial Expression Analysis Toolbox (Py-feat) is a method
    detector that computes emotion_individual component.
    """

    components = ["emotion_individual"]
    algorithm = "py_feat"

    def __init__(self, config, io, data) -> None:
        """
        Initialize the PyFeat method detector with all inference preparation.
        """

        logging.info(f"Prepare Inference for '{self.algorithm}' and component {self.components}.")

        # Base class setup
        super().__init__(config, io, data, requires_out_folder=config["visualize"])

        self.video_start = data.video_start
        self.camera_names = [n for n in config["camera_names"] if n != ""]
        self.cam_sees_subjects = config["cam_sees_subjects"]
        self.results_folder = config["result_folders"][self.components[0]]

        # Initialize standardized data loader
        self.dataloader = ImagePathsByFrameIndexLoader(config=config, expected_cameras=self.camera_names)

        logging.info("Inference Preparation complete.\n")

    def post_inference(self):
        """
        Post processing after inference.
        """
        pass

    def visualization(self, data):
        """
        Visualizes the processed frames of the pyfeat algorithm as a video.
        """
        n_subj = len(self.subjects_descr)

        prediction_file = os.path.join(self.results_folder, f"{self.algorithm}.npz")
        predictions = np.load(prediction_file, allow_pickle=True)

        algorithm_labels = predictions["data_description"].item()["emotions"]["axis3"]
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

        success = True
        for cam_idx, camera_name in enumerate(self.camera_names):
            os.makedirs(os.path.join(self.viz_folder, camera_name), exist_ok=True)

            # Synchronized Indexing via dataloader
            for frame_idx, (real_frame_idx, frame_paths_per_camera) in enumerate(self.dataloader):
                image_file = frame_paths_per_camera[camera_name]
                image = cv2.imread(image_file)

                if image is None:
                    continue

                for subject_idx in range(n_subj):
                    if subject_idx not in self.cam_sees_subjects[camera_name]:
                        continue

                    sbj_facebbox = facebbox_data[subject_idx, cam_idx, frame_idx]
                    subject_emotion_probability = emotion_data[subject_idx, cam_idx, frame_idx]
                    max_probability_idx = np.argmax(subject_emotion_probability)

                    x, y, width, height = sbj_facebbox[:4].astype(int)

                    cv2.rectangle(
                        image,
                        (x, y),
                        (x + width, y + height),
                        tuple(emotion_colors[max_probability_idx]),
                        2,
                    )

                    label = algorithm_labels[max_probability_idx]
                    _, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                    cv2.putText(
                        image,
                        label,
                        (x, y - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        tuple(emotion_colors[max_probability_idx]),
                        2,
                    )

                # Save using real_frame_idx to keep video timeline sync
                cv2.imwrite(
                    os.path.join(self.viz_folder, camera_name, f"{real_frame_idx:09d}.jpg"),
                    image,
                )

            success *= vd.frames_to_video(
                os.path.join(self.viz_folder, camera_name),
                os.path.join(self.viz_folder, f"{camera_name}.mp4"),
                fps=data.fps,
                start_frame=int(self.video_start),
            )

        logging.info(f"Detector {self.components}: visualization finished with code {success}.")
