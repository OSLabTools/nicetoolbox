"""
XGaze3cams method detector class.

This code is by XuCong taken from
/ps/project/pis/GazeInterpersonalSynchrony/code_from_XuCong
"""

import logging
import os
from typing import Dict, List

import cv2
import numpy as np

from ....utils import video as vd
from ....utils import visual_utils as vis_ut
from ..base_detector import BaseDetector
from ..filters import SGFilter


class MultiviewEthXgaze(BaseDetector):
    """
    The XGaze3cams class is a method detector that computes the gaze_individual
    component.

    The method detector computes the gaze of individuals in the scene using multiple
    cameras.It provides the necessary preparations and post-inference visualizations to
    integrate the XGaze3cams algorithm into our pipeline.

    Component: gaze_individual

    Attributes:
        components (list): A list containing the name of the component: gaze_individual.
        algorithm (str): The name of the algorithm used to compute the gaze_individual
            component.
        camera_names (list): A list of camera names used to capture the original input
            data.
    """

    components = ["gaze_individual"]
    algorithm = "multiview_eth_xgaze"

    def __init__(self, config, io, data) -> None:
        """
        Initialize the XGaze3cams method detector with all inference preparations
        completed.

        Args:
            config (dict): A dictionary containing the configuration settings for
                the method detector.
            io (class): An instance of the IO class for input-output operations.
            data (class): An instance of the Data class for accessing data.
        """

        logging.info(
            f"Prepare Inference for '{self.algorithm}' and "
            f"components {self.components}."
        )

        self.frames_list = data.frames_list
        config["frames_list"] = self.frames_list
        config["frame_indices_list"] = data.frame_indices_list
        self.video_start = data.video_start

        super().__init__(config, io, data, requires_out_folder=config["visualize"])

        self.camera_names = config["camera_names"]
        while "" in self.camera_names:
            self.camera_names.remove("")
        self.cam_sees_subjects = config["cam_sees_subjects"]

        self.result_folders = config["result_folders"][self.components[0]]
        self.alg_out_folder = config["out_folder"]
        self.filtered = config["filtered"]
        if self.filtered:
            self.filter_window_length = config["window_length"]
            self.filter_polyorder = config["polyorder"]

        self.calibration = config["calibration"]
        logging.info("Inference Preparation completed.\n")

    def post_inference(self):
        """
        Post-processing after inference.

        This method is called after the inference step and is used for any
        post-processing tasks that need to be performed.
        """
        # Filter the 3d results for less flickering estimates
        if self.filtered:
            prediction_file = os.path.join(self.result_folders, f"{self.algorithm}.npz")
            prediction = np.load(prediction_file, allow_pickle=True)
            predictions_dict = {key: prediction[key] for key in prediction.files}
            data_description = predictions_dict["data_description"].item()
            # Apply filter
            logging.info("APPLYING filtering to Gaze Individual data...")
            results_3d_filtered = prediction["3d"].copy()[:, :, :, None]
            filter = SGFilter(self.filter_window_length, self.filter_polyorder)
            results_3d_filtered = filter.apply(results_3d_filtered, is_3d=True)
            data_description.update({"3d_filtered": data_description["3d"]})
            predictions_dict["3d_filtered"] = results_3d_filtered[:, :, :, 0]

            if len(self.camera_names) == 1:
                results_2d = prediction["2d"]
                results_2d_filtered = results_2d.copy()[:, :, :, None]
                results_2d_filtered = filter.apply(results_2d_filtered, is_3d=False)
                data_description.update({"2d_filtered": data_description["2d"]})
                predictions_dict["2d_filtered"] = results_2d_filtered[:, :, :, 0]

            results_3d = predictions_dict["3d_filtered"]

        else:
            results_3d = prediction["3d"]

        assert self.camera_names == data_description["landmarks_2d"]["axis1"]

        # project the 3d results back to all camera's 2d images
        projected_data = self._project_gaze_to_camera_views(results_3d)
        k = "2d_projected_from_3d_filtered" if self.filtered else "2d_projected_from_3d"
        predictions_dict[k] = projected_data
        data_description.update(
            {
                k: dict(
                    axis0=data_description["3d"]["axis0"],
                    axis1=self.camera_names,
                    axis2=data_description["3d"]["axis2"],
                    axis3=["coordinate_u", "coordinate_v"],
                )
            }
        )

        np.savez_compressed(prediction_file, **predictions_dict)

    def _project_gaze_to_camera_views(self, data) -> List[Dict[str, np.ndarray]]:
        """
        Projects the 3D gaze data to the 2D camera views.

        This method takes the 3D gaze data and projects it onto the 2D camera views for
        each algorithm in the algorithm list. It iterates over all cameras and computes
        the projected gaze data for each camera view. The projection is done using the
        camera parameters such as the camera matrix, distortion coefficients, rotation
        vectors, and extrinsic parameters. The method handles the transformation of 3D
        points to 2D points using these camera parameters.

        Returns:
            List[Dict]: The projected gaze data for the camera views.
        """
        n_subjects, _, n_frames, _ = data.shape
        projected_data = np.full(
            (n_subjects, len(self.camera_names), n_frames, 2), np.nan
        )

        # Iterate over all cameras
        for cam_name in self.camera_names:
            cam_idx = self.camera_names.index(cam_name)
            _, _, cam_R, _ = vis_ut.get_cam_para_studio(self.calibration, cam_name)

            image_width = self.calibration[cam_name]["image_size"][0]

            for subject_idx, _subject_name in enumerate(self.subjects_descr):
                if subject_idx in self.cam_sees_subjects[cam_name]:
                    # Extract all frames at once
                    gaze_vectors = data[subject_idx, 0, :, :]
                    dx, dy = vis_ut.reproject_gaze_to_camera_view_vectorized(
                        cam_R, gaze_vectors, image_width
                    )
                    projected_data[subject_idx, cam_idx, :, 0] = dx
                    projected_data[subject_idx, cam_idx, :, 1] = dy

        return projected_data

    def visualization(self, data):  # noqa: ARG002
        """
        Visualizes the processed frames of the xgaze3cams algorithm as a video for all
        cameras.

        This function reads the processed frames from each camera, checks if all
        frames are present, and verifies that the number of frames per camera is
        consistent. It then creates a video for each camera using the processed frames.

        Returns:
            None

        Raises:
            AssertionError: If no frames are found for at least one camera or if the
            number of frames per camera is not consistent.
        """
        n_subj = len(self.subjects_descr)

        prediction_file = os.path.join(self.result_folders, f"{self.algorithm}.npz")
        predictions = np.load(prediction_file, allow_pickle=True)

        gaze_data = (
            predictions["2d_projected_from_3d_filtered"]
            if self.filtered
            else predictions["2d_projected_from_3d"]
        )
        mean_face = np.nanmean(predictions["landmarks_2d"], axis=3)

        # per camera and frame, visualize each subject's gaze
        success = True
        for camera_name in self.camera_names:
            cam_idx = self.camera_names.index(camera_name)
            os.makedirs(os.path.join(self.viz_folder, camera_name), exist_ok=True)

            for frame_idx in range(gaze_data.shape[2]):
                # load the original input image
                image_file = [
                    file for file in self.frames_list[frame_idx] if camera_name in file
                ][0]
                image = cv2.imread(image_file)

                for subject_idx in range(n_subj):
                    if subject_idx not in self.cam_sees_subjects[camera_name]:
                        continue

                    # the predicted gaze vector + the mid point of all face landmarks
                    gaze_vector = -gaze_data[subject_idx, cam_idx, frame_idx]
                    subject_eyes_mid = mean_face[subject_idx, cam_idx, frame_idx]
                    # in case no face was detected, draw the arrow in the middle
                    if (subject_eyes_mid != subject_eyes_mid).any():
                        h, w = image.shape[:2]
                        subject_eyes_mid = np.array(
                            [w / n_subj * (0.5 + subject_idx), h / 2]
                        )
                    gaze_direction = subject_eyes_mid + gaze_vector
                    if (gaze_direction != gaze_direction).any():
                        continue

                    # draw the gaze arrow onto the image
                    image = cv2.arrowedLine(
                        image,
                        np.round(subject_eyes_mid).astype(np.int32),
                        np.round(gaze_direction).astype(np.int32),
                        color=(0, 0, 255),
                        thickness=2,
                        line_type=cv2.LINE_AA,
                        tipLength=0.2,
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
