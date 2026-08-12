"""
Base integration of the MMPose framework into the NICE toolbox pipeline.
"""

import logging
import os
from abc import abstractmethod
from typing import Dict, List, Tuple

import cv2
import numpy as np

from nicetoolbox_core.video_loaders import ImagePathsByCameraLoader

from ....configs.schemas.detectors_instances_configs import MMPoseAlgorithmConfig
from ....utils import video as vd
from ..base_method import BaseMethod


class BaseMMPose(BaseMethod):
    """
    The BaseMMPose class is a parent method detector for pose estimation using the MMPose
    framework. It holds shared initialization and visualization logic for both 2D and 3D
    subclasses.
    """

    # Inference scripts live under body_joints/ regardless of the produced component
    # (e.g. MotionBERT declares components=["body_joints_local"] but its script is
    # body_joints/mmpose_3d_inference.py).
    inference_package_name = "body_joints"

    def _initialize_detector(self) -> MMPoseAlgorithmConfig.RuntimeConfig:
        """
        Initializes the MMPose class with extra configuration settings.
        """
        # Convenience references for this class are stored.
        self.fps = self.data.fps
        self.video_start = self.data.video_start_frame_index
        self.calibration = self.data.calibration
        self.subjects_descr = self.data.subjects_descr
        self.cam_sees_subjects = self.data.cam_sees_subjects
        self.min_detection_confidence = self.detector_config.min_detection_confidence

        self.camera_names = self.detector_config.camera_names
        self.filtered = self.detector_config.filtered
        if self.filtered:
            self.filter_window_length = self.detector_config.window_length
            self.filter_polyorder = self.detector_config.polyorder

        # EXTRA FIELDS for MMPose are configured.
        keypoint_mapping_name = self.detector_config.keypoint_mapping
        self.keypoint_mapping = getattr(self.predictions_mapping.human_pose, keypoint_mapping_name)
        self.keypoint_indices = self.keypoint_mapping.keypoints_index

        self._kp_indices, self._kp_description = self.get_per_component_keypoint_mapping(self.keypoint_indices)

        self._prediction_folders = self._get_prediction_folders(make_dirs=True)
        self._image_folders = self._get_image_folders(make_dirs=self.detector_config.visualize)

        # BaseMethod _initialize_detector() is called to build runtime + add extra fields.
        base_runtime = super()._initialize_detector()

        # Extended runtime with MMPose-specific fields is returned.
        return MMPoseAlgorithmConfig.RuntimeConfig(
            **base_runtime.model_dump(),
            prediction_folders=self._prediction_folders,
            image_folders=self._image_folders,
            keypoints_indices=self._kp_indices,
            keypoints_description=self._kp_description,
            fps=int(self.fps),
        )

    @abstractmethod
    def get_per_component_keypoint_mapping(
        self, keypoints_indices
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[str]]]:
        """
        This method extracts the keypoint indices and descriptions for each pose
        estimation component.

        It has to be implemented by the derived classes associated to the available
        pose estimation algorithms. (See HRNetw48 and Vitpose classes below)

        Available algorithms are: hrnetw48, vitpose

        Args:
            keypoints_indices (_type_): _description_
        """
        pass

    def _get_skeleton_connections(self, component_name) -> str:
        """
        Get the skeleton connections for the algorithm index from the predictions
        mapping.

        Args:
            alg_idx (int): The index of the algorithm.
            predictions_mapping (Dict): The predictions mapping.

        Returns:
            List[List[str]]: The skeleton connections for the algorithm index.
        """
        return getattr(self.keypoint_mapping.connections, component_name)

    def _get_prediction_folders(self, make_dirs: bool = False) -> Dict[str, str]:
        """
        Generate and return a dictionary of prediction folders for each camera.

        Structure: detector_output/predictions/<camera_name>/
        """
        folders = {}
        base_output_folder = self.io.get_detector_output_folder(self.components[0], self.algorithm_instance, "output")
        for camera in self.camera_names:
            folder = os.path.join(base_output_folder, "predictions", camera)
            if make_dirs:
                os.makedirs(folder, exist_ok=True)
            folders[camera] = folder
        return folders

    def _get_image_folders(self, make_dirs=False):
        """
        Generate and return a dictionary of image folders for each camera.

        These are the 3rd-party MMPose visualization outputs (per-frame images).
        Structure: detector_output/images/<camera_name>/

        Note: Our own visualizations (videos) are saved to the separate
        visualization/ folder via self.viz_folder.
        """
        folders = {}
        base_output_folder = self.io.get_detector_output_folder(self.components[0], self.algorithm_instance, "output")
        for camera in self.camera_names:
            folder = os.path.join(base_output_folder, "images", camera)
            if make_dirs:
                os.makedirs(folder, exist_ok=True)
            folders[camera] = folder
        return folders

    def visualization(self, data):
        """
        Generates a visualization video for each camera from the processed image frames.

        This method takes the processed image frames for each camera and compiles them
        into a video file. It uses the frames_to_video() function from utils.video.py.
        The success of the video creation is tracked, and the method logs the outcome of
        the visualization process for each camera.
        Args:
            data (class): An instance of a class that stores all data related
                information, including the frame rate (`fps`) for the video and the
                starting frame number (`video_start`).

        Note:
            - The method assumes that the processed image frames are named in a
            specific format (`%09d.jpg`), where each frame's name is a zero-padded
            five-digit number representing its sequence in the video.
        """
        logging.info(f"VISUALIZING the method detector output of {self.components} " f"and {self.algorithm_instance}.")

        # Input data loader from nicetoolbox-core shared code is created.
        dataloader = ImagePathsByCameraLoader(config=self.data.get_input_recipes(), expected_cameras=self.camera_names)

        for _component, result_folder in self.result_folders.items():
            viz_dir = os.path.join(result_folder, self.algorithm_instance, "visualization")
            os.makedirs(viz_dir, exist_ok=True)
            prediction_file = os.path.join(result_folder, f"{self.algorithm_instance}.npz")
            prediction = np.load(prediction_file, allow_pickle=True)

            data = self._get_2d_array_for_visualization(prediction)
            algorithm_labels = prediction["data_description"].item()["2d_interpolated"]["axis3"]
            subject_names = prediction["data_description"].item()["2d"]["axis0"]

            # Visualization parameters are set.
            if _component == "body_joints":
                RADIUS = 4
                THICKNESS = 2
            else:
                RADIUS = 1
                THICKNESS = 1
            JOINT_COLOR = (187, 197, 254)
            SKELETON_COLOR = (255, 144, 30)

            # Per camera and frame, each subject's body_joints are visualized.
            success = True
            for camera_name, frames_list in dataloader:
                cam_idx = self.camera_names.index(camera_name)
                os.makedirs(os.path.join(viz_dir, camera_name), exist_ok=True)

                for frame_idx, image_file in enumerate(frames_list):
                    image = cv2.imread(image_file)
                    if image is None:
                        logging.warning("Visualization skipped missing image: %s", image_file)
                        continue
                    image = image.copy()
                    for subject_idx in range(len(subject_names)):
                        # The predicted joints data is selected.
                        subject_2d_points = data[subject_idx, cam_idx, frame_idx][:, :2]

                        # The joints are drawn onto the image.
                        for joint in subject_2d_points:
                            if np.all(np.isfinite(joint)):
                                cv2.circle(
                                    image,
                                    tuple(int(x) for x in joint),
                                    radius=RADIUS,
                                    color=JOINT_COLOR,
                                    thickness=-1,
                                )

                        keypoints_dict = {label: i for i, label in enumerate(algorithm_labels)}

                        connections = self._get_skeleton_connections(_component)
                        start_points, end_points = [], []
                        for connect in connections:
                            for k in range(len(connect) - 1):
                                if (connect[k] in keypoints_dict) & (connect[k + 1] in keypoints_dict):
                                    start = keypoints_dict[connect[k]]
                                    end = keypoints_dict[connect[k + 1]]
                                    start_points.append([subject_2d_points[start]])
                                    end_points.append([subject_2d_points[end]])

                        start_points = np.array(start_points).reshape(-1, 2)
                        end_points = np.array(end_points).reshape(-1, 2)
                        for start, end in zip(start_points, end_points):
                            if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
                                continue
                            pt1 = tuple(map(int, start))
                            pt2 = tuple(map(int, end))
                            cv2.line(
                                image,
                                pt1,
                                pt2,
                                color=SKELETON_COLOR,
                                thickness=THICKNESS,
                            )

                    cv2.imwrite(
                        os.path.join(
                            viz_dir,
                            camera_name,
                            f"{frame_idx + int(self.video_start):09d}.jpg",
                        ),
                        image,
                    )

                # Video is created and saved.
                success *= vd.frames_to_video(
                    os.path.join(viz_dir, camera_name),
                    os.path.join(viz_dir, f"{camera_name}.mp4"),
                    int(self.fps),
                    start_frame=int(self.video_start),
                )

        logging.info(f"Detector {self.components}: visualization finished with code " f"{success}.")

    def _get_2d_array_for_visualization(self, prediction):
        """2D keypoints drawn on exported videos (subclasses may override)."""
        return prediction["2d_interpolated"]
