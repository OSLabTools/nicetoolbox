"""
Body Distance feature detector class for the proximity component.
"""

import logging
import os

import numpy as np

from ..base_feature import BaseFeature
from . import utils as pro_utils


class BodyDistance(BaseFeature):
    """
    The BodyDistance class is a feature detector that computes the proximity component.

    The BodyDistance feature detector calculates the Euclidean distance between
    keypoints of different individuals in the scene, essentially determining the
    proximity between individuals from one frame to the next.
    """

    components = ["proximity"]
    algorithm_type = "body_distance"

    def _initialize_detector(self) -> None:
        """Initialize Movement class.
        Setup the BodyDistance feature detector and extract gaze component from method
        detector output.

        This method initializes the BodyDistance class by setting up the necessary
        configurations, input/output handler, and data. It extracts the body_joints
        component and prepares the used keypoints and keypoint indices given the
        predictions mapping.
        """
        if len(self.data.subjects_descr) != 2:
            raise ValueError("Feature detector 'proximity' requires data of 2 persons.")

        # 1. Find the body_joints input from input_map (using tuple keys)
        joints_key = None
        for comp, alg in self.input_map:
            if comp == "body_joints":
                joints_key = (comp, alg)
                break
        if joints_key is None:
            raise ValueError("No body_joints input found in input_detector_names")
        joints_component, joints_algorithm = joints_key

        self.input_file = self.get_input_file(joints_component, joints_algorithm)

        # 2. Get upstream detector config to extract keypoint_mapping and camera_names
        upstream_config = self.subsequence_context.get_detector_config(joints_algorithm)
        keypoint_mapping_name = upstream_config.keypoint_mapping  # e.g., "coco_wholebody"
        self.camera_names = upstream_config.camera_names  # Cameras used by pose detector

        # 3. Get predictions_mapping from runtime_config (already loaded) for proximity index
        self.keypoint_mapping = getattr(self.predictions_mapping.human_pose, keypoint_mapping_name)

        self.used_keypoints = self.detector_config.used_keypoints
        keypoints_index = self.keypoint_mapping.keypoints_index.body
        for keypoint in self.used_keypoints:
            if keypoint not in keypoints_index:
                logging.error(f"Given used_keypoint could not be found in predictions_mapping: {keypoint}")

        self.keypoint_index = [keypoints_index[keypoint] for keypoint in self.used_keypoints]

    def compute(self):
        """
        Computes the proximity component.

        This method calculates the Euclidean distance between the keypoints of personL
        and personR. If the length of the keypoint index list is greater than 1, the
        midpoint of the keypoints will be used in the proximity measure.

        The results are saved in a numpy .npz file with the following structure:
        - body_distance_2d: A numpy array containing the proximity scores in 2D.
        - body_distance_3d: A numpy array containing the proximity scores in 3D.
        - data_description: A dictionary containing the data description for the above
            output numpy arrays. See the documentation of the output for more details.

        Returns:
            out_dict (dict): A dictionary containing the proximity scores
            (2D and/or 3D).

        """

        joint_data = np.load(self.input_file, allow_pickle=True)
        dimensions = ["2d"]
        if "3d" in joint_data["data_description"].item():
            dimensions.append("3d")

        out_dict = {"data_description": {}}
        for dim in dimensions:
            dim_data = "2d_filtered" if dim == "2d" else dim
            data = joint_data[dim_data]
            data_description = joint_data["data_description"].item()[dim]

            if len(data) != 2:
                logging.error(
                    "The number of persons in the video is != 2. " "Proximity can not be calculated. Skipping."
                )
                return None

            personL, personR = data

            # Calculate the average coordinates for the selected keypoints in both
            # objects for each frame
            average_coords_L = np.mean(personL[:, :, self.keypoint_index, :], axis=2, keepdims=True)
            average_coords_R = np.mean(personR[:, :, self.keypoint_index, :], axis=2, keepdims=True)

            # Calculate the Euclidean distance between the average coordinates for
            # each frame
            proximity_score = np.linalg.norm(average_coords_L - average_coords_R, axis=-1)

            # update results dictionary
            del data_description["axis3"], data_description["axis4"]
            out_dict.update({f"body_distance_{dim}": np.stack((proximity_score, proximity_score), axis=0)})
            out_dict["data_description"].update({f"body_distance_{dim}": dict(**data_description, axis3="distance")})

        # save results
        save_file_path = os.path.join(self.result_folders["proximity"], f"{self.algorithm_instance}.npz")
        np.savez_compressed(save_file_path, **out_dict)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out_dict

    def visualization(self, out_dict):
        """
        Creates visualizations for the computed proximity component.

        The visualization includes a line graph of the proximity scores over time,
        and the proximity scores are also displayed on top of the original video frames.
        The video is saved as 'proximity_score_on_video.mp4' in the visualization
        folder.

        Args:
            out_dict (dict): A dictionary containing the proximity scores computed by
                the feature detector. It should contain keys 'body_distance_2d' and/or
            'body_distance_3d', each mapping to a numpy array containing the proximity
                scores for the respective dimension.
        """
        if out_dict is not None:
            logging.info(f"Visualizing the feature detector output {self.components}.")

            data = {}
            if "body_distance_2d" in out_dict:
                data["2d"] = out_dict["body_distance_2d"]
            if "body_distance_3d" in out_dict:
                data["3d"] = out_dict["body_distance_3d"]

            for dim, body_distance in data.items():
                camera_names = self.camera_names if dim == "2d" else ["3d"]
                pro_utils.visualize_proximity_score(body_distance, self.viz_folder, self.used_keypoints, camera_names)
                # # Determine global_min and global_max - define y-lims of graphs
                # global_min = data[0].min() + 0.5
                # global_max = data[0].max() - 0.5
                # # Get a sample image to determine video dimensions
                # sample_frame = cv2.imread(self.frames_data_list[0])
                # sample_combined_img = pro_utils.frame_with_linegraph(
                #   sample_frame, data, 0, global_min, global_max)
                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4 format
                # output_path = os.path.join(self.viz_folder,
                #   'proximity_score_on_video.mp4')
                # out = cv2.VideoWriter(output_path, fourcc, 30.0,
                #   (sample_combined_img.shape[1], sample_combined_img.shape[0]))
                #
                # for i, frame_path in enumerate(self.frames_data_list):
                #     frame = cv2.imread(frame_path)
                #     if i % 100 == 0:
                #         logging.info(f"Image ind: {i}")
                #     else:
                #         combined = pro_utils.frame_with_linegraph(
                #   frame, data, i, global_min, global_max)
                #         out.write(combined)
                # out.release()
            logging.info(f"Visualization of feature detector {self.components} completed.")
