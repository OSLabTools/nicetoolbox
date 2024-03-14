import os
import numpy as np
import logging
from features.base_feature import BaseFeature
import oslab_utils.filehandling as fh
import oslab_utils.config as cfg
import oslab_utils.logging_utils as log_ut
import oslab_utils.linear_algebra as alg
import features.kinematics.utils as kinematics_utils


class GazeDistance(BaseFeature):
    """
    """
    name = 'gazeDistance'
    behavior = 'look at'

    def __init__(self, config, io, data):
        """ Initialize Movement class.

        Parameters
        ----------
        config : dict
            the method-specific configurations dictionary
        io: class
            a class instance that handles in-output folders
        """
        # then, call the base class init
        super().__init__(config, io, data)
        pose_results_folder = self.get_input(self.input_folders, 'pose')
        pose_config = cfg.load_config(os.path.join(pose_results_folder,
                                                        'run_config.toml'))

        # shows that there are 16 keypointa in total, take the five for the head
        self.face_keypoints = \
            cfg.load_config("./configs/predictions_mapping.toml")[
                "human_pose"][pose_config["keypoint_mapping"]][
                'keypoints_index']['face']['face_landmarks']

        gaze_detector_name = self.get_input(config['input_detector_names'], 'gaze')
        gaze_detector_output_folder = io.get_output_folder(gaze_detector_name, 'output')
        self.pose_detector_file_list = [
            [os.path.join(gaze_detector_output_folder, f)
             for f in os.listdir(gaze_detector_output_folder) if 'cam3' in f],
            [os.path.join(gaze_detector_output_folder, f)
             for f in os.listdir(gaze_detector_output_folder) if 'cam4' in f]
        ]
        logging.info(f"Feature {self.name} initialized.")

    def compute(self):
        """
            Calculate euclidean distance between adjacent frames - how changed from t to t-1 - first frame will be empty
        """
        gaze_data, _ = fh.read_hdf5(self.get_input(self.input_files, "gaze"))
        gaze_p1, gaze_p2 = gaze_data
        keypoints_data, _ = fh.read_hdf5(self.get_input(self.input_files, "pose"))
        keypoints_p1, keypoints_p2 = keypoints_data

        head_p1 = keypoints_p1[:, self.face_keypoints].mean(axis=1)
        head_p2 = keypoints_p2[:, self.face_keypoints].mean(axis=1)

        distance_p1 = alg.distance_line_point(head_p1, gaze_p1, head_p2)
        distance_p2 = alg.distance_line_point(head_p2, gaze_p2, head_p1)

        filepath = os.path.join(self.result_folder, "distance_face.hdf5")
        fh.save_to_hdf5([distance_p1, distance_p2],
                        groups_list=self.subjects_descr,
                        output_file=filepath)

        look_at = self.post_compute(np.stack((distance_p1, distance_p2), axis=0))

        logging.info(f"Computation of feature {self.name} completed.")
        return np.stack((distance_p1, distance_p2), axis=0), *look_at

    def visualization(self, data):
        """

        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        logging.info(f"Visualizing the method output {self.name}."
                     f"This may take longer due to the evolving linegraph video creation.")

        distances, look_at, mutual = data

        # Determine global_min and global_max - define y-lims of graphs
        global_min = min(distances[0].min(), distances[1].min())
        global_max = max(distances[0].max(), distances[1].max())

        # scale binary data
        look_at = look_at * (global_max - global_min) / 2
        look_at += global_min
        mutual = mutual * (global_max - global_min) / 2
        mutual += global_min

        input_data = np.concatenate((distances, look_at, mutual), axis=2)
        categories = ["face", "gaze fixed", "mutual gaze"]
        kinematics_utils.visualize_sum_of_motion_magnitude_by_bodypart(
            input_data, categories, global_min, global_max, self.viz_folder,
            self.subjects_descr)
        kinematics_utils.create_video_evolving_linegraphs(
            self.pose_detector_file_list[0], input_data, categories, global_min,
                global_max, self.viz_folder, 'distance_face_cam3', 3.0)
        kinematics_utils.create_video_evolving_linegraphs(
            self.pose_detector_file_list[1], input_data, categories, global_min,
                global_max, self.viz_folder, 'distance_face_cam4', 3.0)

        logging.info(f"Visualization of feature {self.name} completed.")

    def post_compute(self, distances_face):
        """

        """
        threshold = 0.3
        look_at = distances_face <= threshold
        gazeFixedL, gazeFixedR = look_at
        mutual = np.all(look_at, axis=0)

        filepath = os.path.join(self.result_folder, "look_at.hdf5")
        fh.save_to_hdf5(
            [gazeFixedL, ~gazeFixedL, gazeFixedR, ~gazeFixedR, mutual],
            groups_list=["gazeFixedL", "gazeAvertedL", "gazeFixedR", "gazeAvertedR", "mutual"],
            output_file=filepath)

        return [look_at, np.stack((mutual, mutual), axis=0)]
