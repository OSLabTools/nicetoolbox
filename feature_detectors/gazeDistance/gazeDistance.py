import os
import numpy as np
import logging
from feature_detectors.base_feature import BaseFeature
import oslab_utils.filehandling as fh
import oslab_utils.config as cfg
import oslab_utils.logging_utils as log_ut
import oslab_utils.linear_algebra as alg
import feature_detectors.kinematics.utils as kinematics_utils


class GazeDistance(BaseFeature):
    """
    """
    components = ['gaze_interaction']
    algorithm = 'gaze_distance'

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
        super().__init__(config, io, data, requires_out_folder=False)

        # GAZE
        gaze_component, gaze_algorithm = [l for l in config['input_detector_names'] 
                                          if any(['gaze' in s for s in l])][0]
        gaze_out_folder = io.get_detector_output_folder(gaze_component, gaze_algorithm, 'output')
        self.gaze_detector_file_list = [
            [os.path.join(gaze_out_folder, f) for f in os.listdir(gaze_out_folder) if 'cam3' in f],
            [os.path.join(gaze_out_folder, f) for f in os.listdir(gaze_out_folder) if 'cam4' in f]
        ]
        logging.info(f"Feature detector for component {self.components} initialized.")

    def compute(self):
        """
            Calculate euclidean distance between adjacent frames - how changed from t to t-1 - first frame will be empty
        """
        gaze_data = np.load(self.get_input(self.input_files, 'gaze', listdir=False), allow_pickle=True)
        gaze = gaze_data['3d'][:, 0]
        gaze_description = gaze_data['data_description'].item()['3d']
        keypoints_data = np.load(self.get_input(self.input_files, 'landmarks', listdir=False), allow_pickle=True)
        keypoints = keypoints_data['3d'][:, 0]
        keypoints_description = keypoints_data['data_description'].item()['3d']

        assert gaze_description['axis0'] == keypoints_description['axis0']
        subject_description = gaze_description['axis0']
        if len(subject_description) < 2:
            return None

        indices = ['face_landmarks' in key for key in keypoints_description['axis3']]
        head = keypoints[:, :, indices].mean(axis=2)

        distance_p1 = alg.distance_line_point(head[0], gaze[0], head[1])
        distance_p2 = alg.distance_line_point(head[1], gaze[1], head[0])
        distances_face = np.stack((distance_p1, distance_p2), axis=0)[:, None]

        # calculate look_at and mutual_gaze
        threshold = 0.2
        look_at = distances_face <= threshold
        mutual = np.all(look_at, axis=0, keepdims=True)
        visualization_data = [distances_face, look_at, mutual]

        # reshape arrays
        def reshape(arr):
            return np.stack((
                np.concatenate((np.zeros_like(arr[0]), arr[0]), axis=-1), 
                np.concatenate((arr[1], np.zeros_like(arr[1])), axis=-1)
                ), axis=0)
        distances_face = reshape(distances_face)
        look_at = reshape(look_at)
        mutual = reshape(np.concatenate((mutual, mutual), axis=0))

        #  save as npz file
        data_description = dict(
            axis0=subject_description,
            axis1=None,
            axis2=gaze_description['axis2']
            )
        out_dict = {
            'distance_gaze': distances_face,
            'gaze_look_at': look_at,
            'gaze_mutual': mutual,
            'data_description': {
                'distance_gaze': dict(
                    **data_description, 
                    axis3=[f'to_face_{subj}' for subj in subject_description]
                    ),
                'gaze_look_at': dict(
                    **data_description, 
                    axis3=[f'look_at_{subj}' for subj in subject_description]
                    ),
                'gaze_mutual': dict(
                    **data_description, 
                    axis3=[f'with_{subj}' for subj in subject_description]
                    )
            }
        }
        filepath = os.path.join(self.result_folders['gaze_interaction'], f"{self.algorithm}.npz")
        np.savez_compressed(filepath, **out_dict)

        logging.info(f"Computation of feature detector for {self.components} completed.")

        return visualization_data

    def visualization(self, data):
        """

        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        if data is not None:
            logging.info(f"Visualizing the feature detector output {self.components}."
                        f"This may take longer due to the evolving linegraph video creation.")

            distances, look_at, mutual = data
            mutual = np.concatenate((mutual, mutual), axis=0)

            # Determine global_min and global_max - define y-lims of graphs
            global_min = min(distances[0].min(), distances[1].min())
            global_max = max(distances[0].max(), distances[1].max())

            # scale binary data
            look_at = look_at * (global_max - global_min) / 2
            look_at += global_min
            mutual = mutual * (global_max - global_min) / 2
            mutual += global_min

            input_data = np.concatenate((distances, look_at, mutual), axis=-1)[:, 0]
            categories = ["distance_gaze_face", "gaze_look_at", "gaze_mutual"]
            kinematics_utils.visualize_mean_of_motion_magnitude_by_bodypart(
                input_data, categories, global_min, global_max, self.viz_folder,
                self.subjects_descr)
            kinematics_utils.create_video_evolving_linegraphs(
                self.gaze_detector_file_list[0], input_data, categories, global_min,
                    global_max, self.viz_folder, 'distance_face_cam3', 3.0)
            kinematics_utils.create_video_evolving_linegraphs(
                self.gaze_detector_file_list[1], input_data, categories, global_min,
                    global_max, self.viz_folder, 'distance_face_cam4', 3.0)

            logging.info(f"Visualization of feature detector {self.components} completed.")
