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
    The GazeDistance class is a feature detector that computes the gaze_interaction component.

    The GazeDistance feature detector accepts two primary inputs: the gaze_individual and 
    face_landmarks components. These components are computed using the gaze_individual and 
    body_joints method detectors, respectively. This feature detector calculates the smallest 
    distance between a gaze direction vector and face landmarks within a 2-person context. 
    Additionally, it has the ability to determine whether the gaze is directed 
    at the face and if the gaze interaction is mutual.
    
    Component: gaze_interaction
    
    Attributes:
        components (list): A list containing the name of the component this class is responsible for:
            gaze_interaction:
                - distance_gaze , distances from the gaze (of person 1) to the face (of person 2)
                - gaze_look_at  , boolean array indicating whether the gaze is directed at the face
                - gaze_mutual   , boolean array indicating whether the gaze is mutual
        algorithm (str): The name of the algorithm used to compute the components (gaze_interaction).
        gaze_detector_file_list (list): A list of file paths for the gaze detector output.
        threshold_look_at (float): The threshold value for determining whether the gaze is directed
            at the face.
    """
    
    components = ['gaze_interaction']
    algorithm = 'gaze_distance'

    def __init__(self, config, io, data):
        """
        Setup the GazeDistance feature detector and extract gaze component from method detector output.
        
        This method initializes the GazeDistance class by setting up the necessary configurations, 
        input/output handler, and data. It also extracts the gaze component and algorithm from the 
        configuration and prepares the list of gaze detector output files. It supports handling of 
        multiple cameras.

        Args:
            config (dict): The configuration settings for the feature detector. It should include 
                'input_detector_names' key which contains gaze component and algorithm.
            io (class): The input/output handler , including 'get_detector_output_folder' 
                method which returns the output folder for the gaze detector.
            data (class): The data class.

    """
        super().__init__(config, io, data, requires_out_folder=False)

        # Extract gaze component and algorithm from the config
        gaze_component, gaze_algorithm = [l for l in config['input_detector_names'] 
                                          if any(['gaze' in s for s in l])][0]
        gaze_out_folder = io.get_detector_output_folder(gaze_component, gaze_algorithm, 'output')
        self.gaze_detector_file_list = [
            [os.path.join(gaze_out_folder, f) for f in os.listdir(gaze_out_folder) if 'cam3' in f],
            [os.path.join(gaze_out_folder, f) for f in os.listdir(gaze_out_folder) if 'cam4' in f]
        ]

        self.threshold_look_at = config['threshold_look_at']
        logging.info(f"Feature detector for component {self.components} initialized.")

    def compute(self):
        """
        This method computes the gaze_interaction component and saves the results as a 
        compressed .npz file.

        It calculates the Euclidean distance between gaze direction vectors and face landmarks within 
        a 2-person context. The distance is calculated between adjacent frames, measuring the
        change from t to t-1. The first frame will be empty.

        The method also determines whether the gaze is directed at the face (look_at) and if 
        the gaze interaction is mutual.

        The results are saved as a compressed .npz file with the following structure:

        - distance_gaze: smallest distances from the gaze vector (of person A) to the face 
            (of person B), and vice versa.
        - gaze_look_at: a boolean array indicating whether the gaze is directed at the face
        - gaze_mutual: a boolean array indicating whether the gaze is mutual
        - data_description: A dictionary containing the data description for all of the 
            above output numpy arrays. See the documentation of the output for more details.

        Returns:
            visualization_data (list): A list containing the distances from the gaze to the face,
            a boolean array indicating whether the gaze is directed at the face,
            and a boolean array indicating whether the gaze is mutual.
        """
        
        gaze_data = np.load(self.get_input(self.input_files, 'gaze', listdir=False), allow_pickle=True)
        gaze = gaze_data['3d'][:, 0]
        gaze_description = gaze_data['data_description'].item()['3d']
        keypoints_data = np.load(
            self.get_input(self.input_files, 'landmarks', listdir=False), allow_pickle=True)
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
        look_at = distances_face <= self.threshold_look_at
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

        # TODO: Return outdict
        return visualization_data

    def visualization(self, data):
        """
        Creates visualizations for the computed gaze interaction features.

        This method generates line graphs showing the distance between 
        gaze points and face landmarks, binary graphs indicating whether 
        the gaze is directed at the face, and binary graphs indicating 
        whether the gaze is mutual. Additionally, it creates videos of 
        these line graphs evolving over time.

        Args:
            data (tuple): Output data from the compute method containing:
                - the distances of the gaze to the face
                - a boolean array indicating whether the gaze is directed at the face
                - a boolean array indicating whether the gaze is mutual.

        Returns:
            None
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
