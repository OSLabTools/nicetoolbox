"""
Body Distance feature detector class for the proximity component.
"""

import os
import sys
import numpy as np
import logging
import cv2
from pathlib import Path

top_level_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(top_level_dir))

# internal imports
from detectors.feature_detectors.base_feature import BaseFeature
import detectors.feature_detectors.proximity.utils as pro_utils
import utils.filehandling as fh
import utils.config as cfg

class BodyDistance(BaseFeature):
    """
    The BodyDistance class is a feature detector that computes the proximity component.

    The BodyDistance feature detector calculates the Euclidean distance between keypoints of 
    different individuals in the scene, essentially determining the proximity between individuals 
    from one frame to the next.
    
    Component: proximity

    Attributes:
        components (list): A list containing the name of the component this class is responsible for:
            proximity.
        algorithm (str): The name of the algorithm used to compute the proximity component.
            body_distance: the Euclidean distance between individuals in the scene
        valid_run (bool): A boolean value indicating whether the feature detector can run with the
            given input data.
        predictions_mapping (dict): A dictionary containing the mapping of body parts to their 
            respective indices, used to identify the keypoints for distance calculation.
        camera_names (list): A list containing the names of the cameras used to capture the 
            original input data.
        used_keypoints (list): A list containing the names of the keypoints used for calculating 
            the distance.
        keypoint_index (list): A list containing the indices of the keypoints used for calculating
    """
    components = ['proximity']
    algorithm = 'body_distance'

    def __init__(self, config, io, data):
        """ Initialize Movement class.
        Setup the BodyDistance feature detector and extract gaze component from method detector output.
        
        This method initializes the BodyDistance class by setting up the necessary configurations, 
        input/output handler, and data. It extracts the body_joints component and prepares the 
        used keypoints and keypoint indices given the predictions mapping.

        Args:
            config (dict): The method-specific configurations dictionary.
            io (class): A class instance that handles input and output folders.
            data (class): A class instance that stores all input file locations.
        """
        self.valid_run = True
        if len(data.subjects_descr) == 1:
            logging.warning("Feature detector 'proximity' requires data of more than one persons. Skipping.")
            self.valid_run = False
            return

        super().__init__(config, io, data, requires_out_folder=False)

        # POSE
        joints_component, joints_algorithm = [l for l in config['input_detector_names'] 
                                          if any(['joints' in s for s in l])][0]
        pose_config_folder = io.get_detector_output_folder(joints_component, joints_algorithm, 'run_config')
        pose_config = cfg.load_config(os.path.join(pose_config_folder, 'run_config.toml'))
        self.predictions_mapping = \
            cfg.load_config("./detectors/configs/predictions_mapping.toml")[
                "human_pose"][pose_config["keypoint_mapping"]]
        self.camera_names = pose_config["camera_names"]

        # # will be used during visualizations        
        # viz_camera_name = config['viz_camera_name'].strip('<').strip('>')
        # self.frames_data = os.path.join(pose_config['input_data_folder'], data.camera_mapping[viz_camera_name])
        # self.frames_data_list = [os.path.join(self.frames_data, f) for f in sorted(os.listdir(self.frames_data))]
        self.used_keypoints = config["used_keypoints"]
        # proximity index
        for keypoint in self.used_keypoints:
            if keypoint not in self.predictions_mapping["keypoints_index"]["body"].keys():
                logging.error(f"Given used_keypoint could not find in predictions_mapping {keypoint}")
        self.keypoint_index = \
            [self.predictions_mapping["keypoints_index"]["body"][keypoint]
             for keypoint in self.used_keypoints]

        logging.info(f"Feature detector for component {self.components} initialized.")

    def compute(self):
        """ 
        Computes the proximity component.
        
        This method calculates the Euclidean distance between the keypoints of personL and 
        personR. If the length of the keypoint index list is greater than 1, the midpoint of the
        keypoints will be used in the proximity measure.
        
        The results are saved in a numpy .npz file with the following structure:
        - body_distance_2d: A numpy array containing the proximity scores in 2D.
        - body_distance_3d: A numpy array containing the proximity scores in 3D.
        - data_description: A dictionary containing the data description for the above output numpy arrays. See the documentation of the output for more details.
        
        Returns:
            out_dict (dict): A dictionary containing the proximity scores (2D and/or 3D).
    
        """
        if not self.valid_run:
            return None
        
        dimensions = ['2d'] if len(self.camera_names) < 2 else ['2d', '3d']

        out_dict = {'data_description': {}}
        for dim in dimensions:
            joint_data = np.load(self.input_files[0], allow_pickle=True)
            dim_data = '2d_filtered' if dim == '2d' else dim
            data = joint_data[dim_data]
            data_description = joint_data['data_description'].item()[dim]

            if len(data) != 2:
                logging.error("The number of persons in the video is != 2. "
                            "Proximity can not be calculated. Skipping.")
                return None

            personL, personR = data

            # Calculate the average coordinates for the selected keypoints in both objects for each frame
            average_coords_L = np.mean(personL[:, :, self.keypoint_index, :], axis=2, keepdims=True)
            average_coords_R = np.mean(personR[:, :, self.keypoint_index, :], axis=2, keepdims=True)

            # Calculate the Euclidean distance between the average coordinates for each frame
            proximity_score = np.linalg.norm(average_coords_L - average_coords_R, axis=-1)

            # update results dictionary
            del data_description['axis3'], data_description['axis4']
            out_dict.update({
                f'body_distance_{dim}': np.stack((proximity_score, proximity_score), axis=0),
            })
            out_dict['data_description'].update({
                    f'body_distance_{dim}': dict(**data_description, axis3='distance'),
            })

        # save results
        save_file_path = os.path.join(self.result_folders['proximity'], f"{self.algorithm}.npz")
        np.savez_compressed(save_file_path, **out_dict)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out_dict

    def visualization(self, out_dict):
        """
        Creates visualizations for the computed proximity component.
        
        The visualization includes a line graph of the proximity scores over time, and the 
        proximity scores are also displayed on top of the original video frames. The video is 
        saved as 'proximity_score_on_video.mp4' in the visualization folder.
        
        Args:
            out_dict (dict): A dictionary containing the proximity scores computed by the
            feature detector. It should contain keys 'body_distance_2d' and/or 
            'body_distance_3d', each mapping to a numpy array containing the proximity scores 
            for the respective dimension.
        """
        if out_dict is not None:
            logging.info(f"Visualizing the feature detector output {self.components}.")

            data = {}
            if 'body_distance_2d' in out_dict.keys():
                data['2d'] = out_dict['body_distance_2d']
            if 'body_distance_3d' in out_dict.keys():
                data['3d'] = out_dict['body_distance_3d']

            for dim, body_distance in data.items():

                camera_names = self.camera_names if dim == '2d' else ['3d']
                pro_utils.visualize_proximity_score(body_distance, self.viz_folder, self.used_keypoints, camera_names)
                # # Determine global_min and global_max - define y-lims of graphs
                # global_min = data[0].min() + 0.5
                # global_max = data[0].max() - 0.5
                # # Get a sample image to determine video dimensions
                # sample_frame = cv2.imread(self.frames_data_list[0])
                # sample_combined_img = pro_utils.frame_with_linegraph(sample_frame, data, 0, global_min, global_max)
                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4 format
                # output_path = os.path.join(self.viz_folder, 'proximity_score_on_video.mp4')
                # out = cv2.VideoWriter(output_path, fourcc, 30.0, (sample_combined_img.shape[1], sample_combined_img.shape[0]))
                #
                # for i, frame_path in enumerate(self.frames_data_list):
                #     frame = cv2.imread(frame_path)
                #     if i % 100 == 0:
                #         logging.info(f"Image ind: {i}")
                #     else:
                #         combined = pro_utils.frame_with_linegraph(frame, data, i, global_min, global_max)
                #         out.write(combined)
                # out.release()
            logging.info(f"Visualization of feature detector {self.components} completed.")

    def post_compute(self, distance_data):
        pass


