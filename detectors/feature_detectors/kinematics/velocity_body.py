"""
Velocity Body feature detector class for kinematics of the body.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path

top_level_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(top_level_dir))

# internal imports
from detectors.feature_detectors.base_feature import BaseFeature
import detectors.feature_detectors.kinematics.utils as kinematics_utils

import utils.filehandling as fh
import utils.check_and_exception as check


class VelocityBody(BaseFeature):
    """
    The VelocityBody class is a feature detector that computes the kinematics component.

    The VelocityBody feature detector accepts the human_pose component (body_joints) as its 
    primary input, which is computed using the human_pose method detector. The kinematics 
    component of this feature detector calculates the Euclidean distance between adjacent 
    frames, essentially determining the velocity of body movement from one frame to the next.
    
    Component: kinematics

    Attributes:
        components (list): A list containing the name of the component this class is responsible for:
            kinematics:
        algorithm (str): The name of the algorithm used to compute the kinematics component.
            velocity_body: the velocity of body movement from one frame to the next
        predictions_mapping (dict): A dictionary containing the mapping of body parts to their 
            respective indices.
        camera_names (list): A list containing the names of the cameras used to capture the 
            original input data.
        bodyparts_list (list): A list containing the names of the body parts.
        fps (int): The frames per second of the input data.
    """
    components = ['kinematics']
    algorithm = 'velocity_body'

    def __init__(self, config, io, data):
        """
        Setup input/output folders and data for the Kinematics feature detector.

        This method initializes the Kinematics class by setting up the necessary configurations, 
        input/output handler, and data. It also extracts the body_joints component and algorithm from 
        the configuration and prepares the list of body parts. It supports handling of 
        multiple cameras by loading the camera names from the pose configuration.

        Args:
            config (dict): The configuration settings for the feature detector. It should include 
                           'input_detector_names' key which contains joints component and algorithm.
            io (object): The input/output handler object. It should have a method 
                         'get_detector_output_folder' which returns the output folder for the joints detector.
            data (object): The data object containing the input data. It should have an attribute 'fps' 
                           which represents the frames per second of the input data.

        """
        super().__init__(config, io, data, requires_out_folder=False)

        joints_component, joints_algorithm = [l for l in config['input_detector_names'] 
                                              if any(['joints' in s for s in l])][0]
        pose_config_folder = io.get_detector_output_folder(joints_component, joints_algorithm, 'run_config')
        pose_config = fh.load_config(os.path.join(pose_config_folder, 'run_config.toml'))
        self.predictions_mapping = \
            fh.load_config("./detectors/configs/predictions_mapping.toml")[
                "human_pose"][pose_config["keypoint_mapping"]]
        
        self.camera_names = pose_config["camera_names"]
        self.bodyparts_list =  list(self.predictions_mapping['bodypart_index'].keys())
        self.fps = data.fps

        # # will be used during visualizations
        # viz_camera_name = config['viz_camera_name'].strip('<').strip('>')
        # self.frames_data = os.path.join(pose_config['input_data_folder'], data.camera_mapping[viz_camera_name])
        # self.frames_data_list = [os.path.join(self.frames_data, f) for f in sorted(os.listdir(self.frames_data))]

        logging.info(f"Feature detector {self.components} {self.algorithm} initialized.")

    def compute(self):
        """
        Computes the kinematics component.

        This method calculates the Euclidean distance between adjacent frames for each keypoint. 
        It handles both 2D and 3D data. The method starts by loading the joint data from the input 
        files. It then computes the differences between adjacent frames for each keypoint. A 
        zero-filled frame is added at the beginning to match the original number of frames. 
        Finally, the Euclidean distance for each keypoint between adjacent frames is computed. The 
        computed differences are stored in a dictionary and saved to a compressed .npz file with 
        the following structure:

        - displacement_vector_body_2d: A numpy array containing the computed differences 
            for 2D data.
        - velocity_body_2d: A numpy array containing the computed velocity for 2D data.
        - displacement_vector_body_3d: A numpy array containing the computed differences 
            for 3D data.
        - velocity_body_3d: A numpy array containing the computed velocity for 3D data.
        - data_description: A dictionary containing the data description for all of the 
            above output numpy arrays. See the documentation of the output for more details.        
        
        Returns:
            out_dict (dict): A dictionary containing the above mentioned parts of the 
                kinematics component.
        """
        dimensions = ['2d'] if len(self.camera_names) < 2 else ['2d', '3d']

        out_dict = {'data_description': {}}
        for dim in dimensions:
            joint_data = np.load(self.input_files[0], allow_pickle=True)
            dim_data = '2d_filtered' if dim == '2d' else dim
            data = joint_data[dim_data]
            data_description = joint_data['data_description'].item()[dim]

            # data.shape = (#persons, #cameras, #frames, #joints/keypoints, 3)
            if dim == '2d': # check if it is got from single camera)
                data = data[..., :2]  #if data is 2d get only 2 values from last cell

            # Compute the differences for each keypoint between adjacent frames
            differences = data[:, :, 1:] - data[:, :, :-1]  # differences[t] gives the diff btw [t] and [t-1]  # first frame is empty - will create num_of_frames-1

            # Add a zero-filled frame at the beginning
            zero_frame = np.zeros_like(differences).mean(axis=2, keepdims=True)
            differences = np.concatenate((zero_frame, differences), axis=2)

            # Compute the Euclidean distance for each keypoint between adjacent frames
            motion_magnitude = np.linalg.norm(differences, axis=-1, keepdims=True)
            motion_velocity = motion_magnitude * self.fps
            
            ## Standardized
            # motion_magnitude_mean = np.nanmean(motion_magnitude, axis=0)
            # motion_magnitude_std = np.nanstd(motion_magnitude, axis=0)
            # standardized_magnitudes = (motion_magnitude - motion_magnitude_mean) / motion_magnitude_std

            # save results
            del data_description['axis4']
            out_dict.update({
                f'displacement_vector_body_{dim}': differences,
                f'velocity_body_{dim}': motion_velocity
            })
            out_dict['data_description'].update({
                    f'displacement_vector_body_{dim}': dict(**data_description, axis4=['coordinate_x', 'coordinate_y', 'coordinate_z']),
                    f'velocity_body_{dim}': dict(**data_description, axis4='velocity')
            })

        save_file_path = os.path.join(self.result_folders['kinematics'], f"{self.algorithm}.npz")
        np.savez_compressed(save_file_path, **out_dict)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out_dict

    def visualization(self, out_dict):
        """
        Creates visualizations for the computed kinematics component.

        This method generates visualizations for the computed kinematics component. It checks if 
        the output dictionary contains the keys 'velocity_body_2d' and 'velocity_body_3d'. If these 
        keys are present, their corresponding values are used to create the visualizations.

        The method calculates the sum of movement per body part using the post_compute method. 
        It then determines the global minimum and maximum values to define the y-limits of the graphs. 
        Finally, it calls the visualize_mean_of_motion_magnitude_by_bodypart function from the 
        kinematics_utils module to generate the visualizations.

        Parameters:
            out_dict (dict): The output dictionary containing the computed kinematics component. 
                It should contain the keys 'velocity_body_2d' and/or 'velocity_body_3d'.
        """
        logging.info(f"Visualizing the feature detector output {self.components}.")

        data = {}
        if 'velocity_body_2d' in out_dict.keys():
            data['2d'] = out_dict['velocity_body_2d']
        if 'velocity_body_3d' in out_dict.keys():
            data['3d'] = out_dict['velocity_body_3d']

        for dim, velocity_body in data.items():
            # Calculate sum of movement per bodypart
            motion_per_bodypart = self.post_compute(velocity_body)

            # Determine global_min and global_max - define y-lims of graphs
            global_min = np.nanmin(motion_per_bodypart) - 0.05
            global_max = np.nanmax(motion_per_bodypart) + 0.05
            camera_names = self.camera_names if dim == '2d' else ['3d']
            kinematics_utils.visualize_mean_of_motion_magnitude_by_bodypart(
                motion_per_bodypart, self.bodyparts_list, global_min, global_max, self.viz_folder,
                self.subjects_descr, camera_names)
            # kinematics_utils.create_video_evolving_linegraphs(
            #     self.frames_data_list, motion_per_bodypart, self.bodyparts_list, global_min, global_max, self.viz_folder)

        logging.info(f"Visualization of feature detector {self.components} completed.")

    def post_compute(self, motion_velocity):
        """
        Calculates the sum of movement per body part.

        This method calculates the sum of movement per body part by averaging the motion velocity 
        over the joint indices corresponding to each body part. It then concatenates the results 
        for all body parts.

        The method also checks for any [0,0,0] prediction using the check_zeros function from the 
        check module.

        Parameters:
            motion_velocity (numpy.ndarray): A 5D numpy array with shape 
                (#persons, #cameras, #frames, #joints/keypoints, 1/velocity) 
                representing the motion velocity.

        Returns:
            bodypart_motion (numpy.ndarray): A 4D numpy array with shape 
                (#persons, #cameras, #frames, #bodyparts) representing the 
                sum of movement per body part.
        """
        # motion_velocity.shape = (#persons, #cameras, #frames, #joints/keypoints, 1/velocity)
        bodypart_motion = []
        for joint_indices in self.predictions_mapping['bodypart_index'].values():
            bodypart_motion.append(np.nanmean(motion_velocity[:, :, :, joint_indices, :], axis=-2))

        bodypart_motion = np.concatenate(bodypart_motion, axis=-1)

        #check for any [0,0,0] prediction
        check.check_zeros(bodypart_motion[:, :, 1:])

        return bodypart_motion

