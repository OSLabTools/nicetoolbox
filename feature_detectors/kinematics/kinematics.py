import os
import numpy as np
import logging
from feature_detectors.base_feature import BaseFeature
import oslab_utils.filehandling as fh
import oslab_utils.config as cfg
import feature_detectors.kinematics.utils as kinematics_utils
import oslab_utils.check_and_exception as check


class Kinematics(BaseFeature):
    """
    """
    components = ['kinematics']
    algorithm = 'velocity_body'

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

        # POSE
        joints_component, joints_algorithm = [l for l in config['input_detector_names'] 
                                          if any(['joints' in s for s in l])][0]
        pose_config_folder = io.get_detector_output_folder(joints_component, joints_algorithm, 'run_config')
        pose_config = cfg.load_config(os.path.join(pose_config_folder, 'run_config.toml'))
        self.predictions_mapping = \
            cfg.load_config("./configs/predictions_mapping.toml")[
                "human_pose"][pose_config["keypoint_mapping"]]
        
        self.camera_names = pose_config["camera_names"]
        self.bodyparts_list =  list(self.predictions_mapping['bodypart_index'].keys())
        self.fps = data.fps

        # will be used during visualizations
        self.frames_data = os.path.join(pose_config['input_data_folder'], self.camera_names[1]) ##ToDo select camera4 using camera_names[1] hardcoded
        self.frames_data_list = [os.path.join(self.frames_data, f) for f in sorted(os.listdir(self.frames_data))]

        logging.info(f"Feature detector {self.components} {self.algorithm} initialized.")

    def compute(self):
        """
            Calculate euclidean distance between adjacent frames - how changed from t to t-1 - first frame will be empty
        """
        joint_data = np.load(self.input_files[0], allow_pickle=True)
        data = joint_data['3d'][:, 0]
        data_description = joint_data['data_description'].item()['3d']

        person_data_list_displacement_vector = []
        person_data_list_velocity = []
        for person in data:
            if len(self.camera_names) < 1: # check if it is got from single camera)
                person = person[:, :, :2]  #if data is 2d get only 2 values from last cell

            # Compute the differences for each keypoint between adjacent frames
            differences = person[1:, :, :] - person[:-1, :, :]  # differences[t] gives the diff btw [t] and [t-1]  # first frame is empty - will create num_of_frames-1

            # Add a zero-filled frame at the beginning
            zero_frame = np.zeros((1, differences.shape[1], differences.shape[2]))
            differences = np.vstack((zero_frame, differences))

            # Compute the Euclidean distance for each keypoint between adjacent frames
            motion_magnitude = np.linalg.norm(differences, axis=-1, keepdims=True)

            motion_velocity = motion_magnitude * self.fps
            
            ## Standardized
            # motion_magnitude_mean = np.nanmean(motion_magnitude, axis=0)
            # motion_magnitude_std = np.nanstd(motion_magnitude, axis=0)
            # standardized_magnitudes = (motion_magnitude - motion_magnitude_mean) / motion_magnitude_std

            person_data_list_displacement_vector.append(differences)
            person_data_list_velocity.append(motion_velocity)

        # save results
        data_description = dict(
            axis0=self.subjects_descr,
            axis1=None,
            axis2=data_description['axis2'],
            axis3=data_description['axis3']
            )
        out_dict = {
            'displacement_vector_body': np.stack(person_data_list_displacement_vector, axis=0)[:, None],
            'velocity_body': np.stack(person_data_list_velocity, axis=0)[:, None],
            'data_description': {
                'displacement_vector_body': dict(**data_description, axis4=['coordinate_x', 'coordinate_y', 'coordinate_z']),
                'velocity_body': dict(**data_description, axis4='velocity')
            }                
        }
        save_file_path = os.path.join(self.result_folders['kinematics'], f"{self.algorithm}.npz")
        np.savez_compressed(save_file_path, **out_dict)

        #calculate sum of movement per bodypart
        sum_of_motion_per_bodypart = self.post_compute(person_data_list_velocity)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return sum_of_motion_per_bodypart


    def visualization(self, data):
        """
        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        logging.info(f"Visualizing the feature detector output {self.components}.")
        # Determine global_min and global_max - define y-lims of graphs
        global_min = np.nanmin(np.array(data)) - 0.05
        global_max = np.nanmax(np.array(data)) + 0.05
        kinematics_utils.visualize_sum_of_motion_magnitude_by_bodypart(
            data, self.bodyparts_list, global_min, global_max, self.viz_folder,
            self.subjects_descr)
        # kinematics_utils.create_video_evolving_linegraphs(
        #     self.frames_data_list, data, self.bodyparts_list, global_min, global_max, self.viz_folder)

        logging.info(f"Visualization of feature detector {self.components} completed.")

    def post_compute(self, distance_data):
        """
            calculate sum of movement per bodypart
        """
        person_data_list = []
        for person_data in distance_data:
            result = np.zeros((person_data.shape[0], len(self.bodyparts_list)))

            for i, indices in enumerate(self.predictions_mapping['bodypart_index'].values()):
                result[:, i] = np.nanmean(person_data[:, indices, 0], axis=1)

            person_data_list.append(result)

        if len(person_data_list) == 2:
            if person_data_list[0].shape != person_data_list[1].shape:
                logging.error(f"Shape mismatch: Shapes for personL and personR are not the same.")

        #check if any [0,0,0] prediction
        for person_results in person_data_list:
            check.check_zeros(person_results[1:]) #raise assertion if there is any [0,0,0] inference

        return person_data_list

