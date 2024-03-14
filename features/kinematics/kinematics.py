import os
import numpy as np
import logging
from features.base_feature import BaseFeature
import oslab_utils.filehandling as fh
import oslab_utils.config as cfg
import oslab_utils.logging_utils as log_ut
import features.kinematics.utils as kinematics_utils
import tests.test_data as test_data


class Kinematics(BaseFeature):
    """
    """
    name = 'kinematics'
    behavior = 'motion analysis'

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
        self.predictions_mapping = \
            cfg.load_config("./configs/predictions_mapping.toml")[
                "human_pose"][pose_config["keypoint_mapping"]]
        self.camera_names = pose_config["camera_names"]
        self.bodyparts_list =  list(self.predictions_mapping['bodypart_index'].keys())

        # will be used during visualizations
        self.frames_data = os.path.join(pose_config['input_data_folder'], self.camera_names[1]) ##ToDo select camera4 using camera_names[1] hardcoded
        self.frames_data_list = [os.path.join(self.frames_data, f) for f in os.listdir(self.frames_data)]

        logging.info(f"Feature {self.name} initialized.")

    def compute(self):
        """
            Calculate euclidean distance between adjacent frames - how changed from t to t-1 - first frame will be empty
        """
        data, _ = fh.read_hdf5(self.input_files[0])
        person_data_list_motion = []
        person_data_list_velocity_per_frame = []
        for person in data:
            if len(self.camera_names) < 1: # check if it is got from single camera)
                person = person[:, :, :2]  #if data is 2d get only 2 values from last cell

            # Compute the differences for each keypoint between adjacent frames
            differences = person[1:, :, :] - person[:-1, :, :]  # differences[t] gives the diff btw [t] and [t-1]  # first frame is empty - will create num_of_frames-1

            # # Add a zero-filled frame at the beginning
            # zero_frame = np.zeros((1, differences.shape[1], differences.shape[2]))
            # differences = np.vstack((zero_frame, differences))

            # Compute the Euclidean distance for each keypoint between adjacent frames
            motion_magnitude = np.linalg.norm(differences, axis=-1)
            ## Standardized
            # motion_magnitude_mean = np.mean(motion_magnitude, axis=0)
            # motion_magnitude_std = np.std(motion_magnitude, axis=0)
            # standardized_magnitudes = (motion_magnitude - motion_magnitude_mean) / motion_magnitude_std

            person_data_list_motion.append(motion_magnitude)
            person_data_list_velocity_per_frame.append(differences)

            # save results
        filepath_movement= os.path.join(self.result_folder, "movement.hdf5")
        filepath_velocity_per_frame = os.path.join(self.result_folder, "velocity_per_frame.hdf5")
        fh.save_to_hdf5(person_data_list_motion,
                        groups_list=self.subjects_descr,
                        output_file=filepath_movement)
        fh.save_to_hdf5(person_data_list_velocity_per_frame,
                        groups_list=self.subjects_descr,
                        output_file=filepath_velocity_per_frame)
        #calculate sum of movement per bodypart
        sum_of_motion_per_bodypart = self.post_compute(person_data_list_motion)

        logging.info(f"Computation of feature {self.name} completed.")
        return sum_of_motion_per_bodypart


    def visualization(self, data):
        """
        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        logging.info(f"VISUALIZING the method output {self.name}")
        # Determine global_min and global_max - define y-lims of graphs
        global_min = np.array(data).min() - 0.05
        global_max = np.array(data).max() + 0.05
        kinematics_utils.visualize_sum_of_motion_magnitude_by_bodypart(
            data, self.bodyparts_list, global_min, global_max, self.viz_folder,
            self.subjects_descr)
        # kinematics_utils.create_video_evolving_linegraphs(
        #     self.frames_data_list, data, self.bodyparts_list, global_min, global_max, self.viz_folder)

        logging.info(f"Visualization of feature {self.name} completed.")

    def post_compute(self, distance_data):
        """
            calculate sum of movement per bodypart
        """
        person_data_list = []
        for person_data in distance_data:
            result = np.zeros((person_data.shape[0], len(self.bodyparts_list)))

            for i, indices in enumerate(self.predictions_mapping['bodypart_index'].values()):
                result[:, i] = person_data[:, indices].mean(axis=1)

            person_data_list.append(result)

        if len(person_data_list) == 2:
            log_ut.assert_and_log(
                person_data_list[0].shape == person_data_list[1].shape,
                f"Shape mismatch: Shapes for personL and personR are not the same.")

        #check if any [0,0,0] prediction
        for person_results in person_data_list:
            test_data.check_zeros(person_results) #raise assertion if there is any [0,0,0] inference

        return person_data_list

