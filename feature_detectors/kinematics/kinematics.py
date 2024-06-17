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
        super().__init__(config, io, data, requires_out_folder=False)

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

        # # will be used during visualizations
        # viz_camera_name = config['viz_camera_name'].strip('<').strip('>')
        # self.frames_data = os.path.join(pose_config['input_data_folder'], data.camera_mapping[viz_camera_name])
        # self.frames_data_list = [os.path.join(self.frames_data, f) for f in sorted(os.listdir(self.frames_data))]

        logging.info(f"Feature detector {self.components} {self.algorithm} initialized.")

    def compute(self):
        """
            Calculate euclidean distance between adjacent frames - how changed from t to t-1 - first frame will be empty
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
        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        logging.info(f"Visualizing the feature detector output {self.components}.")

        data = {}
        if 'velocity_body_2d' in out_dict.keys():
            data['2d'] = out_dict['velocity_body_2d']
        if 'velocity_body_3d' in out_dict.keys():
            data['3d'] = out_dict['velocity_body_3d']

        for dim, velocity_body in data.items():
            #calculate sum of movement per bodypart
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
            calculate sum of movement per bodypart
        """
        # motion_velocity.shape = (#persons, #cameras, #frames, #joints/keypoints, 1/velocity)
        bodypart_motion = []
        for joint_indices in self.predictions_mapping['bodypart_index'].values():
            bodypart_motion.append(np.nanmean(motion_velocity[:, :, :, joint_indices, :], axis=-2))

        bodypart_motion = np.concatenate(bodypart_motion, axis=-1)

        #check for any [0,0,0] prediction
        check.check_zeros(bodypart_motion[:, :, 1:])

        return bodypart_motion

