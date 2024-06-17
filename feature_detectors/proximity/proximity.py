import os
import numpy as np
import logging
import cv2
from feature_detectors.base_feature import BaseFeature
import oslab_utils.filehandling as fh
import oslab_utils.config as cfg
import feature_detectors.proximity.utils as pro_utils

class Proximity(BaseFeature):
    """
    """
    components = ['proximity']
    algorithm = 'body_distance'

    def __init__(self, config, io, data):
        """ Initialize Movement class.

        Parameters
        ----------
        config : dict
            the method-specific configurations dictionary
        io: class
            a class instance that handles in-output folders
        """
        self.valid_run = True
        if len(data.subjects_descr) == 1:
            logging.warning("Feature detector 'proximity' requires data of more than one persons. Skipping.")
            self.valid_run = False
            return

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

        # # will be used during visualizations        
        # viz_camera_name = config['viz_camera_name'].strip('<').strip('>')
        # self.frames_data = os.path.join(pose_config['input_data_folder'], data.camera_mapping[viz_camera_name])
        # self.frames_data_list = [os.path.join(self.frames_data, f) for f in sorted(os.listdir(self.frames_data))]
        # self.used_keypoints = config["used_keypoints"]
        # proximity index
        for keypoint in self.used_keypoints:
            if keypoint not in self.predictions_mapping["keypoints_index"]["body"].keys():
                logging.error(f"Given used_keypoint could not find in predictions_mapping {keypoint}")
        self.keypoint_index = \
            [self.predictions_mapping["keypoints_index"]["body"][keypoint]
             for keypoint in self.used_keypoints]


        logging.info(f"Feature detector for component {self.components} initialized.")

    def compute(self):
        """ Compute the euclidean distance between the keypoint coord of personL and personR
        index of keypoint(s). If length of index of keypoint(s) list is greater than 1,
        the midpoint of the keypoints will be used in proximity measure

        Returns
         -------
        The proximity measure - The euclidean distance between the two person's body (one location on body)

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

        Parameters
        ----------
        out_dict: class
            a class instance that stores all input file locations
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


