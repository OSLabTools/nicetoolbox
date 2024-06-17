import os
import numpy as np
import logging
import cv2
from feature_detectors.base_feature import BaseFeature
import oslab_utils.filehandling as fh
import oslab_utils.config as cfg
import feature_detectors.leaning.utils as lean_utils


class Leaning(BaseFeature):
    """
    """
    components = ['leaning']
    algorithm = 'body_angle'

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

        self.predictions_mapping = cfg.load_config("./configs/predictions_mapping.toml")[
                "human_pose"][pose_config["keypoint_mapping"]]
        self.camera_names = pose_config["camera_names"]

        # will be used during visualizations
        # viz_camera_name = config['viz_camera_name'].strip('<').strip('>')
        # self.frames_data = os.path.join(pose_config['input_data_folder'], data.camera_mapping[viz_camera_name])
        # self.frames_data_list = [os.path.join(self.frames_data, f) for f in sorted(os.listdir(self.frames_data))]
        self.used_keypoints = config["used_keypoints"]

        # leaning index
        for pair in self.used_keypoints:
            for keypoint in pair:
                if keypoint not in self.predictions_mapping["keypoints_index"]["body"].keys():
                    logging.error(f"Given used_keypoint could not find in predictions_mapping {keypoint}")

        self.keypoint_index = [[self.predictions_mapping["keypoints_index"]["body"][keypoint] for keypoint in keypoint_pair] for keypoint_pair in self.used_keypoints]

        logging.info(f"Feature detector for component {self.components} initialized.")

    def compute(self):
        """ Compute the euclidean distance between the keypoint coord of personL and personR
        index of keypoint(s). If length of index of keypoint(s) list is greater than 1,
        the midpoint of the keypoints will be used in proximity measure

        Returns
         -------
        The proximity measure - The euclidean distance between the two person's body (one location on body)

        """
        dimensions = ['2d'] if len(self.camera_names) < 2 else ['2d', '3d']

        out_dict = {'data_description': {}}
        for dim in dimensions:
            joint_data = np.load(self.input_files[0], allow_pickle=True)
            dim_data = '2d_filtered' if dim == '2d' else dim
            data = joint_data[dim_data]
            data_description = joint_data['data_description'].item()[dim]

            # Calculate midpoints of the specified pairs
            midpoints = np.empty((*data.shape[:3], len(self.keypoint_index), 3))
            for i, pair in enumerate(self.keypoint_index):
                kp1 = data[:, :, :, pair[0], :]
                kp2 = data[:, :, :, pair[1], :]
                midpoints[:, :, :, i, :] = (kp1 + kp2) / 2.0

            # Calculate the angles between midpoints
            leaning = lean_utils.calculate_angle_btw_three_points(midpoints)
            leaning_gradient = np.gradient(leaning, axis=2)  # gradient wrt. frames/time
            leaning_data = np.concatenate((leaning, leaning_gradient), axis=-1)

            # save results
            del data_description['axis3'], data_description['axis4']
            out_dict.update({f'body_angle_{dim}': leaning_data})
            out_dict['data_description'].update({
                    f'body_angle_{dim}': dict(**data_description, axis3=['angle_deg', 'gradient_angle'])
            })

        save_file_path = os.path.join(self.result_folders['leaning'], f"{self.algorithm}.npz")
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
        if 'body_angle_2d' in out_dict.keys():
            data['2d'] = out_dict['body_angle_2d']
        if 'body_angle_3d' in out_dict.keys():
            data['3d'] = out_dict['body_angle_3d']

        for dim, data in data.items():
            camera_names = self.camera_names if dim == '2d' else ['3d']
            lean_utils.visualize_lean_in_out_per_person(data, self.subjects_descr, self.viz_folder, camera_names)
            # Determine global_min and global_max - define y-lims of graphs
            # global_min = np.nanmin(data[0])
            # global_max = np.nanmax(data[0])
            # num_of_frames = data[0].shape[0]
            #
            # fig, canvas, axL, axR = lean_utils.create_video_canvas(num_of_frames, global_min, global_max)
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4 format
            # output_path = os.path.join(self.viz_folder, 'leaning_angle_on_video.mp4')
            # out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 320+240))
            #
            # for i, frame_path in enumerate(self.frames_data_list):
            #     frame = cv2.resize(cv2.imread(frame_path), (640,320))
            #     if i % 100 == 0:
            #         logging.info(f"Image ind: {i}")
            #     else:
            #         combined = lean_utils.frame_with_linegraph(frame, i, data, fig, canvas, axL, axR)
            #         out.write(combined)
            # out.release()

        logging.info(f"Visualization of feature detector {self.components} completed.")

    def post_compute(self, distance_data):
        pass


