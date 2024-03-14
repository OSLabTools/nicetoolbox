import os
import numpy as np
import logging
import cv2
from features.base_feature import BaseFeature
import oslab_utils.filehandling as fh
import oslab_utils.config as cfg
import oslab_utils.logging_utils as log_ut
import features.proximity.utils as pro_utils
import tests.test_data as test_data


class Proximity(BaseFeature):
    """
    """
    name = 'proximity'
    behavior = 'interaction'

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

        # will be used during visualizations
        self.frames_data = os.path.join(pose_config['input_data_folder'], self.camera_names[1]) ##ToDo select camera4 using camera_names[1] hardcoded
        self.frames_data_list = [os.path.join(self.frames_data, f) for f in os.listdir(self.frames_data)]
        self.used_keypoints = config["used_keypoints"]
        # proximity index
        for keypoint in self.used_keypoints:
            log_ut.assert_and_log(keypoint in self.predictions_mapping["keypoints_index"]["body"].keys(),
                                  f"Given used_keypoint could not find in predictions_mapping {keypoint}")
        self.keypoint_index = \
            [self.predictions_mapping["keypoints_index"]["body"][keypoint]
             for keypoint in self.used_keypoints]

        logging.info(f"Feature {self.name} initialized.")

    def compute(self):
        """ Compute the euclidean distance between the keypoint coord of personL and personR
        index of keypoint(s). If length of index of keypoint(s) list is greater than 1,
        the midpoint of the keypoints will be used in proximity measure

        Returns
         -------
        The proximity measure - The euclidean distance between the two person's body (one location on body)

        """
        data, _ = fh.read_hdf5(self.input_files[0])

        if len(data) != 2:
            logging.error("The number of persons in the video is != 2. "
                          "Proximity can not be calculated. Skipping.")
            return None

        personL = data[0]
        personR = data[1]
        proximity_data_list = []

        if len(self.keypoint_index) == 1:
            # Calculate the Euclidean distance for the selected keypoints between object_a and object_b for each frame
            proximity_score = np.linalg.norm(personL[:, self.keypoint_index[0], :] - personR[:, self.keypoint_index[0], :], axis=-1)
            proximity_data_list.append(proximity_score)
        elif len(self.keypoint_index) > 2:
            # Calculate the average coordinates for the selected keypoints in both objects for each frame
            average_coords_L = np.mean(personL[:, self.keypoint_index, :], axis=1)
            average_coords_R = np.mean(personR[:, self.keypoint_index, :], axis=1)

            # Calculate the Euclidean distance between the average coordinates for each frame
            proximity_score = np.linalg.norm(average_coords_L - average_coords_R, axis=-1)
            proximity_data_list.append(proximity_score)

        else:
            logging.error("Compute proximity function - unkonown structure ")
            raise ValueError("Compute proximity function - unkonown structure ")

        # save results
        filepath = os.path.join(self.result_folder, "proximity.hdf5")
        fh.save_to_hdf5(proximity_data_list, groups_list=["dyad"], output_file=filepath)

        logging.info(f"Computation of feature {self.name} completed.")
        return proximity_data_list

    def visualization(self, data):
        """

        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        if data is not None:
            logging.info(f"Visualizing the feature output {self.name}")
            pro_utils.visualize_proximity_score(data, self.viz_folder, self.used_keypoints)
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
            logging.info(f"Visualization of feature {self.name} completed.")

    def post_compute(self, distance_data):
        pass


