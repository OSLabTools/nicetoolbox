import os
import numpy as np
import logging
import cv2
from features.base_feature import BaseFeature
import oslab_utils.filehandling as fh
import oslab_utils.config as cfg
import oslab_utils.logging_utils as log_ut
import features.leaning.utils as lean_utils
import tests.test_data as test_data


class Leaning(BaseFeature):
    """
    """
    name = 'leaning'
    behavior = 'Forward-Backward-lean'

    def __init__(self, config, io):
        """ Initialize Movement class.

        Parameters
        ----------
        config : dict
            the method-specific configurations dictionary
        io: class
            a class instance that handles in-output folders
        """
        # then, call the base class init
        super().__init__(config, io)
        self.predictions_mapping = cfg.load_config("./configs/predictions_mapping.toml")["human_pose"][config["keypoint_mapping"]]
        self.camera_names = config["camera_names"]

        # will be used during visualizations
        self.frames_data = os.path.join(self.input_data_folder,self.camera_names[1]) ##ToDo select camera4 using camera_names[1] hardcoded
        self.frames_data_list = [os.path.join(self.frames_data, f) for f in os.listdir(self.frames_data)]
        self.used_keypoints = config["used_keypoints"]
        # proximity index
        for keypoint in self.used_keypoints:
            log_ut.assert_and_log(keypoint not in [self.predictions_mapping["keypoints_index"]["body"].keys()],
                                  f"Given used_keypoint could not find in predictions_mapping {keypoint}")

        self.keypoint_index = [[self.predictions_mapping["keypoints_index"]["body"][keypoint] for keypoint in keypoint_pair] for keypoint_pair in config["used_keypoints"]]


    def compute(self):
        """ Compute the euclidean distance between the keypoint coord of personL and personR
        index of keypoint(s). If length of index of keypoint(s) list is greater than 1,
        the midpoint of the keypoints will be used in proximity measure

        Returns
         -------
        The proximity measure - The euclidean distance between the two person's body (one location on body)

        """
        data = fh.read_hdf5(self.input_file)
        leaning_data_list = []

        for person_data in data:
            midpoints = []
            # Calculate midpoints of the specified pairs
            for pair in self.keypoint_index:
                kp1 = person_data[:, pair[0], :]
                kp2 = person_data[:, pair[1], :]
                midpoint = (kp1 + kp2) / 2.0
                midpoints.append(np.array(midpoint))

            # Calculate the angles between midpoints
            leaning_data = lean_utils.calculate_angle_btw_three_points(midpoints)
            leaning_gradient = np.gradient(leaning_data)
            merged_data = np.concatenate(leaning_data, leaning_gradient).reshape(-1,2)
            leaning_data_list.append(merged_data)
            print(merged_data)

        filepath = os.path.join(self.result_folder, "leaning.hdf5")
        fh.save_to_hdf5(leaning_data_list, groups_list=["personL", "personR"], output_file=filepath)

        return leaning_data_list


    def visualization(self, data):
        """
        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        logging.info(f"VISUALIZING the feature output {self.name}")
        lean_utils.visualize_lean_in_out_per_person(data, self.viz_folder)
        # Determine global_min and global_max - define y-lims of graphs
        # global_min = data[0].min()
        # global_max = data[0].max()
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


    def post_compute(self, distance_data):
        pass


