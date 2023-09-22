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
    name = 'proximity'
    behavior = 'interaction'

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

        # proximity index
        for keypoint in config["used_keypoint"]:
            log_ut.assert_and_log(keypoint in [self.predictions_mapping["keypoints_index"]["body"].keys()],
                                  f"Given used_keypoint could not find in predictions_mapping {keypoint}")
        self.keypoint_index = self.predictions_mapping["keypoints_index"]["body"][config["used_keypoint"]]


    def compute(self):
        """ Compute the euclidean distance between the keypoint coord of personL and personR
        index of keypoint(s). If length of index of keypoint(s) list is greater than 1,
        the midpoint of the keypoints will be used in proximity measure

        Returns
         -------
        The proximity measure - The euclidean distance between the two person's body (one location on body)

        """
        data = fh.read_hdf5(self.input_file)
        personL = data[0]
        personR = data[1]

        if len(self.keypoint_index) == 1:
            # Calculate the Euclidean distance for the selected keypoints between object_a and object_b for each frame
            proximity_score = np.linalg.norm(personL[:, self.keypoint_index[0], :] - personR[:, self.keypoint_index[0], :], axis=-1)
        elif len(self.keypoint_index) > 2:
            # Calculate the average coordinates for the selected keypoints in both objects for each frame
            average_coords_L = np.mean(personL[:, self.keypoint_index, :], axis=1)
            average_coords_R = np.mean(personR[:, self.keypoint_index, :], axis=1)

            # Calculate the Euclidean distance between the average coordinates for each frame
            proximity_score = np.linalg.norm(average_coords_L - average_coords_R, axis=-1)

        else:
            logging.error("Compute proximity function - unkonown structure ")
            raise ValueError("Compute proximity function - unkonown structure ")

        # save results
        filepath = os.path.join(self.result_folder, "proximity.hdf5")
        fh.save_to_hdf5(proximity_score, groups_list=["dyad"], output_file=filepath)

        return proximity_score


    def visualization(self, data):
        """

        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        logging.info(f"VISUALIZING the method output {self.name}")




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
        log_ut.assert_and_log(person_data_list[0].shape == person_data_list[1].shape, f"Shape mismatch: Shapes for personL and personR are not the same.")

        #check if any [0,0,0] prediction
        for person_results in person_data_list:
            test_data.check_zeros(person_results) #raise assertion if there is any [0,0,0] inference

        return person_data_list

