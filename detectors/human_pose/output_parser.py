import os
import numpy as np
from abc import ABC, abstractmethod
import third_party.file_handling as fh


KEYPOINT_MAPPING_PATH = "./configs/predictions_mapping.yml"

class BaseParser(ABC):
    """Class to parse algorithm output -- Used in tests.poseprocessing
    """

    def __init__(self, keypoint_type = ""):
        """InitializeMethod class.

        Parameters
        ----------
        keypoint_type : str
            the key in keypoint_mapping.yml
        """
        self.keypoint_type = keypoint_type

    def get_keypoint_index(self, bodypart, keypoint_name):
        """Returns index for the given keypoint

        Parameters
        ----------
        bodypart: bodypart for the given keypoint: body, foot, face,or hand
        keypoint_name: the name of the keypoint or keypoint group

        Returns
        -------
        The index of the keypoint/keypoint group.

        """
        keypoint_mapping_config = fh.load_config(KEYPOINT_MAPPING_PATH)
        return keypoint_mapping_config["human_pose"][self.keypoint_type][bodypart][keypoint_name]

    def get_number_of_keypoints(self, input_file):
        pass

    def get_keypoint_location(self, input_file, keypoint_index, person, person_threshold = ""):
        """Returns the location of asked keypoint in the frame

        Parameters
        ----------
        frame_name: filename or frame index
        keypoint_index: the index of given keypoint
        person: if multi-person either personL or personR, is single person - person

        Returns
        -------
        x, y, and [z] coordinates of the keypoint as a numpy array
        """
        pass


class MMPoseParser(BaseParser):
    """Class to parse MMPose original output
    """

    def __init__(self, keypoint_type):
        super().__init__(keypoint_type)

    def get_keypoint_index(self, keypoint_name):
        pass

    def get_keypoint_location(self, input_file, keypoint_index, person, person_threshold):
        json_predictions = fh.load_json_file(input_file)
        xy_coord = None
        if person == "personL":
            if json_predictions[0]["bbox"][0][0] < person_threshold:
                xy_coord = json_predictions[0]["keypoints"][keypoint_index]
            elif json_predictions[1]["bbox"][0][0] < person_threshold:
                xy_coord = json_predictions[1]["keypoints"][keypoint_index]
            else:
                raise ValueError("PersonL could not find in the output file")
        elif person == "personR":
            if json_predictions[0]["bbox"][0][0] > person_threshold:
                xy_coord = json_predictions[0]["keypoints"][keypoint_index]
            elif json_predictions[1]["bbox"][0][0] > person_threshold:
                xy_coord = json_predictions[1]["keypoints"][keypoint_index]
            else:
                raise ValueError("PersonR could not find in the output file")
        else:
            raise ValueError(f"Unknown person identifier: {person}")
        if xy_coord is None:
            raise ValueError(f"Could not determine xy_coord for {person}")
        return np.array(xy_coord)

    def get_number_of_keypoints(self, input_file):
        json_predictions = fh.load_json_file(input_file)
        return len(json_predictions[0]["keypoints"])

class HDF5Parser(BaseParser):
    """Class to parse saved output
    """

    def __init__(self, keypoint_type):
        super().__init__(keypoint_type)

    def get_keypoint_index(self, keypoint_name):
        pass

    def get_keypoint_location(self, input_file, frame_index, keypoint_index, person, xyz = False):
        hdf5_data, _ = fh.read_hdf5(input_file)
        if person == "personL":
            data = hdf5_data[0]
        else:
            data = hdf5_data[1]

        # find the keypoint location in data
        if xyz:
            coord = data[frame_index, keypoint_index, :]
        else:
            coord = data[frame_index, keypoint_index, :2]
        return np.array(coord)

    def get_number_of_keypoints(self, input_file, frame_name):
        pass




