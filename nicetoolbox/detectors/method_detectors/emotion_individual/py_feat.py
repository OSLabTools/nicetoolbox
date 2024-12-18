"""
Py-feat method detector class.
"""

import logging

from ..base_detector import BaseDetector


class PyFeat(BaseDetector):
    """
    The Python - Facial Expression Analysis Toolbox (Py-feat) is a method
    detector that computes emotion_individual component

    Component emotion_individual

    Attributes:
    components (list): A list containing the name of the component: emotion_individual
    algorithm (st): Algorithm name used to compute the emotion_individual component.
    camera_names (list): A list of camera names used to capture original input data.
    """

    components = ["emotion_individual"]
    algorithm = "py_feat"

    def __init__(self, config, io, data) -> None:
        """
        Initialize the PyFeat method detector with all inference preparation.

        Args:
            config (dict): A dictionary containing the configuration settings for
            the method detector.
            io (class): An instance of the IO class for input-output operations.
            data (class): An instance of the Data class for accessing data.
        """

        logging.info(
            f"Prepare Inference for '{self.algorithm}' and component {self.components}."
        )

        config["frames_list"] = data.frames_list
        config["frame_indices_list"] = data.frame_indices_list

        # call the base class __init__
        super().__init__(config, io, data, requires_out_folder=config["visualize"])

        self.camera_names = config["camera_names"]
        while "" in self.camera_names:
            self.camera_names.remove("")

        self.results_folder = config["result_folders"][self.components[0]]

        logging.info("Inference Preparation complete.\n")

    def post_inference(self):
        """
        Post processing after inference.
        """
        pass

    def visualization(self, data):
        """
        Visualizes the processed frames of the pyfeat algorithm as a video.

        Returns:
            None

        Raises:
            AssertionError: If no frames are found for at least one camera or if
            the number of frames per camera is not consistent.
        """
        pass
