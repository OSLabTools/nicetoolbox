"""
SPIGA method detector class.
"""

import logging

from ..base_detector import BaseDetector


class Spiga(BaseDetector):
    """
    SPIGA is a method detector that computes the head_orientation component.

    Component: head_orientation

    Attributes:
        components (list): A list containing the name of the component: head_orientation
        algorithm (str): Algorithm name used to compute the head_orientation component.
        camera_names (list): List of camera names used to capture original input data.
    """

    components = ["head_orientation"]
    algorithm = "spiga"

    def __init__(self, config, io, data) -> None:
        """
        Initialize the SPIGA method detector with all inference preparation.

        Args:
            config (dict): Configuration settings for SPIGA.
            io (class): IO class instance for input/output operations.
            data (class): Data class instance for frame and subject data.
        """
        logging.info(
            f"Prepare Inference for '{self.algorithm}' and component {self.components}."
        )

        self.frames_list = data.frames_list
        config["frames_list"] = self.frames_list
        config["frame_indices_list"] = data.frame_indices_list
        self.video_start = data.video_start

        # Call the base class constructor
        super().__init__(config, io, data, requires_out_folder=config["visualize"])

        self.camera_names = [n for n in config["camera_names"] if n]
        self.cam_sees_subjects = config["cam_sees_subjects"]
        self.results_folder = config["result_folders"][self.components[0]]

        logging.info("SPIGA Inference Preparation complete.\n")

    def post_inference(self):
        """
        Optional post-processing after SPIGA inference.
        """
        pass

    def visualization(self, data):
        """
        Skip visualization for now.
        """
        pass
