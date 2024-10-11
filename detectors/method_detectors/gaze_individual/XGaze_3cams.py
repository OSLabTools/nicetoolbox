"""
XGaze3cams method detector class.

This code is by XuCong taken from /ps/project/pis/GazeInterpersonalSynchrony/code_from_XuCong
"""

import os
import glob
import numpy as np
import cv2
import logging

from detectors.method_detectors.base_detector import BaseDetector
from detectors.method_detectors.filters import SGFilter
from utils.video import frames_to_video
import utils.logging_utils as log_ut


class XGaze3cams(BaseDetector):
    """
    The XGaze3cams class is a method detector that computes the gaze_individual component.
    
    The method detector computes the gaze of individuals in the scene using multiple
    cameras.It provides the necessary preparations and post-inference visualizations to 
    integrate the XGaze3cams algorithm into our pipeline.
    
    Component: gaze_individual
    
    Attributes:
        components (list): A list containing the name of the component: gaze_individual.
        algorithm (str): The name of the algorithm used to compute the gaze_individual component.
        camera_names (list): A list of camera names used to capture the original input data.
    """
    components = ['gaze_individual']
    algorithm = 'xgaze_3cams'

    def __init__(self, config, io, data) -> None:
        """
        Initialize the XGaze3cams method detector with all inference preparations completed.
        
        Args:
            config (dict): A dictionary containing the configuration settings for the method detector.
            io (class): An instance of the IO class for input-output operations.
            data (class): An instance of the Data class for accessing data.
        """

        logging.info(f"Prepare Inference for '{self.algorithm}' and components {self.components}.")

        config['frames_list'] = data.frames_list
        config['frame_indices_list'] = data.frame_indices_list

        super().__init__(config, io, data, requires_out_folder=config['visualize'])

        self.camera_names = config['camera_names']
        while '' in self.camera_names:
            self.camera_names.remove('')

        self.result_folders = config['result_folders'][self.components[0]]
        self.filtered = config["filtered"]
        if self.filtered:
            self.filter_window_length = config["window_length"]
            self.filter_polyorder = config["polyorder"]

        logging.info(f"Inference Preparation completed.\n")

    def post_inference(self):
        """
        Post-processing after inference.

        This method is called after the inference step and is used for any post-processing tasks that need to be performed.
        """
        if self.filtered:
            prediction_file = os.path.join(self.result_folders, f"{self.algorithm}.npz")
            prediction = np.load(prediction_file, allow_pickle=True)
            predictions_dict = {key:prediction[key] for key in prediction.files}
            data_description = predictions_dict['data_description'].item()
            results_3d = prediction['3d']
            ## Apply filter
            logging.info("APPLYING filtering to Gaze Individual data...")
            results_3d_filtered = results_3d.copy()[:, :, :, None]
            filter = SGFilter(self.filter_window_length, self.filter_polyorder)
            results_3d_filtered = filter.apply(results_3d_filtered, is_3d=True)
            data_description.update({'3d_filtered':data_description['3d']})
            predictions_dict['3d_filtered'] = results_3d_filtered[:, :, :, 0]

            if len(self.camera_names) ==1:
                results_2d = prediction['2d']
                results_2d_filtered = results_2d.copy()[:, :, :, None]
                results_2d_filtered = filter.apply(results_2d_filtered, is_3d=False)
                data_description.update({'2d_filtered': data_description['2d']})
                predictions_dict['2d_filtered'] = results_2d_filtered[:, :, :, 0]

            np.savez_compressed(prediction_file, **predictions_dict)

        else:
            pass

    def visualization(self, data):
        """
        Visualizes the processed frames of the xgaze3cams algorithm as a video for all cameras.

        This function reads the processed frames from each camera, checks if all frames are 
        present, and verifies that the number of frames per camera is consistent. It then creates
        a video for each camera using the processed frames.

        Returns:
            None

        Raises:
            AssertionError: If no frames are found for at least one camera or if the number of 
            frames per camera is not consistent.
        """
        frames_lists = [
            sorted(glob.glob(os.path.join(self.out_folder, f'{cam}_*.png')))
            for cam in self.camera_names]
        log_ut.assert_and_log(
            np.all([len(l) != 0 for l in frames_lists]),
            f"XGaze3cams visualization: no frames found for at least one camera "
            f"'{self.camera_names}' in '{self.out_folder}'.")
        log_ut.assert_and_log(
            np.all([len(frames_lists[0]) == len(l) for l in frames_lists[1:]]),
            f"Not the same number of frames per camera in '{self.out_folder}'.")

        success = True
        for camera_name, frames_list in zip(self.camera_names, frames_lists):
            os.makedirs(os.path.join(self.viz_folder, camera_name), exist_ok=True)
            for idx, file in enumerate(frames_list):
                image = cv2.imread(file)
                cv2.imwrite(os.path.join(self.viz_folder, camera_name,
                                         "%05d.png" % idx), image)

            out_filename = os.path.join(self.viz_folder,
                                        f'output_{camera_name}.mp4')
            success *= frames_to_video(
                    os.path.join(self.viz_folder, camera_name),
                    out_filename, fps=3.0)

        logging.info(f"Detector {self.components}: visualization finished with code "
                     f"{success}.")
