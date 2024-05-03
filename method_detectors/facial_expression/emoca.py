"""
    https://github.com/radekd91/emoca
"""

import os
import glob
import numpy as np
import cv2
import logging

from method_detectors.base_detector import BaseDetector
from oslab_utils.video import frames_to_video
import oslab_utils.logging_utils as log_ut


class Emoca(BaseDetector):
    """Class to setup and run existing computer vision research code.
    """
    name = 'facial_expressions'
    algorithm = 'emoca'

    def __init__(self, config, io, data) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        config : dict
            some configurations/settings dictionary, here it must contain the key 'image_file'
        """
        config['input_data_folder'] = data.create_symlink_input_folder(
                config['input_data_format'], config['camera_names'])

        # last, call the the base class init
        super().__init__(config, io, data)

        self.camera_names = config['camera_names']
        self.method_out_folder = config['out_folder']

    def visualization(self, data):
        method_folders = [os.path.join(self.method_out_folder, name)
                          for name in self.camera_names]
        frames_lists = [sorted(glob.glob(os.path.join(folder, '*.png')))
                        for folder in method_folders]
        log_ut.assert_and_log(
            np.all([len(l) != 0 for l in frames_lists]),
            f"emoca visualization: no frames found in at least one of "
            f"'{method_folders}'.")
        log_ut.assert_and_log(
            np.all([len(frames_lists[0]) == len(l) for l in frames_lists[1:]]),
            f"Not the same number of frames per camera in '{method_folders}'.")

        for idx in range(len(frames_lists[0])):
            images = [cv2.imread(file) for file in [f[idx] for f in frames_lists]]
            image = np.concatenate(images, axis=1)
            cv2.imwrite(os.path.join(self.viz_folder, "%05d.png" % idx), image)

        out_filename = os.path.join(self.viz_folder, 'method_output.mp4')
        success = frames_to_video(self.viz_folder, out_filename, fps=3.0)
        logging.info(f"Detector {self.name}: visualization finished with code "
                     f"{success}.")
