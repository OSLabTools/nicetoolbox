"""
    This code is by XuCong
    taken from /ps/project/pis/GazeInterpersonalSynchrony/code_from_XuCong
"""

import os
import glob
import numpy as np
import cv2
import logging

from detectors.base_detector import BaseDetector
from oslab_utils.video import frames_to_video
import oslab_utils.logging_utils as log_ut


class XGaze3cams(BaseDetector):
    """Class to setup and run existing computer vision research code.
    """
    name = 'xgaze_3cams'
    behavior = 'gazeDistance'

    def __init__(self, config, io, data) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        config : dict
            some configurations/settings dictionary, here it must contain the key 'image_file'
        """
        # first, make additions to the method/detector's config:
        # extract the relevant data input files from the data class

        # assert data.all_camera_names == set(config['camera_names']), \
        #     f"camera_names do not match! all loaded cameras = " \
        #     f"'{data.all_camera_names}' and {self.name} requires cameras " \
        #     f"'{config['camera_names']}'."
        config['frames_list'] = data.frames_list
        config['frame_indices_list'] = data.frame_indices_list

        # last, call the the base class init
        super().__init__(config, io, data)

        self.camera_names = config['camera_names']
        while '' in self.camera_names:
            self.camera_names.remove('')
        self.method_out_folder = config['out_folder']

    def visualization(self, data):
        frames_lists = [
            sorted(glob.glob(os.path.join(self.method_out_folder, f'{cam}_*.png')))
            for cam in self.camera_names]
        log_ut.assert_and_log(
            np.all([len(l) != 0 for l in frames_lists]),
            f"XGaze3cams visualization: no frames found for at least one camera "
            f"'{self.camera_names}' in '{self.method_out_folder}'.")
        log_ut.assert_and_log(
            np.all([len(frames_lists[0]) == len(l) for l in frames_lists[1:]]),
            f"Not the same number of frames per camera in '{self.method_out_folder}'.")

        success = True
        for camera_name, frames_list in zip(self.camera_names, frames_lists):
            os.makedirs(os.path.join(self.viz_folder, camera_name), exist_ok=True)
            for idx, file in enumerate(frames_list):
                image = cv2.imread(file)
                cv2.imwrite(os.path.join(self.viz_folder, camera_name,
                                         "%05d.png" % idx), image)

            out_filename = os.path.join(self.viz_folder,
                                        f'method_output_{camera_name}.mp4')
            success *= frames_to_video(
                    os.path.join(self.viz_folder, camera_name),
                    out_filename, fps=3.0)

        logging.info(f"Detector {self.name}: visualization finished with code "
                     f"{success}.")
