"""
    Run the code from
    https://github.com/xucong-zhang/ETH-XGaze
    imported here as package 'ethXgaze'
"""

import os
import json
from ethXgaze import run_ethXgaze
from third_party.xgaze_3cams.xgaze_3cams.utils import get_cam_para_studio

from detectors.base_detector import BaseDetector


class ETHXGaze(BaseDetector):
    """Class to setup and run existing computer vision research code.
    """
    name = 'ETH-XGaze'
    behavior = 'gazeDistance'

    def __init__(self, settings) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        settings : dict
            some configurations/settings dictionary, here it must contain the key 'image_file'
        """
        super().__init__(settings)
        self.settings = settings['ethXgaze']

    def data_initialization(self):
        self.data_initialization_frames()

    def inference(self):
        """Run inference of the method on the pre-loaded image

        Returns
        -------
        dict
            dict(method_name:str, values:list)
            the values list contains entries of the form
                dict(feature:str, start:int, end:int, label:str)
        """
        # check that the data was created
        assert self.frames_list is not None, \
            f"{self.name}: Please initialize the data " \
            f"(via 'self.data_initialization()') before running inference."

        label_names = dict(AatB='AlookatB',
                           BatA='BlookatA',
                           mutual='mutual_gaze',
                           averted='gaze_averted')

        detections = dict(name=self.name, values=[])

        cam_matrix, cam_distor, cam_rotation = self.load_calibration()

        for frame_file, frame_idx in zip(self.frames_list, self.frame_indices_list):
            result = run_ethXgaze(
                frame_file,
                self.settings['shape_predictor_filename'],
                self.settings['face_model_filename'],
                self.settings['pretrained_model_filename'],
                os.path.join(self.out_folder, f'gaze_%05d.png' % frame_idx),
                cam_matrix, cam_distor,
                verbose=True)

        return None






