"""
    https://github.com/radekd91/emoca
"""

import os
import glob
import numpy as np
import cv2
import logging

from detectors.base_detector import BaseDetector
from oslab_utils.video import frames_to_video
import oslab_utils.logging_utils as log_ut


class SPELL(BaseDetector):
    """Class to setup and run existing computer vision research code.
    """
    name = 'active_speaker'
    behavior = 'active_speaker'

    def __init__(self, config, io, data) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        config : dict
            some configurations/settings dictionary, here it must contain the key 'image_file'
        """
        config['snippets_list'] = data.snippets_list
        config['segment_length'] = 60  # in sec

        # create output folders
        out = io.get_output_folder(self.name, 'output')
        config['intermediate_results'] = io.get_output_folder(self.name, 'additional')
        config['audio_dir'] = os.path.join(out, 'audio')
        config['audio_slices_dir'] = os.path.join(config['audio_dir'], 'slices')
        config['instance_crops_dir'] = os.path.join(out, 'instance_crops')
        config['csv_file'] = os.path.join(
                out, f"metadata_seg{config['segment_length']}s.csv")
        config['features_dir'] = os.path.join(out, 'features', config['ASC_model_name'])
        config['graphs_dir'] = os.path.join(out, 'graphs')

        # create folders
        for dir in [config['audio_dir'], config['audio_slices_dir'],
                    config['instance_crops_dir'], config['features_dir'],
                    config['graphs_dir']]:
            os.makedirs(dir, exist_ok=True)

        # logfile
        self.log = os.path.join(out, f"{self.name}_inference.log")
        config['log_file'] = self.log


        # last, call the the base class init
        super().__init__(config, io, data)

        # for visualization code
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
