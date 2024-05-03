"""
Run nodding pigeon. 

https://github.com/bhky/nodding-pigeon
"""

import os
import numpy as np
from noddingpigeon.inference import predict_video
from noddingpigeon.video import VideoSegment

from method_detectors.base_detector import BaseDetector
from oslab_utils.filehandling import save_to_hdf5


class NoddingPigeon(BaseDetector):
    """
    """
    name = 'head'
    algorithm = 'nodding_pigeon'
    label_names = dict(nodding='nod',
                       turning='shake',
                       stationary='static',
                       undefined='unknown')

    def __init__(self, config, io) -> None:
        """ InitializeMethod class.

        Parameters
        ----------
        config : dict
            the method-specific configurations dictionary
        io: class
            a class instance that handles in-output folders
        """
        super().__init__(config, io)

    def inference(self, data):
        """ Run inference of the method on the pre-loaded image
        """

        results = []
        for camera_id in self.camera_ids:
            list_idx = list(data.all_camera_ids).index(camera_id)

            results_array = np.empty((len(data.segments_list[list_idx]), 4))
            for i, (video_file, start_time, end_time) in enumerate(data.segments_list[list_idx]):

                result = predict_video(
                    video_file,
                    video_segment=VideoSegment.LAST,  # Optionally change the parameters.
                    motion_threshold=0.5,
                    gesture_threshold=0.9
                    )

                gesture = result['gesture']
                # retrieve estimated probability of prediction
                if gesture in ['nodding', 'turning']:
                    probability = result['probabilities']['gestures'][gesture]
                else:
                    probability = -1

                results_array[i] = np.array([
                    start_time,
                    end_time,
                    list(self.label_names.keys()).index(gesture),
                    probability])

            results.append(results_array)

        cam_pers_map = dict(Cam1='PersonL', Cam2='PersonR')
        desc = '_' + '_'.join(self.label_names.values())
        save_to_hdf5(results,
                     [cam_pers_map[f"Cam{camera_id}"] + desc
                      for camera_id in self.camera_ids],
                     os.path.join(self.result_folder, f"{self.algorithm}.hdf5"))

    def visualization(self, data):
        pass

