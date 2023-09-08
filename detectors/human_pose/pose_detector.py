"""

"""

import os
from detectors.base_detector import BaseDetector
from oslab_utils.filehandling import save_to_hdf5


class PoseDetector(BaseDetector):
    """
    """
    name = 'mmpose'
    behavior = 'keypoints'

    def __init__(self, config, io, data):
        """ Initialize PoseHRNet class.

        Parameters
        ----------
        config : dict
            the method-specific configurations dictionary
        io: class
            a class instance that handles in-output folders
        """
        self.algorithm_name = config['algorithm']
        self.name += '_' + self.algorithm_name

        # first, make additions to the method/detector's config:
        # extract the relevant data input files from the data class
        videos_list = []
        for camera_id in config['camera_ids']:
            # find the index of this camera_id in the data.videos_list:
            all_cameras_idx = list(data.all_camera_ids).index(camera_id)

            # get the filename of the video for this camera_id
            videos_list.append(data.videos_list[all_cameras_idx])
        config['videos_list'] = videos_list

        # then, call the the base class init
        super().__init__(config, io, data)

        # maybe needed/helpful: create a folder for additional outputs
        self.add_output_folder = io.get_output_folder(self.name, 'additional')

        # notes:
        # the camera IDs are in self.camera_ids
        # all input video filenames are listed in data.videos_list
        # in the config, we can define the number_of_frames to only use a part
        # of the input videos, this is available under data.number_of_frames
        # the annotation interval (in secs) is in data.annotation_interval

    def visualization(self, data):
        """

        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """

        pass

