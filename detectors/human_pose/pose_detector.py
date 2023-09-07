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

    def __init__(self, config, io):
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

        super().__init__(config, io)
        # self.viz_folder and self.result_folder are available to all detectors
        self.output_folder = io.get_output_folder(self.name, 'output')
        self.add_output_folder = io.get_output_folder(self.name, 'additional')

        self.framework = config['framework']
        self.env_name = config['env_name']
        self.multi_person = config['multi_person']
        self.save_images = config['save_images']
        self.resolution = config['resolution']
        self.keypoint_mapping = config['keypoint_mapping']
        self.min_detection_confidence = config['min_detection_confidence']
        self.pose_config = config['pose_config']
        self.detection_config = config['detection_config']
        self.pose_checkpoint = config['pose_checkpoint']
        self.detection_checkpoint = config['detection_checkpoint']

    def inference(self, data):
        """

        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        # the camera IDs are in self.camera_ids
        # all input video filenames are listed in data.videos_list

        results = []
        for camera_id in self.camera_ids:
            # find the index of this camera_id in the data.videos_list:
            all_cameras_idx = list(data.all_camera_ids).index(camera_id)

            # get the filename of the video for this camera_id
            video_file = data.videos_list[all_cameras_idx]
            print(video_file)

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

