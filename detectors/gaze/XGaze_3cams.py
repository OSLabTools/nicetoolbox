"""
    This code is by XuCong
    taken from /ps/project/pis/GazeInterpersonalSynchrony/code_from_XuCong
"""

from detectors.base_detector import BaseDetector


class XGaze3cams(BaseDetector):
    """Class to setup and run existing computer vision research code.
    """
    name = 'xgaze_3cams'
    behavior = 'gaze'

    def __init__(self, config, io, data) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        config : dict
            some configurations/settings dictionary, here it must contain the key 'image_file'
        """
        # first, make additions to the method/detector's config:
        # extract the relevant data input files from the data class
        assert data.all_camera_names == set(config['camera_names']), \
            f"camera_names do not match! all loaded cameras = " \
            f"'{data.all_camera_names}' and {self.name} requires cameras " \
            f"'{config['camera_names']}'."
        config['frames_list'] = data.frames_list
        config['frame_indices_list'] = data.frame_indices_list

        # last, call the the base class init
        super().__init__(config, io, data)

    def visualization(self, data):
        pass
