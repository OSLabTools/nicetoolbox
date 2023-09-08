"""
    A class for data handling
"""

import os
import glob
import json
from oslab_utils.video import equal_splits_by_frames, get_fps, \
    read_segments_list_from_file, split_into_frames, cut_length


class Data:
    name = 'data'

    def __init__(self, config, io) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        config : dict
            some configurations/settings dictionary
        """

        # TODO later: add caching for tmp folder

        self.tmp_folder = io.get_output_folder(self.name, 'tmp')
        self.code_folder = config['io']['code_folder']

        # collect which data slices and formats are required
        self.data_formats = set()
        self.all_camera_ids = set()
        for detector in config['methods']['names']:
            assert 'input_data_format' in config['methods'][detector].keys(), \
                f"Please specify the key 'input_data_format' for detector " \
                f"'{detector}'. Options are 'segments' and 'frames'."
            self.data_formats.add(config['methods'][detector]['input_data_format'])
            assert 'camera_ids' in config['methods'][detector].keys(), \
                f"Please create the key 'camera_ids' for detector '{detector}'."
            self.all_camera_ids.update(config['methods'][detector]['camera_ids'])

        # DATA INITIALIZATION
        self.video_folder = io.get_input_folder()
        self.annotation_interval = config['annotation_interval']
        self.number_of_frames = config['number_of_frames']
        self.segments_list = None
        self.frames_list = None
        self.frame_indices_list = None
        self.videos_list = None

        self.data_initialization()

        # LOAD CALIBRATION
        self.calibration = self.load_calibration(io.get_calibration_file())

    def get_inference_path(self, detector_name):
        filepath = os.path.join(self.code_folder, 'third_party', detector_name,
                                'inference.py')
        #assert os.path.exists(filepath), f"detector inference file {filepath}" \
        #                                 f" does not exist!"
        return filepath

    def get_venv_path(self, detector_name, env_name):
        filepath = os.path.join(self.code_folder, 'third_party', detector_name,
                                f'{env_name}/bin/activate')
        assert os.path.exists(filepath), f"detector inference file {filepath}" \
                                         f" does not exist!"
        return filepath

    def load_calibration(self, calibration_file):
        fid = open(calibration_file)
        calib = json.load(fid)
        fid.close()
        return calib

    def data_initialization(self):
        # detect all video input files
        video_files = sorted(glob.glob(os.path.join(self.video_folder, '*')))

        # calculate frames per annotation interval
        frames_per_split = self.annotation_interval * get_fps(video_files[0])

        self.frames_list = []
        self.segments_list = []
        self.videos_list = []
        for video_file in video_files:
            camera_id = int(video_file[video_file.find('Cam') + 3:][0])
            if camera_id not in self.all_camera_ids:
                continue

            # split video into frames
            data_folder = os.path.join(self.tmp_folder, f'Cam{camera_id}')
            os.makedirs(data_folder, exist_ok=True)

            if 'frames' in self.data_formats:
                frames_list, frame_indices_list = split_into_frames(
                        video_file,
                        os.path.join(data_folder, 'frame'),
                        self.number_of_frames,
                        self.annotation_interval
                )

                if self.frame_indices_list is None:
                    self.frame_indices_list = frame_indices_list
                    self.frames_list = [[f] for f in frames_list]
                else:
                    for n, (i_old, i_new, f) in enumerate(zip(
                            self.frame_indices_list, frame_indices_list,
                            frames_list)):
                        assert i_old == i_new, \
                            "Frame indices of different cameras do not match!"
                        self.frames_list[n].append(f)

            if 'segments' in self.data_formats:
                # split video into segments of length annotation_interval
                segments_list_file = equal_splits_by_frames(
                        video_file,
                        os.path.join(data_folder, 'segment'),
                        frames_per_split,
                        keep_last_split=False,
                        number_of_frames=self.number_of_frames
                )

                # read result list
                segments_list = read_segments_list_from_file(segments_list_file)
                self.segments_list.append(segments_list)

            if 'video' in self.data_formats:
                # cut video to the required number of frames
                video_file = cut_length(
                        video_file,
                        os.path.join(data_folder, 'video'),
                        number_of_frames=self.number_of_frames
                )

                # read result list
                self.videos_list.append(video_file)

