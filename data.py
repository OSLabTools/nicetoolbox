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
        self.all_camera_names = set()
        for detector in config['methods']['names']:
            assert 'input_data_format' in config['methods'][detector].keys(), \
                f"Please specify the key 'input_data_format' for detector " \
                f"'{detector}'. Options are 'segments', 'frames', and 'snippets'." #ToDo detector specific assert?
            assert config['methods'][detector]['input_data_format'] in ["frames", "segments", "snippets"], \
                f"Unsupported input data format in " \
                f"'{detector}'. Options are 'segments', 'frames', and 'snippets'."
            self.data_formats.add(config['methods'][detector]['input_data_format'])
            assert 'camera_names' in config['methods'][detector].keys(), \
                f"Please create the key 'camera_names' for detector '{detector}'."
            self.all_camera_names.update(config['methods'][detector]['camera_names'])

        # DATA INITIALIZATION
        self.video_folder = io.get_input_folder()
        self.video_length = config['video_length']
        self.video_start = config['video_start']
        self.video_skip_frames = config['video_skip_frames']
        self.annotation_interval = config['annotation_interval']
        self.segments_list = None
        self.frames_list = None
        self.frame_indices_list = None
        self.snippets_list = None

        self.data_folder = io.get_data_folder()
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

        data_list = []
        for data_format in self.data_formats:
            data_list += self.get_inputs_list(data_format, self.all_camera_names)

        data_exists = True
        for file in data_list:
            if not os.path.isfile(file):
                data_exists = False

        if data_exists:
            print(f"DATA FOUND in '{self.data_folder}'!")
        else:
            print(f"DATA NOT EXISTING OR INCOMPLETE! Creating data in "
                  f"'{self.data_folder}'!")

            self.frames_list = []
            self.segments_list = []
            self.snippets_list = []
            for video_file in video_files:
                camera_name_indices = [name in video_file for name in
                                      list(self.all_camera_names)]
                if not any(camera_name_indices):
                    continue
                camera_name = list(self.all_camera_names)[camera_name_indices.index(True)]

                # split video into frames
                data_folder = os.path.join(self.data_folder, camera_name)
                os.makedirs(data_folder, exist_ok=True)

                if 'frames' in self.data_formats:
                    os.makedirs(os.path.join(data_folder, 'frames'), exist_ok=True)
                    frames_list, frame_indices_list = split_into_frames(
                            video_file,
                            os.path.join(data_folder, 'frames/'),
                            self.video_start,
                            self.video_length,
                            self.video_skip_frames
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
                    os.makedirs(os.path.join(data_folder, 'segments'), exist_ok=True)

                    # calculate frames per annotation interval
                    frames_per_segment = self.annotation_interval * get_fps(
                            video_files[0])

                    # split video into segments of length annotation_interval
                    self.segments_list = equal_splits_by_frames(
                            video_file,
                            os.path.join(data_folder, 'segments/'),
                            frames_per_segment,
                            keep_last_split=False,
                            start_frame=self.video_start,
                            number_of_frames=self.video_length
                    )

                if 'snippets' in self.data_formats:
                    os.makedirs(os.path.join(data_folder, 'snippets'), exist_ok=True)
                    out_name = f's{self.video_start}_e' \
                               f'{self.video_start + self.video_length}'
                    # cut video to the required number of frames
                    out_video_file = cut_length(
                            video_file,
                            os.path.join(data_folder, f'snippets/{out_name}'),
                            start_frame=self.video_start,
                            number_of_frames=self.video_length
                    )

                    # read result list
                    self.snippets_list.append(out_video_file)

    def get_inputs_list(self, data_format, camera_names):
        input_formats = [name in '_'.join(os.listdir(self.video_folder)) for
                         name in ['mp4', 'avi']]
        assert sum(input_formats) == 1, \
            f"Multiple or no valid input format found in " \
            f"'{self.video_folder}'. {input_formats} for ['mp4', 'avi']."
        input_format = ['mp4', 'avi'][input_formats.index(True)]

        start = self.video_start
        end = self.video_start + self.video_length

        inputs_list = []
        if data_format == 'snippets':
            file_name = f"s{start}_e{end}.{input_format}"
            for camera_name in camera_names:
                inputs_list.append(os.path.join(
                        self.data_folder, camera_name, 'snippets', file_name))

        elif data_format == 'segments':
            video_files = sorted(glob.glob(os.path.join(self.video_folder, '*')))
            step = self.annotation_interval * get_fps(video_files[0])
            file_names = [f"s{s}_e{s + step}.{input_format}"
                          for s in range(start, end, step)]
            for camera_name in camera_names:
                inputs_list += [
                    os.path.join(self.data_folder, camera_name, 'segments', n)
                    for n in file_names]

        elif data_format == 'frames':
            skip = self.video_skip_frames
            file_names = [f"%05d.png" % (start + x * skip)
                          for x in range(start, end, skip)]
            for camera_name in camera_names:
                inputs_list += [
                    os.path.join(self.data_folder, camera_name, 'frames', n)
                    for n in file_names]

        return inputs_list

    def create_symlink_input_folder(self, data_format, camera_names):
        # define folder structure and naming
        folder_name = f"{data_format}_{'_'.join(camera_names)}_" \
                      f"s{self.video_start}_" \
                      f"e{self.video_start + self.video_length}"
        if data_format == 'frames':
            folder_name += f"_s{self.video_skip_frames}"
        if data_format == 'segments':
            folder_name += f"_s{self.annotation_interval}"

        data_folder = os.path.join(
                self.data_folder, 'symlink_input_folders', folder_name)

        if not os.path.isdir(data_folder):
            # create all folders and subfolders
            os.makedirs(data_folder, exist_ok=True)
            for camera_name in camera_names:
                os.makedirs(os.path.join(data_folder, camera_name), exist_ok=True)

            # get the list of needed input files
            file_list = self.get_inputs_list(data_format, camera_names)

            # create symbolic links
            for filename in file_list:
                if not os.path.exists(filename):
                    print(f"WARNING! file '{filename}' does not exist!")

                else:
                    indices = [filename.find(name) for name in camera_names]
                    cam_name = filename[max(indices):]
                    cam_name = cam_name[: cam_name.find('/')]
                    os.symlink(filename,
                               os.path.join(data_folder, cam_name,
                                            os.path.basename(filename)))

        return data_folder
