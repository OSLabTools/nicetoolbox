"""
    A class for data handling
"""

import os
import glob
import json
import logging
import numpy as np
from oslab_utils.video import equal_splits_by_frames, get_fps, \
    read_segments_list_from_file, split_into_frames, cut_length
import oslab_utils.system as oslab_sys



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
        self.video_skip_frames = None if config['video_skip_frames'] is False \
            else config['video_skip_frames']
        self.annotation_interval = config['annotation_interval']
        self.segments_list = None
        self.frames_list = None
        self.frame_indices_list = None
        self.snippets_list = None

        self.data_folder = io.get_data_folder()
        self.data_initialization()

        # LOAD CALIBRATION
        self.calibration = self.load_calibration(io.get_calibration_file(),
                                                 config['io']['dataset_name'])

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

    def load_calibration(self, calibration_file, dataset_name):
        if dataset_name == 'ours':
            fid = open(calibration_file)
            loaded_calib = json.load(fid)
            fid.close()
            calib = dict((key, value) for key, value in loaded_calib.items()
                         if key in self.all_camera_names)

        elif dataset_name == 'mpi_inf_3dhp':
            def eval_string(string):
                if string.isdigit():
                    return int(string)
                elif '.' in string:
                    return float(string)
                else:
                    return string

            def load_dict_from_textfile(filepath):
                file = open(filepath, 'r')
                _ = file.readline()
                lines = file.readlines()
                file.close()

                d = {}
                current_key = None
                for line in lines:
                    line = line.removesuffix('\n')
                    if line.startswith('name'):
                        line = line.removeprefix('name')
                        line_parts = [l for l in line.split(' ') if l != '']
                        assert len(line_parts) == 1, \
                            f"Loading dict from textfile failed."
                        current_key = line_parts[0]
                        d.update({current_key: {}})

                    elif line.startswith(' '):
                        assert current_key is not None, \
                            f"Loading dict from textfile failed."
                        line = line.strip(' ')
                        line_parts = [eval_string(l) for l in line.split(' ') if l != '']
                        d[current_key].update({line_parts[0]: line_parts[1:]})

                    else:
                        raise NotImplementedError(
                                "Loading dict from textfile failed.")

                return d

            def create_projection_matrix(int_matrix, ext_matrix):
                """
                The function takes intrinsics and extrinsic matrices of the camera as input
                and calculates the projection matrix
                :param int_matrix: intrinsic matrix of the camera (3x3)
                :param ext_matrix: extrinsic matrix of the camera (3x4)
                :return: projection matrix (3x4)
                """
                projection_matrix = np.matmul(int_matrix, ext_matrix)
                return projection_matrix

            loaded_dict = load_dict_from_textfile(calibration_file)
            calib = {}
            for camera in self.all_camera_names:
                idx = camera[-1]
                K = np.array(loaded_dict[idx]['intrinsic']).reshape(4, 4)
                Rt = np.array(loaded_dict[idx]['extrinsic']).reshape(4, 4)
                calib.update({camera: dict(
                        camera_name=camera,
                        image_size=loaded_dict[idx]['size'],
                        intrinsic_matrix=K[:3, :3].tolist(),
                        distortions=[0, 0, 0, 0],
                        rotation_matrix=Rt[:3, :3].tolist(),
                        translation=Rt[2:3, :3].tolist(),
                        extrinsics_matrix=Rt[:3].tolist(),
                        projection_matrix=create_projection_matrix(
                                K[:3, :3], Rt[:3]).tolist()
                )})

        else:
            raise NotImplementedError(f"Loading camaera calibration for dataset"
                                      f" '{dataset_name}' is not implemented.")
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

        # initialize data lists
        self.frames_list = []
        self.segments_list = []
        self.snippets_list = []

        if data_exists:
            print(f"DATA FOUND in '{self.data_folder}'!")

            frame_indices_list = set()
            for filename in data_list:
                if 'frames' in filename:
                    self.frames_list.append(filename)
                    frame_indices_list.add(int(os.path.basename(filename)[:-4]))
                elif 'segments' in filename:
                    self.segments_list.append(filename)
                elif 'snippets' in filename:
                    self.snippets_list.append(filename)
            self.frame_indices_list = list(frame_indices_list)
        else:
            print(f"DATA NOT EXISTING OR INCOMPLETE! Creating data in "
                  f"'{self.data_folder}'!")

            for video_file in video_files:
                camera_name_indices = [name in video_file.lower() for name in
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
                    assert len(frame_indices_list) == self.video_length, \
                        f"ERROR. len(frame_indices_list) = " \
                        f"{len(frame_indices_list)} and self.video_length = " \
                        f"{self.video_length}"

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
                    out_name = f'{camera_name}_s{self.video_start}_e' \
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
            for camera_name in camera_names:
                file_name = f"{camera_name}_s{start}_e{end}.{input_format}"
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
            skip = 1 if not self.video_skip_frames else self.video_skip_frames
            file_names = [f"%05d.png" % x for x in range(start, end, skip)]
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
                    os_type = oslab_sys.detect_os_type()
                    if os_type == "linux":
                        cam_name = cam_name[: cam_name.find('/')]
                    elif os_type == "windows":
                        cam_name = cam_name[: cam_name.find('\\')]
                    else:
                        logging.error("Unknown os type in create_symlink_input_folder")
                    os.symlink(filename,
                               os.path.join(data_folder, cam_name,
                                            os.path.basename(filename)))

            # #assertion
            # for camera_name in camera_names:
            #     camera_path = os.path.join(data_folder, camera_name)
            #     log_ut.assert_and_log(os.listdir(camera_path)!=[], f"Symlink input folder is empty {camera_name}")
            #     os.rmdir(data_folder)
        return data_folder
