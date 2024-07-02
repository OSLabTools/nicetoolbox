"""
    A class for data handling
"""

import os
import glob
import json
import logging
import cv2
import numpy as np
from oslab_utils.video import equal_splits_by_frames, get_fps, \
    read_segments_list_from_file, split_into_frames, cut_length
import oslab_utils.system as oslab_sys
import oslab_utils.video as vid
import oslab_utils.check_and_exception as exc
import oslab_utils.in_out as ut_in_out
import shutil


class Data:
    """
    A data class for loading and processing data.

    Attributes:
        name (str): The name of the data.
        data_folder (str): The folder path for the data.
        tmp_folder (str): The folder path for temporary files.
        code_folder (str): The folder path for code files.
        data_input_folder (str): The folder path for input data.
        session_ID (str): The session ID.
        sequence_ID (str): The sequence ID.
        video_length (int): The length of the video.
        video_start (int): The starting frame of the video.
        video_skip_frames (int or None): The number of frames to skip in the video.
        annotation_interval (float): The interval for annotations.
        subjects_descr (str): The description of the subjects.
        camera_mapping (dict): The mapping of camera names.
        data_formats (list): The list of data formats required.
        all_camera_names (list): The list of all camera names.
        segments_list (None or list): The list of data segments.
        frames_list (None or list): The list of frames.
        frame_indices_list (None or list): The list of frame indices.
        snippets_list (None or list): The list of snippets.
        calibration (dict): The camera calibration.
        fps (int): The frames per second of the input video files.
    """

    name = 'data'

    def __init__(self, config, io, data_formats, all_camera_names) -> None:
        """
        Initialize the Data class.

        Args:
            config (dict): The configuration dictionary.
            io (IO): The IO object for file and folder operations.
            data_formats (list): The list of data formats required.
            all_camera_names (list): The list of all camera names.

        Returns:
            None
        """
        logging.info("Start DATA LOADING and processing.")
        # TODO later: add caching for tmp folder

        # check given data-config
        #self.check_config(config)

        # collect all required file/folder paths
        self.data_folder = io.get_data_folder()
        self.tmp_folder = io.get_output_folder(self.name, 'tmp')
        self.code_folder = config['code_folder']
        self.data_input_folder = io.get_input_folder()

        # collect data details from config
        self.session_ID = config['session_ID']
        self.sequence_ID = config['sequence_ID']
        self.video_length = config['video_length']
        self.video_start = config['video_start']
        self.video_skip_frames = None if config['video_skip_frames'] is False \
            else config['video_skip_frames']
        self.annotation_interval = config['annotation_interval']
        self.subjects_descr = config['subjects_descr']
        self.camera_mapping = dict((key, config[key]) for key in config.keys() if 'cam_' in key)

        # collect which data slices and formats are required
        self.data_formats = data_formats
        self.all_camera_names = all_camera_names
        self.segments_list = None
        self.frames_list = None
        self.frame_indices_list = None
        self.snippets_list = None

        # DATA INITIALIZATION
        self.data_initialization()

        # LOAD CALIBRATION
        self.calibration = self.load_calibration(io.get_calibration_file(),
                                                 config['dataset_name'])
        self.fps = self.get_fps(config['fps'])

        logging.info("Data loading and processing finished.\n\n")

    def get_inference_path(self, detector_name):
        """
        Get the file path for the inference script of a given detector.

        Args:
            detector_name (str): The name of the detector.

        Returns:
            str: The file path for the inference script.

        Raises:
            FileNotFoundError: If the inference script file does not exist.
        """
        filepath = os.path.join(self.code_folder, 'third_party', detector_name, 'inference.py')
        try:
            exc.file_exists(filepath)
        except FileNotFoundError:
            logging.exception(f"Detector inference file {filepath} does not exist!")
            raise
        return filepath

    def get_venv_path(self, detector_name, env_name):
        """
        Get the file path of the virtual environment for the given detector and environment name.

        Args:
            detector_name (str): The name of the detector.
            env_name (str): The name of the environment.

        Returns:
            str: The file path of the virtual environment.

        Raises:
            FileNotFoundError: If the virtual environment does not exist.
        """
        os_type = oslab_sys.detect_os_type()
        if os_type == "linux":
            filepath = os.path.join(self.code_folder, 'third_party', detector_name,
                                    f'{env_name}/bin/activate')
        elif os_type == "windows":
            filepath = os.path.join(self.code_folder, 'third_party', detector_name,env_name, 'Scripts', 'activate')
        try:
            exc.file_exists(filepath)
        except FileNotFoundError:
            logging.exception(
                f"Virtual environment file {filepath} for detector = "
                f"'{detector_name}' does not exist!")
            raise
        return filepath

    def get_fps(self, config_fps):
        """
        Get the frames per second (fps) of the input video files.

        Args:
            config_fps (int): The desired fps specified in the configuration.

        Returns:
            int: The fps of the input video files. If the input formats are not 'mp4' or 'avi',
                 the config_fps value is returned.
        """
        input_formats = self.get_input_format(self.all_camera_names)
        if input_formats in ['mp4', 'avi']:
            video_files = sorted(glob.glob(os.path.join(self.data_input_folder, '*')))
            fps = get_fps(video_files[0])
            if fps != config_fps:
                logging.warning(
                    f"Detected fps = {fps} does not match fps given in the config = {config_fps}!")
            return fps
        else:
            return config_fps

    def check_config(self, config):
        """
        Checks the validity of the configuration for the detector methods.

        Args:
            config (dict): The configuration dictionary.

        Raises:
            KeyError: If a required key is missing in the configuration.
            TypeError: If the value of a key is of an incorrect type.
            ValueError: If the value of a key is out of bounds.
        """
        for detector in config['methods']['names']:
            try:
                config['methods'][detector]['input_data_format']
            except KeyError:
                # ToDo detector specific assert?
                logging.exception(
                    f"Please specify the key 'input_data_format' for detector "
                    f"'{detector}'. Options are 'segments', 'frames', and "
                    f"'snippets'.")
                raise
            try:
                exc.check_options(
                        config['methods'][detector]['input_data_format'],
                        str, ["frames", "segments", "snippets"])
            except (TypeError, ValueError):
                logging.exception(
                    f"Unsupported 'input data format' in '{detector}'. "
                    f"Options are 'segments', 'frames', and 'snippets'.")
                raise
            try:
                config['methods'][detector]['camera_names']
            except KeyError:
                logging.exception(
                    f"Please create the key 'camera_names' for detector "
                    f"'{detector}'.")
                raise

            # ensure that the detector runs on the given dataset
            if 'exclude_datasets' in config['methods'][detector].keys():
                if config['io']['dataset_name'] in config['methods'][detector]['exclude_datasets']:
                    logging.warning(f"Detector {detector} excludes dataset {config['io']['dataset_name']}. "
                                    f"Removing the detector from the methods to be run.")
                    config['methods']['names'].remove(detector)

        # check video details
        data_input_files = sorted(glob.glob(os.path.join(config['io']['data_input_folder'], '*')))
        total_frames = 0
        # TODO update to data_input_files instead of video_files
        for video_file in data_input_files:
            total_frames = max(total_frames,
                               vid.get_number_of_frames(video_file))

        try:
            exc.check_value_bounds(config['video_start'], int, object_min=0,
                                   object_max=total_frames - 1)
        except (TypeError, ValueError):
            logging.exception(f"Invalid argument for video_start.")
            raise
        try:
            exc.check_value_bounds(
                config['video_length'], int,
                object_max=total_frames - config['video_start'])
        except (TypeError, ValueError):
            logging.exception(f"Invalid argument for video_length.")
            raise
        try:
            exc.check_value_bounds(config['annotation_interval'], float,
                                   object_min=0, object_max=total_frames)
        except (TypeError, ValueError):
            logging.exception(f"Invalid argument for annotation_interval.")
            raise
        try:
            if not isinstance(config['video_skip_frames'], bool):
                exc.check_value_bounds(
                    config['video_skip_frames'], int, object_min=1,
                    object_max=config['video_length'])
        except (TypeError, ValueError):
            logging.exception(f"Invalid argument for video_skip_frames.")
            raise

        # frames_per_segment = self.annotation_interval * get_fps(
        #                             video_files[0])
        # total_frames (60) > frames_per_split (60)

    def load_calibration(self, calibration_file, dataset_name):
        """
        Load camera calibration from a file for a specific dataset.
        
        Currently implemented for the datasets 'dyadic_communication' and 'mpi_inf_3dhp'.

        Args:
            calibration_file (str): The path to the calibration file.
            dataset_name (str): The name of the dataset.

        Returns:
            dict: A dictionary containing the loaded camera calibration.

        Raises:
            NotImplementedError: If loading camera calibration for the specified dataset is not implemented.
        """
        try:
            exc.check_options(dataset_name, str, ['dyadic_communication', 'mpi_inf_3dhp'])
        except (TypeError, ValueError):
            logging.exception(
                f"Loading camera calibration for dataset '{dataset_name}' is "
                f"not implemented.")
            raise NotImplementedError

        if self.sequence_ID == '':
            loaded_calib = np.load(calibration_file, allow_pickle=True)[self.session_ID].item()
        else:
            loaded_calib = np.load(calibration_file, allow_pickle=True)[self.session_ID + '_' + self.sequence_ID].item()

        calib = dict((key, value) for key, value in loaded_calib.items()
                        if key in self.all_camera_names)

        return calib

    def get_input_format(self, camera_names):
        """
        Get the input format for the given camera names.

        Args:
            camera_names (list): A list of camera names.

        Returns:
            str: The input format for the given camera names.

        Raises:
            ValueError: If multiple or no valid input format is found in the data input folder.
        """
        example_input_folder = self.data_input_folder.replace('<camera_name>', next(iter(camera_names)))
        input_formats = [name in '_'.join(sorted(os.listdir(example_input_folder))) for
                        name in ['.mp4', '.avi', '.png', '.jpg', '.jpeg']]
        if sum(input_formats) != 1:
            exc.error_log_and_raise(
                ValueError, 
                'Reading input data', 
                f"Multiple or no valid input format found in '{self.data_input_folder}'. "
                f"Found '{input_formats}', valid formats are ['mp4', 'avi']."
                )
        input_format = ['mp4', 'avi', 'png', 'jpg', 'jpeg'][input_formats.index(True)]
        return input_format
        
    def get_inputs_list(self, input_format, data_format, camera_names):
        """
        Returns a list of input file paths based on the specified input format,
        data format, and camera names.

        Args:
            input_format (str): The format of the input files.
            data_format (str): The format/type of the video data. 
                One of snippets, segments, or frames.
            camera_names (list): A list of camera names.

        Returns:
            list: A list of input file paths.
        """

        start = self.video_start
        end = self.video_start + self.video_length

        inputs_list = []
        if data_format == 'snippets':
            for camera_name in camera_names:
                file_name = f"{camera_name}_s{start}_e{end}.{input_format}"
                inputs_list.append(os.path.join(
                        self.data_folder, camera_name, 'snippets', file_name))

        elif data_format == 'segments':
            video_files = sorted(glob.glob(os.path.join(self.data_input_folder, '*')))
            step = int(self.annotation_interval * get_fps(video_files[0]))
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

    def data_initialization(self):
        """
        Initializes the data required for running NICE toolbox.

        This method performs the following steps:
        1. Determines the input format based on the available camera names.
        2. Creates a list of all input files required to run NICE toolbox.
        3. Checks whether all required data files exist.
        4. Initializes data lists for frames, segments, and snippets.
        5. If the data exists, extracts frame indices and organizes frames by camera name.
        6. If the data does not exist, creates the required data from video or frames.
        7. Logs the completion of data creation.
        """
        # find data input format
        input_format = self.get_input_format(self.all_camera_names)

        # create a list of all input files required to run the nice toolbox
        # given the current run_config.toml
        data_list = []
        for data_format in self.data_formats:
            data_list += self.get_inputs_list(input_format, data_format, self.all_camera_names)

        # check whether all required data exists already
        data_exists = True
        for file in data_list:
            if not os.path.isfile(file):
                data_exists = False

        # initialize data lists
        self.frames_list = []
        self.segments_list = []
        self.snippets_list = []

        if data_exists:
            logging.info(f"DATA FOUND in '{self.data_folder}'!")

            frames_list = []
            frame_indices_list = set()
            for filename in data_list:
                if 'frames' in filename:
                    frames_list.append(filename)
                    frame_indices_list.add(int(os.path.basename(filename)[:-4]))
                elif 'segments' in filename:
                    self.segments_list.append(filename)
                elif 'snippets' in filename:
                    self.snippets_list.append(filename)

            self.frame_indices_list = list(frame_indices_list)
            for camera_name in sorted(self.all_camera_names):
                cam_frames = sorted([file for file in frames_list if camera_name in file])
                self.frames_list.append(cam_frames)
            self.frames_list = [l.tolist() for l in np.array(self.frames_list).T]

        else:
            logging.info(f"DATA NOT EXISTING OR INCOMPLETE! Creating data in "
                  f"'{self.data_folder}'!")

            if input_format in ['avi', 'mp4']:
                self.create_inputs_from_video()
            elif input_format in ['png', 'jpg', 'jpeg']:
                self.create_inputs_from_frames(input_format)

            logging.info(f"DATA creation completed.")

    def create_inputs_from_video(self):
        """
        Create inputs from video files.

        This method detects video input files, splits them into frames, and organizes the frames
        into different data formats which are frames, segments, and snippets.
        
        Raises:
            AssertionError: If the length of the frame indices list does not match the specified
                video length.
            AssertionError: If the frame indices of different cameras do not match.        
        """
        # detect all video input files
        video_files = sorted(glob.glob(os.path.join(self.data_input_folder, '*')))

        for video_file in video_files:
            camera_name_indices = [name in video_file.lower() 
                                   for name in list(self.all_camera_names)]
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
                frames_per_segment = int(self.annotation_interval * get_fps(
                    video_files[0]))

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

    def create_inputs_from_frames(self, input_format):
        """
        Processes frames and organizes them into specified data formats for further
        processing in the NICE pipeline.

        This method iterates through all camera names, and for each camera, it performs the 
        following operations based on the given data formats:

        1. Frames: For each frame in the specified range, it checks if the frame exists in the
        input directory. If it does, the method creates a symbolic link in the output directory
        under a 'frames' subdirectory.

        2. Segments: (Not Implemented) This part is intended to split the video into segments of a
        specified length based on the annotation interval.

        3. Snippets: (Not Implemented) This part is intended to cut the video into snippets based 
        on the specified start and length.

        Args:
            input_format (str): The file format of the input frames (e.g., 'jpg', 'png').

        Raises:
            NotImplementedError: If the filename convention inferred from the first frame's 
            filename does not apply to any frame or if the 'segments' or 'snippets' data formats are specified, as these are not implemented.
        """
        frames_list = []
        frame_indices_list = set()
        for camera_name in self.all_camera_names:
            # define frames input folder
            frames_input_folder = self.data_input_folder.replace('<camera_name>', camera_name)
            input_frame_paths = sorted(glob.glob(os.path.join(frames_input_folder, f"*.{input_format}")))
            
            # guess for number of characters in filename base
            n_filename_chars = len(os.path.basename(input_frame_paths[0]).split('.')[0])
            filename_template = f"%0{n_filename_chars}d.{input_format}"

            # split video into frames
            data_folder = os.path.join(self.data_folder, camera_name)
            os.makedirs(data_folder, exist_ok=True)

            if 'frames' in self.data_formats:
                os.makedirs(os.path.join(data_folder, 'frames'), exist_ok=True)
                camera_frames_list = []

                skip = self.video_skip_frames if self.video_skip_frames is not None else 1
                for frame_idx in range(self.video_start, self.video_start + self.video_length, skip):

                    input_frame_indices = np.where([filename_template % frame_idx in path for path in input_frame_paths])[0]
                    if len(input_frame_indices) != 1:
                        exc.error_log_and_raise(
                            NotImplementedError, 
                            'Create input data from frames.',
                            f"Detected dataset filename convention '{filename_template}', not applicable for " \
                            f"camera name '{camera_name}' and frame index '{frame_idx}'."
                            )
                    input_frame_idx = input_frame_indices[0]

                    in_framename = input_frame_paths[input_frame_idx]
                    out_filename = os.path.join(data_folder, 'frames', "%05d.png" % frame_idx)

                    if not os.path.exists(out_filename):
                        # create system link
                        os.symlink(in_framename, out_filename)

                    # update class attributes frame_list and frame_indices_list
                    frame_indices_list.add(frame_idx)
                    camera_frames_list.append(out_filename)

                frames_list.append(camera_frames_list)

            # TODO: Clean up or implement
            if 'segments' in self.data_formats:
                raise NotImplementedError
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

            # TODO: Clean up or implement
            if 'snippets' in self.data_formats:
                raise NotImplementedError
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

        self.frame_indices_list = list(frame_indices_list)
        #self.frames_list = np.array(frames_list).T
        self.frames_list = [list(pair) for pair in zip(*frames_list)]


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

        # get the list of needed input files
        input_format = self.get_input_format(camera_names)
        source_file_list = self.get_inputs_list(input_format, data_format, camera_names)

        # Check if the data folder & symlinks inside is already exists.
        if os.path.isdir(data_folder):
            logging.info(f"Data folder is found'{data_folder}' - Checking if the symlinks are valid")
            existing_symlink_list = ut_in_out.list_files_under_root(data_folder)
            if len(source_file_list) != len(existing_symlink_list):
                logging.info(f"Checking data folder - Number of files in the existing data folder does not match. "
                             f"New symlinks will be created")
                # delete already existing symlinks
                ut_in_out.delete_files_into_list(existing_symlink_list)
            #check if the first file into list is a valid file
            elif not os.path.isfile(existing_symlink_list[0]):
                logging.info(f"Checking data folder - Symlink is not valid New symlinks will be created")
                # delete already existing symlinks
                ut_in_out.delete_files_into_list(existing_symlink_list)
            else:
                logging.info(f"Symlinks are found in {data_folder}")
                return data_folder

        # create all folders and subfolders
        os.makedirs(data_folder, exist_ok=True)
        for camera_name in camera_names:
            os.makedirs(os.path.join(data_folder, camera_name), exist_ok=True)

        # create symbolic links
        logging.info(f"Creating symlinks under {data_folder}")
        for source_file in source_file_list:
            if not os.path.exists(source_file):
                logging.warning(f"WARNING! data file '{source_file}' does not exist!")

            else:
                indices = [source_file.find(name) for name in camera_names]
                cam_name = source_file[max(indices):]
                os_type = oslab_sys.detect_os_type()
                if os_type == "linux":
                    cam_name = cam_name[: cam_name.find('/')]
                elif os_type == "windows":
                    cam_name = cam_name[: cam_name.find('\\')]
                else:
                    logging.error("Unknown os type in create_symlink_input_folder")

                try:
                    os.symlink(source_file,
                               os.path.join(data_folder, cam_name,
                                            os.path.basename(source_file)))
                except OSError as e:
                    logging.error(f"Error creating symlink: {e}")

        return data_folder
