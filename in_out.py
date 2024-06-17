"""

"""

import os
import glob
import copy
import logging
import oslab_utils.check_and_exception as exc


class IO:
    def __init__(self, config):
        # create folders
        self.out_folder = config['out_folder']
        try:
            os.makedirs(self.out_folder, exist_ok=True)
        except OSError:
            logging.exception("Failed creating the output folder.")
            raise

        try:
            os.makedirs(os.path.dirname(config['data_folder']), exist_ok=True)
        except OSError:
            logging.exception(f"Failed creating the base data folder {config['data_folder']}.")
            raise

        self.log_level = config['log_level']

    def get_log_file_level(self):
        return os.path.join(self.out_folder, "ISA-Tool.log"), self.log_level

    def get_config_file(self):
        return self.out_folder

    def initialization(self, config, algorithm_names):
        self.out_sub_folder = config['out_sub_folder']
        self.tmp_folder = config['tmp_folder']
        if config['process_data_to'] == 'tmp_folder':
            self.data_folder = self.tmp_folder
        elif config['process_data_to'] == 'data_folder':
            self.data_folder = config['data_folder']
        self.create_folders()

        # check the given io-config
        self.check_config(config)

        # get the relevant config entries
        self.data_input_folder = config['data_input_folder']
        self.calibration_file = config['path_to_calibrations']
        self.conda_path = config['conda_path']

        self.algorithm_names = algorithm_names
        self.detector_out_folder = config['detector_out_folder']
        self.detector_visualization_folder = config['detector_visualization_folder']
        self.detector_additional_output_folder = config['detector_additional_output_folder']
        self.detector_tmp_folder = config['detector_tmp_folder']
        self.detector_run_config_path = config['detector_run_config_path']
        self.detector_final_result_folder = config['detector_final_result_folder']

    def get_all_tmp_folders(self):
        detector_tmp_folders = [
            self.detector_tmp_folder.replace('<algorithm_names>', algorithm_names)
            for algorithm_names in self.algorithm_names]
        return [self.tmp_folder, *detector_tmp_folders]

    def get_input_folder(self):
        return self.data_input_folder

    def get_calibration_file(self):
        return self.calibration_file

    def get_data_folder(self):
        return self.data_folder

    def get_conda_path(self):
        return self.conda_path

    def get_output_folder(self, name, token):
        if name == 'data':
            if token == 'tmp':
                return self.tmp_folder

        elif name == 'config':
            if token == 'output':
                return self.out_sub_folder

        else:
            raise NotImplementedError(
                    f"IO return output folder: Name '{name}' unknown! "
                    f"Supported are 'data', 'config'.")

    def get_detector_output_folder(self, component, algorithm, token):
        if not any([m_name in algorithm for m_name in self.algorithm_names]):
            raise NotImplementedError(
                f"IO return component output folder: Algorithm '{algorithm}' unknown! "
                f"Supported are: {self.algorithm_names}.")
        
        if token == 'output':
            folder_name = copy.deepcopy(self.detector_out_folder)
        elif token == 'visualization':
            folder_name = copy.deepcopy(self.detector_visualization_folder)
        elif token == 'additional':
            folder_name = copy.deepcopy(self.detector_additional_output_folder)
        elif token == 'tmp':
            folder_name = copy.deepcopy(self.detector_tmp_folder)
        elif token == 'result':
            folder_name = copy.deepcopy(self.detector_final_result_folder)
        elif token == 'run_config':
            folder_name = copy.deepcopy(self.detector_run_config_path)
        else:
            raise NotImplementedError(
                    f"IO return output folder: Token '{token}' unknown! "
                    f"Supported are 'output', 'visualization', "
                    f"'additional', 'tmp', 'result'.")

        folder_name = folder_name.replace('<component_name>', component).replace('<algorithm_name>', algorithm)
        os.makedirs(folder_name, exist_ok=True)
        return folder_name

    def create_folders(self):
        # create output folders
        try:
            os.makedirs(self.out_sub_folder, exist_ok=True)
        except OSError:
            logging.exception("Failed creating the output folder.")
            raise
        # create the data folders
        try:
            os.makedirs(self.data_folder, exist_ok=True)
        except OSError:
            logging.exception("Failed creating the data folder.")
            raise

    def check_config(self, config):
        # check the config['process_data_to'] input
        try:
            exc.check_options(config['process_data_to'], str,
                              ['tmp_folder', 'data_folder'])
        except (TypeError, ValueError):
                logging.exception(
                    f"Unsupported 'process_data_to' in io. "
                    f"Valid options are 'tmp_folder' and 'data_folder'.")
                raise

        # check all given detector output folders
        def check_base_folder(folder_name, token, description):
            try:
                exc.check_token_in_filepath(folder_name, token, description)
            except ValueError:
                logging.exception("The given detector input folder is invalid.")
                raise

            base = folder_name.split(token)[0][:-1]
            try:
                _ = sorted(os.listdir(base))
            except OSError:
                logging.exception(f"'{base}' is not an accessible directory.")
                raise

        check_base_folder(config['detector_out_folder'], '<component_name>',
                          'detector_out_folder')
        check_base_folder(config['detector_visualization_folder'], '<component_name>',
                          'detector_visualization_folder')
        check_base_folder(config['detector_additional_output_folder'],
                          '<component_name>', 'detector_additional_output_folder')
        check_base_folder(config['detector_final_result_folder'], '<component_name>',
                          'detector_final_result_folder')
        check_base_folder(config['detector_tmp_folder'], 'tmp',
                          'detector_tmp_folder')
