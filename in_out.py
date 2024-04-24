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

        self.log_level = config['log_level']

    def get_log_file_level(self):
        return os.path.join(self.out_folder, "ISA-Tool.log"), self.log_level

    def get_config_file(self):
        return self.out_folder

    def initialization(self, config, method_names):
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

        self.method_names = method_names
        self.method_out_folder = config['method_out_folder']
        self.method_visualization_folder = config['method_visualization_folder']
        self.method_additional_output_folder = config['method_additional_output_folder']
        self.method_tmp_folder = config['method_tmp_folder']
        self.method_final_result_folder = config['method_final_result_folder']

    def get_all_tmp_folders(self):
        method_tmp_folders = [
            self.method_tmp_folder.replace('<method_name>', method_name)
            for method_name in self.method_names]
        return [self.tmp_folder, *method_tmp_folders]

    def get_input_folder(self):
        return self.data_input_folder

    def get_calibration_file(self):
        return self.calibration_file

    def get_data_folder(self):
        return self.data_folder

    def get_conda_path(self):
        return self.conda_path

    def get_output_folder(self, name, token):
        if any([m_name in name for m_name in self.method_names]):
            if token == 'output':
                folder_name = copy.deepcopy(self.method_out_folder)
            elif token == 'visualization':
                folder_name = copy.deepcopy(self.method_visualization_folder)
            elif token == 'additional':
                folder_name = copy.deepcopy(self.method_additional_output_folder)
            elif token == 'tmp':
                folder_name = copy.deepcopy(self.method_tmp_folder)
            elif token == 'result':
                folder_name = copy.deepcopy(self.method_final_result_folder)
            else:
                raise NotImplementedError(
                        f"IO return output folder: Token '{token}' unknown! "
                        f"Supported are 'output', 'visualization', "
                        f"'additional', 'tmp', 'result'.")

            folder_name = folder_name.replace('<method_name>', name)
            os.makedirs(folder_name, exist_ok=True)
            return folder_name

        elif name == 'data':
            if token == 'tmp':
                return self.tmp_folder

        elif name == 'config':
            if token == 'output':
                return self.out_sub_folder

        else:
            raise NotImplementedError(
                    f"IO return output folder: Name '{name}' unknown! "
                    f"Supported are 'data', 'config', and method_names: "
                    f"{self.method_names}.")

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

        # check all given method output folders
        def check_base_folder(folder_name, token, description):
            try:
                exc.check_token_in_filepath(folder_name, token, description)
            except ValueError:
                logging.exception("The given method input folder is invalid.")
                raise

            base = folder_name.split(token)[0][:-1]
            try:
                _ = sorted(os.listdir(base))
            except OSError:
                logging.exception(f"'{base}' is not an accessible directory.")
                raise

        check_base_folder(config['method_out_folder'], '<method_name>',
                          'method_out_folder')
        check_base_folder(config['method_visualization_folder'], '<method_name>',
                          'method_visualization_folder')
        check_base_folder(config['method_additional_output_folder'],
                          '<method_name>', 'method_additional_output_folder')
        check_base_folder(config['method_final_result_folder'], '<method_name>',
                          'method_final_result_folder')
        check_base_folder(config['method_tmp_folder'], 'tmp',
                          'method_tmp_folder')
