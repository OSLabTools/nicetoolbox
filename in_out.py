"""

"""

import os
import glob
import copy


def flatten_list(input_list):
    if isinstance(input_list, str):
        return [input_list]
    elif isinstance(input_list, list):
        output_list = []
        for item in input_list:
            output_list += flatten_list(item)
        return output_list


class IO:
    def __init__(self, config, method_names):
        self.out_folder = config['out_folder']
        os.makedirs(self.out_folder, exist_ok=True)
        self.tmp_folder = config['tmp_folder']

        self.video_folder = config['video_folder']
        self.calibration_file = config['calibration_file']
        if config['process_data_to'] == 'tmp_folder':
            self.data_folder = self.tmp_folder
        elif config['process_data_to'] == 'data_folder':
            self.data_folder = config['data_folder']

        self.method_names = list(set(flatten_list(method_names)))
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
        return self.video_folder

    def get_calibration_file(self):
        return self.calibration_file

    def get_data_folder(self):
        return self.data_folder

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
                return self.out_folder

