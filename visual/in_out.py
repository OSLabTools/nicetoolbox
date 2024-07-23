import os
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
    def __init__(self, config):
        self.dataset_folder = config["dataset_folder"]
        # Experiment details
        self.nice_tool_input_folder = config['nice_tool_input_folder']
        self.experiment_folder = config["experiment_folder"]
        self.experiment_video_component_folder = config['experiment_video_component']

    def initialization(self, dataset_config):
        # Dataset properties
        self.path_to_calibs = dataset_config['dataset_properties']["path_to_calibrations"]

        # Experiment details
        self.experiment_video_component_folder = self.experiment_video_component_folder.replace(
            '<video_name>', dataset_config['media']['video_name']) ## component output

    def get_component_nice_tool_input_folder(self, video_details, dataset_name):
        folder_path = copy.deepcopy(self.nice_tool_input_folder)
        folder_path = folder_path.replace('<dataset_name>', dataset_name)
        folder_path = folder_path.replace('<session_ID>', video_details['session_ID'])
        folder_path = folder_path.replace('<sequence_ID>', video_details['sequence_ID'])
        return folder_path

    def get_component_results_folder(self, video_name, component_name):
        folder_path = copy.deepcopy(self.experiment_video_component_folder)
        folder_path = folder_path.replace('<component_name>', component_name)
        folder_path = folder_path.replace('<video_name>', video_name)
        return folder_path

    def get_algorithm_result(self, component_results_folder, alg_name):
        return os.path.join(component_results_folder, f'{alg_name}.npz')

    def get_calibration_file(self, video_details):
        calib_path = copy.deepcopy(self.path_to_calibs)
        calib_path = calib_path.replace('<session_ID>', video_details['session_ID'])
        calib_path = calib_path.replace('<sequence_ID>', video_details['sequence_ID'])
        return calib_path


