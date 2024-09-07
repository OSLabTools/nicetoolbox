import os
import copy

class IO:
    def __init__(self, config):
        # Experiment details
        self.nice_tool_input_folder = config['io']['nice_tool_input_folder']
        # Dataset properties
        self.path_to_calibs = config['dataset_properties']["path_to_calibrations"]

        # Replace the <video_name> placeholder in the experiment video folder path
        self.experiment_video_folder = config['io']['experiment_video_folder'].replace(
            '<experiment_folder>', config['io']['experiment_folder']).replace(
            '<video_name>', config['media']['video_name']
        )
        # Replace the <component_name> placeholder for the experiment video component folder
        self.experiment_video_component_folder = config['io']['experiment_video_component'].replace(
            '<experiment_video_folder>', self.experiment_video_folder
        )

    def get_component_nice_tool_input_folder(self, video_details, dataset_name):
        folder_path = copy.deepcopy(self.nice_tool_input_folder)
        folder_path = folder_path.replace('<dataset_name>', dataset_name)
        folder_path = folder_path.replace('<session_ID>', video_details['session_ID'])
        folder_path = folder_path.replace('<sequence_ID>', video_details['sequence_ID'])
        return folder_path

    def get_experiment_video_folder(self):
        return self.experiment_video_folder

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


