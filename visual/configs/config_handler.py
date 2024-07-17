import os
import glob

# internal imports
import utils.visual_utils as vis_ut
import utils.config as confh

class Configuration:
    def __init__(self, visualizer_config_file, machine_specifics_file):
        self.config_files = dict(
            visualizer_config_file=visualizer_config_file,
            machine_specifics_file=machine_specifics_file,
            )
        
        # load experiment config dicts - these can contain placeholders
        self.visualizer_config = confh.load_config(visualizer_config_file)
        self.machine_specific_config = confh.load_config(machine_specifics_file)
        # load the latest config from the experiment output of isa-tool
        try:
            experiment_config_file = sorted(glob.glob(os.path.join(
                self.localize(self.visualizer_config)['io']['experiment_folder'], 'config_*.toml'
            )))[-1]
        except IndexError:
            print("\nCould not find the latest experiment config file in "
                  f"{self.localize(self.visualizer_config)['io']['experiment_folder']}\n\n")
            raise 
        
        loaded_experiment_config = confh.load_config(experiment_config_file)
        self.experiment_run_config = loaded_experiment_config['run_config']
        self.experiment_detector_config = loaded_experiment_config['detector_config']
        self.dataset_properties = loaded_experiment_config['dataset_config']

    def localize(self, config, fill_io=True, fill_data=False):
        # fill placeholders
        config = confh.config_fill_auto(config)
        config = confh.config_fill_placeholders(config, self.machine_specific_config)
        if fill_io:
            config = confh.config_fill_placeholders(config, self.get_io_config())
        if fill_data:
            config = confh.config_fill_placeholders(config, self.dataset_properties)
        config = confh.config_fill_placeholders(config, config)
        return config

    def get_io_config(self, add_exp=False):
        io_config = self.localize(self.visualizer_config['io'], fill_io=False)
        if add_exp:
            # add to the return config the isa-tool experiment io
            io_config.update(experiment_io=self.localize(self.localize(self.experiment_run_config['io'])))
        return io_config
    
    def get_visualizer_config(self):
        return self.visualizer_config

    def get_visualizer_dataset_config(self, dataset_name):
        """Merge rerun config with dataset parameters, and videos dict run with this dataset"""
        # add details of the videos
        self.visualizer_config.update(videos=self.experiment_run_config['run'][dataset_name]['videos'])
        # add properties of the dataset
        self.visualizer_config.update(dataset_properties=self.dataset_properties[dataset_name])
        self.visualizer_config.update(detector_properties=self.experiment_detector_config)
        return self.localize(self.visualizer_config)

    def get_experiment_config(self, type):
        if type == 'run':
            return self.experiment_run_config
        elif type == 'detector':
            return self.experiment_detector_config
        elif type == 'dataset':
            return self.dataset_properties

    def get_component_algorithms_map(self):
        experiment_run_config = self.experiment_run_config
        return experiment_run_config['component_algorithm_mapping']

    def check_config(self):
        self.check_start_stop_frames()
        self.check_component_name()
        self.check_algorithms()

    def check_start_stop_frames(self):
        dataset_name = self.visualizer_config['media']['dataset_name']
        video_input_config = self.get_visualizer_dataset_config(dataset_name)['videos'][0]
        video_length = video_input_config['video_length']

        #check start frame
        if self.visualizer_config['media']['visualize']['start_frame'] < 0:
            raise ValueError(f"Visualizer_config 'start_frame' parameter cannot be negative.")
        elif self.visualizer_config['media']['visualize']['start_frame'] >video_length:
            raise ValueError(f"Visualizer_config 'start_frame' parameter cannot be greater than the video length. \n"
                             f"Video length: {video_length} frames.")

        #check stop frame
        if self.visualizer_config['media']['visualize']['end_frame'] > video_length:
            raise ValueError(
                f"Visualizer_config 'end_frame' parameter cannot be greater than the video length. \n"
                f"Video length: {video_length} frames.")

        #check visualize interval
        if self.visualizer_config['media']['visualize']['visualize_interval'] > video_length:
            raise ValueError(
                f"Visualizer_config 'visualize_interval' parameter cannot be greater than the video length. \n"
                f"Video length: {video_length} frames.")


    def check_component_name(self):
        component_algorithm_mapping = self.get_component_algorithms_map()
        for component in self.visualizer_config['media']['visualize']['components']:
            if component not in component_algorithm_mapping.keys():
                raise ValueError(f"Component {component} is not found in run file.\n"
                                 f"Delete or correct {component} from Visualizer_config[media.visualize.components]")

    def check_algorithms(self):
        component_algorithm_mapping = self.get_component_algorithms_map()
        for component in self.visualizer_config['media']['visualize']['components']:
            algorithms = self.visualizer_config['media'][component]['algorithms']
            for alg in algorithms:
                if alg not in component_algorithm_mapping[component]:
                    raise ValueError(f"Algorithm {alg} is not found in {component} run file."
                                     f"Delete or correct {alg} from Visualizer_config[media.{component} algorithms")

    def check_calibration(self, calib, cam_name):
        if self.visualizer_config['media']['visualize']['camera_position'] == True:
            cam_matrix, cam_distor, cam_rotation, cam_extrinsic = vis_ut.get_cam_para_studio(calib,
                                                                                         cam_name)
            if (cam_rotation==None) | (cam_extrinsic==None):
                assert ValueError("The roation and extrinsic matrix of the camera could not found.\n"
                                  "Please either change the Visualizer_config 'camera_position' parameter to false \n"
                                  "or provide extrinsics parameters of the camera")


