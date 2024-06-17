"""

"""

import os
import glob
import logging
import copy
from oslab_utils.logging_utils import log_configs
import oslab_utils.config as cfg
import oslab_utils.check_and_exception as exc


def get_top_level_dict(dictionary):
    return {k: v for k, v in dictionary.items() if not isinstance(v, dict)}


def flatten_list(input_list):
    if isinstance(input_list, str):
        return [input_list]
    if isinstance(input_list, int):
        return [input_list]
    elif isinstance(input_list, list):
        output_list = []
        for item in input_list:
            output_list += flatten_list(item)
        return output_list


def flatten_dict(dictionary):
    output_dict = copy.deepcopy(dictionary)
    for key, value in dictionary.items():
        if isinstance(value, dict):
            del output_dict[key]
    return output_dict


def compare_configs(config1, config2, log_fct=print, config_names=''):
    shared_keys = set(config1.keys()).intersection(set(config2.keys()))

    # do the configs contain different sets of keys?
    all_keys = set(config1.keys()).union(set(config2.keys()))
    non_shared_keys = all_keys.difference(shared_keys)
    if len(non_shared_keys) != 0:
        log_fct(f"Configs {config_names} differ in their keys! "
                f"Key(s) '{non_shared_keys}' only exist(s) for one of them.")

    # do the values match?
    keys_different_values = [k for k in shared_keys if config1[k] != config2[k]]
    if len(keys_different_values) != 0:
        log_fct(f"Configs {config_names} contain different values for keys "
                f"'{keys_different_values}!")

    return len(non_shared_keys) != 0, len(keys_different_values) != 0


def add_to_filename(filename, addition):
    filename_split = filename.split('.')
    filename_split[-2] += addition
    return ('.').join(filename_split)


class Configuration:
    def __init__(self, run_config_file, detector_config_file, machine_specifics_file):
        # load experiment config dicts - these might contain placeholders
        self.run_config = cfg.load_config(run_config_file)
        self.run_config_check_file=add_to_filename(run_config_file, '_check')
        self.machine_specific_config = cfg.load_config(machine_specifics_file)
        self.machine_specific_config.update(dict(pwd=os.getcwd()))

        # detector_config
        self.detector_config = cfg.load_config(detector_config_file)
        for detector_name, detector_dict in self.detector_config['algorithms'].items():
            if 'framework' in detector_dict.keys():
                framework = self.detector_config['frameworks'][detector_dict['framework']]
                self.detector_config['algorithms'][detector_name].update(framework)

        dataset_config_file = self.localize(self.run_config, False)['io']['dataset_config']
        self.dataset_config = cfg.load_config(dataset_config_file)
        self.current_data_config = None

        #self.check_config()

    def localize(self, config, fill_io=True, fill_data=False):
        # fill placeholders
        config = cfg.config_fill_auto(config)
        config = cfg.config_fill_placeholders(config, self.machine_specific_config)
        if fill_io:
            config = cfg.config_fill_placeholders(config, self.get_io_config())
        if fill_data:
            config = cfg.config_fill_placeholders(config, self.current_data_config)
        config = cfg.config_fill_placeholders(config, config)
        return config


        #config = config_fill_placeholders(config, self.localized_config['io'])
        #config_level_one = get_top_level_dict(self.localized_config)
        #self.localized_config = config_fill_placeholders(
        #        self.localized_config, config_level_one)

    #def get_localized_config(self):
    #    return self.localized_config

    def get_io_config(self):
        match self.run_config['run_mode']:
            case "experiment":
                log_level = logging.INFO
            case "development":
                log_level = logging.INFO
            case "production":
                log_level = logging.ERROR
        self.run_config['io']['log_level'] = log_level

        return self.localize(self.run_config['io'], fill_io=False)

    def get_dataset_configs(self):
        for dataset_name, dataset_dict in self.run_config['run'].items():
            if not isinstance(dataset_dict, dict):
                continue

            component_dict = dict((comp, self.run_config['component_algorithm_mapping'][comp]) 
                                  for comp in dataset_dict['components'])

            for video_config in dataset_dict['videos']:
                video_config.update(dataset_name=dataset_name)
                video_config.update(self.dataset_config[dataset_name])
                video_config.update(self.get_io_config())
                self.current_data_config = self.localize(video_config)
                yield self.current_data_config, component_dict

    def get_method_configs(self, method_names):
        for method_name in method_names:
            method_config = flatten_dict(self.detector_config['algorithms'][method_name])
            method_config['visualize'] = self.run_config['visualize']

            if 'algorithm' in method_config.keys():
                method_config.update(
                    self.detector_config['methods'][method_name][method_config['algorithm']])

            yield self.localize(method_config, fill_data=True), method_name

    def get_feature_configs(self, feature_names):
        for feature_name in feature_names:
            feature_config = copy.deepcopy(self.detector_config['algorithms'][feature_name])
            feature_config['visualize'] = self.run_config['visualize']

            yield feature_config, feature_name

    def save_experiment_config(self, output_folder):
        # save all experiment configurations
        log_configs(dict(run_config=cfg.config_fill_auto(self.run_config),
                         dataset_config=self.dataset_config,
                         detector_config=self.detector_config,
                         machine_specific_config=self.machine_specific_config),
                    output_folder,
                    file_name=f"config_<time>")

    def get_all_detector_names(self):
        algorithms = list(self.detector_config['algorithms'].keys())
        feature_methods= [
            self.detector_config['algorithms'][name]['input_detector_names'] 
            for name in algorithms if 'input_detector_names' in self.detector_config['algorithms'][name].keys()]
        return list(set(flatten_list(algorithms + feature_methods)))

    def get_all_camera_names(self, algorithm_names):
        all_camera_names = set()
        detector_config = self.localize(self.detector_config, fill_data=True)
        for detector in algorithm_names:
            if 'camera_names' in detector_config['algorithms'][detector].keys():
                all_camera_names.update(detector_config['algorithms'][detector]['camera_names'])
        if '' in all_camera_names:
            all_camera_names.remove('')
        return all_camera_names

    def get_all_input_data_formats(self, algorithm_names):
        data_formats = set()
        for detector in algorithm_names:
            if 'input_data_format' in self.detector_config['algorithms'][detector].keys():
                data_formats.add(self.detector_config['algorithms'][detector]['input_data_format'])
        return data_formats

    def checker(self):

        # check USER INPUT
        logging.info(f"Start USER INPUT CHECK.")
        run_config_check = cfg.load_config(self.run_config_check_file)
        localized_run_config = self.localize(self.run_config)
        exc.check_user_input_config(localized_run_config, run_config_check, "run_config")
        logging.info(f"User input check finished successfully.\n\n\n")

    def check_config(self):
        try:
            exc.check_options(
                self.config_abstract['run_mode'], str,
                ["experiment", "development", "production"])
        except (TypeError, ValueError):
            logging.exception(
                f"Unsupported input for '{self.config_abstract['run_mode']}'. "
                f"Options are 'experiment', 'development', 'production'.")
            raise

        if self.config_abstract['run_mode'] == 'production' and __debug__:
            logging.error(
                "In 'production' mode, this script is meant to be run "
                "with environment variable 'PYTHONOPTIMIZE=1' or, "
                "equivalently, 'python -0' option for optimized performance.")

        # TODO check detector and feature names - do dict entries exist

        # TODO check whether needed detectors ran before features

        # TODO check cameras per detector - if not set, update config and remove these
            
        # check dataset_config            
        # check the given calibration file and video folder
        try:
            f = open(self.dataset_config['calibration_file'])
            f.close()
        except (KeyError, OSError):
            logging.exception("Failed loading the calibration file.")
            raise
        try:
            _ = sorted(os.listdir(self.dataset_config['video_folder']))
        except OSError:
            logging.exception(
                f"The given video folder is not an accessible directory.")
            raise

    def check_config_consistency(self, folder):
        match self.localized_config['run_mode']:
            case "experiment":
                log_fct = logging.error
            case _:
                log_fct = logging.info

        saved_config_files = glob.glob(os.path.join(folder, 'config_*.toml'))
        if len(saved_config_files) != 0:
            for saved_config_file in saved_config_files:
                config_names = f'current and loaded ' \
                               f'({os.path.basename(saved_config_file)})'
                saved_config = cfg.load_config(saved_config_file)

                # compare top level dicts
                compare_configs(get_top_level_dict(saved_config['config']),
                                get_top_level_dict(self.localized_config),
                                logging.error, config_names)

                # compare machine specifics
                compare_configs(saved_config['machine_specific_config'],
                                self.machine_specific_config, logging.info,
                                config_names)

                # compare io
                compare_configs(saved_config['config']['io'],
                                self.config['io'], log_fct, config_names)


