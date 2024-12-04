"""

"""

import copy
import logging
import os

from ..utils import check_and_exception as exc
from ..utils import config as cfg
from ..utils import filehandling as fh
from ..utils.logging_utils import log_configs


def flatten_list(input_list):
    if isinstance(input_list, str):
        return [input_list]
    if isinstance(input_list, int):
        return [input_list]
    if isinstance(input_list, list):
        output_list = []
        for item in input_list:
            output_list += flatten_list(item)
        return output_list
    raise NotImplementedError


def flatten_dict(dictionary):
    output_dict = copy.deepcopy(dictionary)
    for key, value in dictionary.items():
        if isinstance(value, dict):
            del output_dict[key]
    return output_dict


def add_to_filename(filename, addition):
    filename_split = filename.split(".")
    filename_split[-2] += addition
    return (".").join(filename_split)


class Configuration:
    def __init__(self, run_config_file, machine_specifics_file):
        # load experiment config dicts - these might contain placeholders
        self.run_config = fh.load_config(run_config_file)
        self.run_config_check_file = add_to_filename(run_config_file, "_check")
        self.machine_specific_config = fh.load_config(machine_specifics_file)
        self.machine_specific_config.update(dict(pwd=os.getcwd()))

        # detector_config
        detector_config_file = self.localize(self.run_config, False)["io"][
            "detectors_config"
        ]
        self.detector_config = fh.load_config(detector_config_file)
        for detector_name, detector_dict in self.detector_config["algorithms"].items():
            if "framework" in detector_dict:
                framework = self.detector_config["frameworks"][
                    detector_dict["framework"]
                ]
                self.detector_config["algorithms"][detector_name].update(framework)

        dataset_config_file = self.localize(self.run_config, False)["io"][
            "dataset_properties"
        ]
        self.dataset_config = fh.load_config(dataset_config_file)
        self.current_data_config = None

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

    def get_io_config(self):
        self.run_config["io"]["log_level"] = logging.INFO
        return self.localize(self.run_config["io"], fill_io=False)

    def get_dataset_configs(self):
        for dataset_name, dataset_dict in self.run_config["run"].items():
            if not isinstance(dataset_dict, dict):
                continue

            component_dict = dict(
                (comp, self.run_config["component_algorithm_mapping"][comp])
                for comp in dataset_dict["components"]
            )

            for video_config in dataset_dict["videos"]:
                video_config.update(dataset_name=dataset_name)
                video_config.update(self.dataset_config[dataset_name])
                video_config.update(self.get_io_config())
                self.current_data_config = self.localize(video_config)
                yield self.current_data_config, component_dict

    def get_method_configs(self, method_names):
        for method_name in method_names:
            method_config = flatten_dict(
                self.detector_config["algorithms"][method_name]
            )
            method_config["visualize"] = self.run_config["visualize"]

            if "algorithm" in method_config:
                method_config.update(
                    self.detector_config["methods"][method_name][
                        method_config["algorithm"]
                    ]
                )

            localized_config = self.localize(method_config, fill_data=True)
            localized_config["camera_names"] = [
                cam for cam in localized_config["camera_names"] if cam != ""
            ]
            yield localized_config, method_name

    def get_feature_configs(self, feature_names):
        for feature_name in feature_names:
            feature_config = copy.deepcopy(
                self.detector_config["algorithms"][feature_name]
            )
            feature_config["visualize"] = self.run_config["visualize"]

            yield feature_config, feature_name

    def save_experiment_config(self, output_folder):
        # save all experiment configurations
        log_configs(
            dict(
                run_config=cfg.config_fill_auto(self.run_config),
                dataset_config=self.dataset_config,
                detector_config=self.detector_config,
                machine_specific_config=self.machine_specific_config,
            ),
            output_folder,
            file_name="config_<time>",
        )

    def get_all_detector_names(self):
        algorithms = list(self.detector_config["algorithms"].keys())
        feature_methods = [
            self.detector_config["algorithms"][name]["input_detector_names"]
            for name in algorithms
            if "input_detector_names" in self.detector_config["algorithms"][name]
        ]
        return list(set(flatten_list(algorithms + feature_methods)))

    def get_all_camera_names(self, algorithm_names):
        all_camera_names = set()
        detector_config = self.localize(self.detector_config, fill_data=True)
        for detector in algorithm_names:
            if "camera_names" in detector_config["algorithms"][detector]:
                all_camera_names.update(
                    detector_config["algorithms"][detector]["camera_names"]
                )
        if "" in all_camera_names:
            all_camera_names.remove("")
        return all_camera_names

    def get_all_input_data_formats(self, algorithm_names):
        data_formats = set()
        for detector in algorithm_names:
            if "input_data_format" in self.detector_config["algorithms"][detector]:
                data_formats.add(
                    self.detector_config["algorithms"][detector]["input_data_format"]
                )
        return data_formats

    def get_all_dataset_names(self):
        return list(self.dataset_config.keys())

    def save_csv(self):
        return self.run_config["save_csv"]

    def checker(self):
        # check USER INPUT
        logging.info("Start USER INPUT CHECK.")
        run_config_check = fh.load_config(self.run_config_check_file)
        localized_run_config = self.localize(self.run_config)
        exc.check_user_input_config(
            localized_run_config, run_config_check, "run_config"
        )
        logging.info("User input check finished successfully.\n\n\n")
