""" """

import copy
from typing import Generator, Tuple

from ..configs.config_loader import ConfigLoader
from ..configs.schemas.dataset_properties import DatasetProperties
from ..configs.schemas.detectors_config import DetectorsConfig
from ..configs.schemas.detectors_run_file import DetectorsRunFile
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..configs.utils import (
    default_auto_placeholders,
    default_runtime_placeholders,
    model_to_dict,
)
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
    cfg_loader: ConfigLoader
    auto_placeholders: dict[str, str]
    runtime_placeholders: set[str]

    machine_specific_config: dict
    run_config: dict
    detector_config: dict
    dataset_config: dict
    current_data_config: dict

    def __init__(self, run_config_file, machine_specifics_file):
        # init config loader
        self.auto_placeholders = default_auto_placeholders()
        self.runtime_placeholders = default_runtime_placeholders()
        self.cfg_loader = ConfigLoader(
            self.auto_placeholders, self.runtime_placeholders
        )
        # machine specific
        machine_specific_config = self.cfg_loader.load_config(
            machine_specifics_file, MachineSpecificConfig
        )
        self.cfg_loader.extend_global_ctx(machine_specific_config)
        # run file
        run_config = self.cfg_loader.load_config(run_config_file, DetectorsRunFile)
        self.cfg_loader.extend_global_ctx(run_config.io)
        # detectors config
        # TODO: Need to convert path to string. To be unified.
        detector_config_file = str(run_config.io.detectors_config)
        detector_config = self.cfg_loader.load_config(
            detector_config_file, DetectorsConfig
        )
        # dataset config
        dataset_config_file = str(run_config.io.dataset_properties)
        dataset_config = self.cfg_loader.load_config(
            dataset_config_file, DatasetProperties
        )

        # TODO: rest of the codebase except the configs as dict
        # so we convert them from models to configs
        # will be refactored soon
        self.machine_specific_config = model_to_dict(machine_specific_config)
        self.run_config = model_to_dict(run_config)
        self.detector_config = model_to_dict(detector_config)
        self.dataset_config = model_to_dict(dataset_config)

        self.current_data_config = None

    def get_io_config(self):
        return self.run_config["io"]

    def get_video_and_comps_configs(self) -> Generator[Tuple[dict, dict], None, None]:
        """
        Iterates over all datasets, videos and return combined view of the video
        and components that need to be run on this video.
        """
        for dataset_name, dataset_dict in self.run_config["run"].items():
            # get all components specific for this dataset
            # we resolve actual components name with respect of the mapping
            component_dict = dict(
                (comp, self.run_config["component_algorithm_mapping"][comp])
                for comp in dataset_dict["components"]
            )
            cur_dataset_config = self.dataset_config[dataset_name]

            # next we iterate over each video marked for run
            for video in dataset_dict["videos"]:
                # TODO: we need to refactor setting it here and retrieving in another
                # function. This add a lot of confusion.
                self.runtime_ctx = {
                    "dataset_name": dataset_name,
                    "session_ID": video["session_ID"],
                    "sequence_ID": video["sequence_ID"],
                    "video_start": video["video_start"],
                    "video_length": video["video_length"],
                    "cam_face1": cur_dataset_config["cam_face1"],
                    "cam_face2": cur_dataset_config["cam_face2"],
                    "cam_top": cur_dataset_config["cam_top"],
                    "cam_front": cur_dataset_config["cam_front"],
                }
                # TODO: this is just horific flat config combined from random parts
                # it contains runtime fields in the root which breaks fields collision
                # of main config loader. I currently resolve it as is, but this need to
                # be properly refactored into data structure
                video_config = {
                    **self.runtime_ctx,
                    **cur_dataset_config,
                    **self.run_config["io"],
                    **self.machine_specific_config,
                }
                # by this point, everything should be resolved, except algo and comp
                # they will be resolved latter in main detectors loop
                runtime_placeholders = {"algorithm_name", "component_name"}
                loader = ConfigLoader(self.auto_placeholders, runtime_placeholders)
                video_config_res = loader.resolve(video_config)

                self.current_data_config = video_config_res
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

            localized_config = self.cfg_loader.resolve(
                method_config, runtime_ctx=self.runtime_ctx
            )
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
                run_config=self.run_config,
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
        # TODO: mark that it requires get_video_and_comps_configs first
        all_camera_names = set()
        detector_config = self.cfg_loader.resolve(
            self.detector_config, runtime_ctx=self.runtime_ctx
        )
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
