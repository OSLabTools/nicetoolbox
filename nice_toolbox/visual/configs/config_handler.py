import glob
import os

import utils.config as confh
import utils.filehandling as fh
import utils.visual_utils as vis_ut


class Configuration:
    def __init__(
        self,
        visualizer_config_file: str,
        machine_specifics_file: str,
        stats_only: bool = False,
    ):
        self.config_files = dict(
            visualizer_config_file=visualizer_config_file,
            machine_specifics_file=machine_specifics_file,
        )
        # load experiment config dicts - these can contain placeholders
        self.visualizer_config = fh.load_config(visualizer_config_file)
        self.machine_specific_config = fh.load_config(machine_specifics_file)
        if stats_only:
            self._initialize_statistics()
        else:
            self._initialize_media()

    def _initialize_statistics(self) -> None:
        self.nice_tool_out_folder = self._localize(self.visualizer_config)["io"][
            "nice_tool_output_folder"
        ]

    def _initialize_media(self) -> None:
        # load the latest config from the experiment output of nicetoolbox
        try:
            experiment_config_file = sorted(
                glob.glob(
                    os.path.join(
                        self._localize(self.visualizer_config)["io"][
                            "experiment_folder"
                        ],
                        "config_*.toml",
                    )
                )
            )[
                -1
            ]  # ! <---
        except IndexError:
            # ! Only loads latest config file, but in a single exp folder can be multiple runs with different datasets
            # ! If you want to visualize a dataset from a earlier run, this throws an error
            print(
                "\nCould not find the latest experiment config file in "
                f"{self._localize(self.visualizer_config)['io']['experiment_folder']}\n\n"
            )
            raise

        loaded_experiment_config = fh.load_config(experiment_config_file)
        self.experiment_run_config = loaded_experiment_config["run_config"]
        self.experiment_detector_config = loaded_experiment_config["detector_config"]
        self.dataset_properties = loaded_experiment_config["dataset_config"]

        # get experiment properties
        self.dataset_name = self.visualizer_config["media"]["dataset_name"]

        # update visualizer config - which will be given to components
        self.visualizer_config["video"] = self.experiment_run_config["run"][
            self.dataset_name
        ]["videos"][0]
        # add properties of the dataset
        self.visualizer_config["dataset_properties"] = self.dataset_properties[
            self.dataset_name
        ]

        algorithms_list = list(
            set(
                [
                    alg
                    for alg_list in self.experiment_run_config[
                        "component_algorithm_mapping"
                    ].values()
                    for alg in alg_list
                ]
            )
        )
        self.visualizer_config["algorithms_properties"] = {
            alg: alg_config
            for alg, alg_config in self.experiment_detector_config["algorithms"].items()
            if alg in algorithms_list
        }

    def _localize(self, config, fill_io=True, fill_data=False, dataset_name=None):
        # fill placeholders
        config = confh.config_fill_auto(config)
        config = confh.config_fill_placeholders(config, self.machine_specific_config)
        if fill_io:
            config = confh.config_fill_placeholders(config, self._get_io_config())
        if fill_data:
            config = confh.config_fill_placeholders(
                config, self.dataset_properties[dataset_name]
            )
        config = confh.config_fill_placeholders(config, config)
        return config

    def _get_io_config(self, add_exp=False):
        io_config = self._localize(self.visualizer_config["io"], fill_io=False)
        if add_exp:
            # add to the return config the NICE Toolbox experiment io
            io_config.update(
                experiment_io=self._localize(
                    self._localize(self.experiment_run_config["io"])
                )
            )
        return io_config

    def get_updated_visualizer_config(self):
        return self._localize(
            self.visualizer_config,
            fill_io=False,
            fill_data=True,
            dataset_name=self.dataset_name,
        )

    def get_isa_tool_out_folder(self):
        return self.nice_tool_out_folder

    def get_camera_names(self):
        # Extracting camera names
        camera_names = [
            value
            for key, value in self.visualizer_config["dataset_properties"].items()
            if (key.startswith("cam_")) & (value != "") & (type(value) is str)
        ]
        return camera_names

    def _get_camera_placeholders(self):
        # Extracting camera names
        camera_names = [
            key
            for key, value in self.visualizer_config["dataset_properties"].items()
            if (key.startswith("cam_")) & (value != "") & (type(value) is str)
        ]
        return camera_names

    def get_dataset_starting_index(self):
        return self.dataset_properties[self.dataset_name]["start_frame_index"]

    def check_calibration(self, calib, cam_name):
        if self.visualizer_config["media"]["visualize"]["camera_position"] == True:
            cam_matrix, cam_distor, cam_rotation, cam_extrinsic = (
                vis_ut.get_cam_para_studio(calib, cam_name)
            )
            if (cam_rotation == None) | (cam_extrinsic == None):
                assert ValueError(
                    "The rotation and extrinsic matrix of the camera could not found.\n"
                    "Please either change the Visualizer_config 'camera_position' parameter to false \n"
                    "or provide extrinsics parameters of the camera"
                )

    def check_config(self):
        self._check_start_stop_frames()
        self._check_component_name()
        self._check_algorithms()

    def _check_start_stop_frames(self):
        video_length = self.visualizer_config["video"]["video_length"]

        # check start frame
        if self.visualizer_config["media"]["visualize"]["start_frame"] < 0:
            raise ValueError(
                f"Visualizer_config 'start_frame' parameter cannot be negative."
            )
        elif self.visualizer_config["media"]["visualize"]["start_frame"] > video_length:
            raise ValueError(
                f"Visualizer_config 'start_frame' parameter cannot be greater than the video length. \n"
                f"Video length: {video_length} frames."
            )

        # check stop frame
        if self.visualizer_config["media"]["visualize"]["end_frame"] > video_length:
            raise ValueError(
                f"Visualizer_config 'end_frame' parameter cannot be greater than the video length. \n"
                f"Video length: {video_length} frames."
            )

        # check visualize interval
        if (
            self.visualizer_config["media"]["visualize"]["visualize_interval"]
            > video_length
        ):
            raise ValueError(
                f"Visualizer_config 'visualize_interval' parameter cannot be greater than the video length. \n"
                f"Video length: {video_length} frames."
            )

    def _check_component_name(self):
        component_algorithm_mapping = self.experiment_run_config[
            "component_algorithm_mapping"
        ]
        for component in self.visualizer_config["media"]["visualize"]["components"]:
            if component not in component_algorithm_mapping.keys():
                raise ValueError(
                    f"Component {component} is not found in run file.\n"
                    f"Delete or correct {component} from Visualizer_config[media.visualize.components]"
                )

    def _check_algorithms(self):
        component_algorithm_mapping = self.experiment_run_config[
            "component_algorithm_mapping"
        ]
        for component in self.visualizer_config["media"]["visualize"]["components"]:
            algorithms = self.visualizer_config["media"][component]["algorithms"]
            for alg in algorithms:
                if alg not in component_algorithm_mapping[component]:
                    raise ValueError(
                        f"Algorithm {alg} is not found in {component} run file."
                        f"Delete or correct {alg} from Visualizer_config[media.{component} algorithms"
                    )

    # Extract lists from config, check, and update if necessary
    def check_and_update_canvas(self):
        """
        Checks if the canvas list under a given section and key matches a condition,
        and updates the list if not.
        """
        for component in self.visualizer_config["media"]["visualize"]["components"]:
            a = self.visualizer_config["media"][component]
            for key, values in self.visualizer_config["media"][component][
                "canvas"
            ].items():
                removed_values = []
                for value in values:
                    if "cam" in value:
                        if value.strip("<>") not in self._get_camera_placeholders():
                            removed_values.append(value)
                if removed_values:
                    print(
                        f"WARNING: {removed_values} camera placeholders are not used in your dataset properties."
                        f"They will not be visualized. \n To avoid this warning, consider removing "
                        f"'{removed_values}' from the ['media'][{component}]['canvas'] list "
                        f"in the visualizer_config.toml file"
                    )
                    updated_list = [v for v in values if v not in removed_values]
                    self.visualizer_config["media"][component]["canvas"][
                        key
                    ] = updated_list

        return self.get_updated_visualizer_config()
