import glob
import os

from ..configs.config_loader import ConfigLoader
from ..configs.schemas.experiment_config import DetectorsExperimentConfig
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..configs.schemas.visualizer_config import VisualizerConfig
from ..configs.utils import default_auto_placeholders, default_runtime_placeholders, model_to_dict
from ..utils import visual_utils as vis_ut


class Configuration:
    cfg_loader: ConfigLoader
    auto_placeholders: dict[str, str]
    runtime_placeholders: set[str]

    def __init__(
        self,
        visualizer_config_file: str,
        machine_specifics_file: str,
        stats_only: bool = False,
    ):
        # init config loader
        self.auto_placeholders = default_auto_placeholders()
        self.runtime_placeholders = default_runtime_placeholders()
        self.cfg_loader = ConfigLoader(self.auto_placeholders, self.runtime_placeholders)
        # machine specific
        machine_specific_config = self.cfg_loader.load_config(machine_specifics_file, MachineSpecificConfig)
        self.cfg_loader.extend_global_ctx(machine_specific_config)
        # visualizer config
        visualizer_config = self.cfg_loader.load_config(visualizer_config_file, VisualizerConfig)
        self.cfg_loader.extend_global_ctx(visualizer_config.io)

        # TODO: rest of the codebase except the configs as dict
        # so we convert them from models to configs
        # will be refactored soon
        self.machine_specific_config = model_to_dict(machine_specific_config)
        self.visualizer_config = model_to_dict(visualizer_config)

        if stats_only:
            self._initialize_statistics()
        else:
            self._initialize_media()

    def _initialize_statistics(self) -> None:
        self.nice_tool_out_folder = self.visualizer_config["io"]["nice_tool_output_folder"]

    def _initialize_media(self) -> None:
        # load the latest config from the experiment output of nicetoolbox
        try:
            experiment_config_file = sorted(
                glob.glob(
                    os.path.join(
                        self.visualizer_config["io"]["experiment_folder"],
                        "config_*.toml",
                    )
                )
            )[-1]  # ! <---
        except IndexError:
            # ! Only loads latest config file, but in a single exp folder can be
            # ! multiple runs with different datasets
            # ! If you want to visualize a dataset from a earlier run, this throws
            # ! an error
            print(
                "\nCould not find the latest experiment config file in "
                f"{self.visualizer_config['io']['experiment_folder']}\n\n"
            )
            raise

        # load detectors expirement config
        # it should be already fully resolved except runtime placeholders
        # so we ignore global context and auto
        loaded_experiment_config = self.cfg_loader.load_config(
            experiment_config_file,
            DetectorsExperimentConfig,
            ignore_auto_and_global=True,
        )
        # TODO: rest of the codebase except the configs as dict
        # so we convert them from models to configs
        loaded_experiment_config = model_to_dict(loaded_experiment_config)

        self.experiment_run_config = loaded_experiment_config["run_config"]
        self.experiment_detector_config = loaded_experiment_config["detector_config"]
        self.dataset_properties = loaded_experiment_config["dataset_config"]

        # get experiment properties
        self.dataset_name = self.visualizer_config["io"]["dataset_name"]

        # update visualizer config - which will be given to components
        self.visualizer_config["video"] = self.experiment_run_config["run"][self.dataset_name]["videos"][0]
        # add properties of the dataset
        self.visualizer_config["dataset_properties"] = self.dataset_properties[self.dataset_name]

        algorithms_list = list(
            set(
                [
                    alg
                    for alg_list in self.experiment_run_config["component_algorithm_mapping"].values()
                    for alg in alg_list
                ]
            )
        )
        self.visualizer_config["algorithms_properties"] = {
            alg: alg_config
            for alg, alg_config in self.experiment_detector_config["algorithms"].items()
            if alg in algorithms_list
        }

    def _get_io_config(self, add_exp=False):
        io_config = self.visualizer_config["io"]
        if add_exp:  # add to the return config the NICE Toolbox experiment io
            io_config["experiment_io"] = self.experiment_run_config["io"]
        return io_config

    def get_updated_visualizer_config(self):
        cur_dataset_config = self.dataset_properties[self.dataset_name]
        runtime_ctx = {
            "cur_cam_face1": cur_dataset_config["cam_face1"],
            "cur_cam_face2": cur_dataset_config["cam_face2"],
            "cur_cam_top": cur_dataset_config["cam_top"],
            "cur_cam_front": cur_dataset_config["cam_front"],
        }
        updated_visualizer_config = self.cfg_loader.resolve(
            self.visualizer_config, runtime_ctx, ignore_auto_and_global=True
        )
        return updated_visualizer_config

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
        if self.visualizer_config["media"]["visualize"]["camera_position"] is True:
            _, _, cam_rotation, cam_extrinsic = vis_ut.get_cam_para_studio(calib, cam_name)
            if (cam_rotation is None) | (cam_extrinsic is None):
                assert ValueError(
                    "The rotation and extrinsic matrix of the camera could not found.\n"
                    "Please either change the Visualizer_config 'camera_position' "
                    "parameter to false or provide extrinsics parameters of the camera"
                )

    def check_config(self, calibration_file):
        self._check_start_stop_frames()
        self._check_component_name()
        self._check_algorithms()
        self._check_camera_position(calibration_file)

    def _check_start_stop_frames(self):
        video_length = self.visualizer_config["video"]["video_length"]

        # check start frame
        if self.visualizer_config["media"]["visualize"]["start_frame"] < 0:
            raise ValueError("Visualizer_config 'start_frame' parameter cannot be negative.")

        if video_length == -1:
            return
        if self.visualizer_config["media"]["visualize"]["start_frame"] > video_length:
            raise ValueError(
                f"Visualizer_config 'start_frame' parameter cannot be greater than the "
                f"video length. \nVideo length: {video_length} frames."
            )

        # check stop frame
        if self.visualizer_config["media"]["visualize"]["end_frame"] > video_length:
            raise ValueError(
                f"Visualizer_config 'end_frame' parameter cannot be greater than the "
                f"video length. \nVideo length: {video_length} frames."
            )

        # check visualize interval
        if self.visualizer_config["media"]["visualize"]["visualize_interval"] > video_length:
            raise ValueError(
                f"Visualizer_config 'visualize_interval' parameter cannot be greater "
                f"than the video length. \nVideo length: {video_length} frames."
            )

    def _check_component_name(self):
        component_algorithm_mapping = self.experiment_run_config["component_algorithm_mapping"]
        for component in self.visualizer_config["media"]["visualize"]["components"]:
            if component not in component_algorithm_mapping:
                raise ValueError(
                    f"Component {component} is not found in run file.\n"
                    f"Delete or correct {component} from "
                    "Visualizer_config[media.visualize.components]"
                )

    def _check_algorithms(self):
        component_algorithm_mapping = self.experiment_run_config["component_algorithm_mapping"]
        for component in self.visualizer_config["media"]["visualize"]["components"]:
            algorithms = self.visualizer_config["media"][component]["algorithms"]
            for alg in algorithms:
                if alg not in component_algorithm_mapping[component]:
                    raise ValueError(
                        f"Algorithm {alg} is not found in {component} run file."
                        f"Delete or correct {alg} from Visualizer_config[media."
                        f"{component} algorithms"
                    )

    def _check_camera_position(self, calibration_file) -> None:
        """
        Checks the consistency of the camera position in the visualizer config.

        Raises:
            ValueError: If the camera position parameter is set to True but calibration
            parameters were not provided.
        """
        if (self.visualizer_config["media"]["visualize"]["camera_position"]) and (not calibration_file):
            raise ValueError(
                "ERROR: No valid calibration file is found. Visualization of camera "
                "position requires calibration data. Set camera_position to False "
                "in visualizer_config.toml\n"
            )
        return 0
