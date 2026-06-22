import glob
import os
from pathlib import Path

from ..configs.project_config_handler import ProjectConfigHandler
from ..configs.schemas.detectors_run_file import RunConfigVideo
from ..configs.schemas.experiment_config import DetectorsExperimentConfig
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..configs.schemas.visualizer_config import VisualizerConfig
from ..configs.utils import model_to_dict


class Configuration(ProjectConfigHandler):
    """
    Handles loading and resolving all configurations required for visualizer pipeline.
    """

    # Input paths
    machine_specific_path: Path
    visualizer_config_file_path: Path

    # Loaded configs
    machine_specific_config: MachineSpecificConfig

    def __init__(
        self,
        project_folder: Path,
        machine_specifics_file: Path,
        visualizer_config_file: Path,
        stats_only: bool = False,
    ):
        """
        Load all static configuration files.

        Args:
            project_folder (Path): Path to the project folder containing nice_project.toml.
            machine_specifics_file (Path): Path to machine_specific_paths.toml, may contain placeholders.
            visualizer_config_file (Path): Path to visualizer_config.toml, may contain placeholders.
            stats_only (bool): If True, only initialize statistics (skip media).
        """
        # initialize config loader, default placeholders and project config
        super().__init__(project_folder)

        # this paths we need to resolve manually, because they are external arguments
        self.machine_specific_path = self.cfg_loader.resolve(machine_specifics_file)
        self.visualizer_config_file_path = self.cfg_loader.resolve(visualizer_config_file)

        # machine specific config
        self.machine_specific_config = self.cfg_loader.load_config(self.machine_specific_path, MachineSpecificConfig)
        self.cfg_loader.extend_global_ctx(self.machine_specific_config)

        # visualizer config
        visualizer_config = self.cfg_loader.load_config(self.visualizer_config_file_path, VisualizerConfig)
        self.cfg_loader.extend_global_ctx(visualizer_config.io)

        # TODO: rest of the codebase except the configs as dict
        # so we convert them from models to configs
        # will be refactored soon
        self.machine_specific_config = model_to_dict(self.machine_specific_config)
        self.visualizer_config = model_to_dict(visualizer_config)

        if stats_only:
            self._initialize_statistics()
        else:
            self._initialize_media()

    def _initialize_statistics(self) -> None:
        self.nice_tool_out_folder = self.visualizer_config["io"]["nice_tool_output_folder"]

    def _initialize_media(self) -> None:
        # Load the latest config from the experiment output of nicetoolbox
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

        # Load the video config from the video/sequence output of nicetoolbox
        try:
            video_folder_path = os.path.join(
                self.visualizer_config["io"]["experiment_folder"], self.visualizer_config["io"]["video_name"]
            )

            video_config_file = glob.glob(os.path.join(video_folder_path, "*config*.toml"))[-1]
        except IndexError:
            print("\nCould not find the video config file in " f"{video_folder_path}\n\n")
            raise

        # load detectors expirement config
        # it should be already fully resolved except runtime placeholders
        # so we ignore global context and auto
        loaded_experiment_config = self.cfg_loader.load_config(
            Path(experiment_config_file),
            DetectorsExperimentConfig,
            ignore_auto_and_global=True,
        )

        # load video config
        loaded_video_config = self.cfg_loader.load_config(
            Path(video_config_file),
            RunConfigVideo,
            ignore_auto_and_global=True,
        )
        # TODO: rest of the codebase except the configs as dict
        # so we convert them from models to configs
        loaded_experiment_config = model_to_dict(loaded_experiment_config)
        loaded_video_config = model_to_dict(loaded_video_config)

        # verify that the visualizer project matches the experiment project
        exp_configs_folder = Path(loaded_experiment_config["project_config"]["configs_folder_path"])
        vis_configs_folder = self.project_config.configs_folder_path.resolve()
        if exp_configs_folder != vis_configs_folder:
            raise ValueError(
                f"Project mismatch: visualizer project configs_folder_path '{vis_configs_folder}' "
                f"differs from experiment project '{exp_configs_folder}'"
            )

        self.experiment_run_config = loaded_experiment_config["run_config"]
        self.experiment_detector_config = loaded_experiment_config["detector_config"]
        self.dataset_properties = loaded_experiment_config["dataset_config"]
        self.visualizer_config["predictions_mapping"] = loaded_experiment_config["predictions_mapping"]

        # get experiment properties
        self.dataset_name = self.visualizer_config["io"]["dataset_name"]

        # update visualizer config - which will be given to components
        self.visualizer_config["video"] = loaded_video_config
        # add properties of the dataset
        self.visualizer_config["dataset_properties"] = self.dataset_properties[self.dataset_name]

        algorithms_list = list(set(self.experiment_run_config["algorithms"]))
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

    def check_config(self):
        self._check_start_stop_frames()
        self._check_algorithms()

    def _check_start_stop_frames(self):
        video_length = self.visualizer_config["video"]["video_length"]
        # TODO: currently, we don't support validation check for str timestamps
        if isinstance(video_length, str):
            return

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

    def _check_algorithms(self):
        known_algorithms = set(self.experiment_detector_config["algorithms"].keys())
        for component in self.visualizer_config["media"]["visualize"]["components"]:
            algorithms = self.visualizer_config["media"][component]["algorithms"]
            for alg in algorithms:
                if alg not in known_algorithms:
                    raise ValueError(
                        f"Algorithm {alg} is not found in detectors config."
                        f"Delete or correct {alg} from Visualizer_config[media."
                        f"{component}] algorithms"
                    )
