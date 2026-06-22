from pathlib import Path
from typing import Any, Generator, List

from ..configs.project_config_handler import ProjectConfigHandler
from ..configs.schemas.dataset_properties import DatasetProperties
from ..configs.schemas.detectors_config import DetectorsConfig
from ..configs.schemas.detectors_run_file import DetectorsRunFile, LoggingLevelEnum
from ..configs.schemas.experiment_config import CodeConfig, DetectorsExperimentConfig
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..configs.schemas.predictions_mapping import PredictionsMappingConfig
from ..configs.utils import model_to_dict
from ..configs.video_runtime_config import SequenceRuntimeConfig
from ..utils.config import save_config


def flatten_list(input_list) -> list[Any]:
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


class Configuration(ProjectConfigHandler):
    """
    Handles loading and resolving all configurations required for detectors pipeline. This includes:
    - machine specifics
    - project config
    - run configuration file
    - detectors configuration
    - dataset properties

    Further provides a config factory that produces frozen and resolved runtime configs per video context
    """

    # Input paths
    machine_specific_path: Path
    run_config_file_path: Path

    # Loaded configs
    machine_specific_config: MachineSpecificConfig
    run_config: DetectorsRunFile
    detectors_config: DetectorsConfig
    dataset_properties: DatasetProperties
    predictions_mapping: PredictionsMappingConfig

    def __init__(self, project_folder: Path, machine_specifics_file: Path, run_config_file: Path):
        """
        Load all static configuration files.

        Args:
            project_folder (Path): Path to the project folder containing nice_project.toml.
            machine_specifics_file (Path): Path to machine_specific_paths.toml, may contain placeholders.
            run_config_file (Path): Path to detectors_run_file.toml, may contain placeholders.
        """
        # initialize config handler for this project
        super().__init__(project_folder)

        # this paths we need to resolve manually, because they are external arguments
        self.machine_specific_path = self.cfg_loader.resolve(machine_specifics_file)
        self.run_config_file_path = self.cfg_loader.resolve(run_config_file)

        # start loading configs - order is import for placeholders dependency resolution
        # machine specific config
        self.machine_specific_config = self.cfg_loader.load_config(self.machine_specific_path, MachineSpecificConfig)
        self.cfg_loader.extend_global_ctx(self.machine_specific_config)
        # run file
        self.run_config = self.cfg_loader.load_config(self.run_config_file_path, DetectorsRunFile)
        self.cfg_loader.extend_global_ctx(self.run_config.io)
        # detectors config
        detectors_config_file = self.run_config.io.detectors_config
        self.detectors_config = self.cfg_loader.load_config(detectors_config_file, DetectorsConfig)
        # dataset config
        dataset_properties_file = self.run_config.io.dataset_properties
        self.dataset_properties = self.cfg_loader.load_config(dataset_properties_file, DatasetProperties)
        # predictions mapping
        predictions_mapping_file = self.run_config.io.predictions_mapping
        self.predictions_mapping = self.cfg_loader.load_config(predictions_mapping_file, PredictionsMappingConfig)

    # -------------------------------------------------------------------------
    # Factory Method for Video Runtime Configurations
    # -------------------------------------------------------------------------

    def iter_sequence_contexts(self) -> Generator[SequenceRuntimeConfig, None, None]:
        """
        Iterate over all videos and yield frozen runtime configurations.

        Each yielded SequenceRuntimeConfig is fully resolved and immutable.
        It should be discarded after the video is processed.

        Yields:
            SequenceRuntimeConfig for each video defined in the run configuration
        """
        for dataset_name, videos_run_config in self.run_config.run.items():
            # Get dataset properties
            dataset_props = self.dataset_properties[dataset_name]

            for video in videos_run_config.videos:
                yield self._create_video_runtime_config(
                    dataset_name=str(dataset_name),
                    video=video,
                    dataset_props=dataset_props,
                    algorithms=self.run_config.algorithms,
                )

    def _create_video_runtime_config(
        self,
        dataset_name: str,
        video,
        dataset_props,
        algorithms: List[str],
    ) -> SequenceRuntimeConfig:
        """
        Create a fully resolved, frozen SequenceRuntimeConfig.

        All placeholders are resolved before constructing the frozen model.
        """
        # Collect all camera names defined in dataset properties
        # We process all cameras all the time, no matter if any detector actually use them
        # This important for data consistency for visualizer and audio detectors
        cameras = {
            "cur_cam_face1": dataset_props.cam_face1,
            "cur_cam_face2": dataset_props.cam_face2,
            "cur_cam_top": dataset_props.cam_top,
            "cur_cam_front": dataset_props.cam_front,
        }
        all_camera_names = list(cameras.values())

        # Construct frozen model with all resolved values
        runtime_config = SequenceRuntimeConfig(
            log_level=self.log_level,
            log_file=self.log_file,
            dataset_name=dataset_name,
            video_config=video,
            dataset_properties=dataset_props,
            io=self.run_config.io,
            machine=self.machine_specific_config,
            detectors_config=self.detectors_config,
            predictions_mapping=self.predictions_mapping,
            algorithms=algorithms,
            all_camera_names=all_camera_names,
        )
        # Build runtime context for this video
        runtime_ctx = {
            "cur_dataset_name": dataset_name,
            "cur_session_ID": video.session_ID,
            "cur_sequence_ID": video.sequence_ID,
            "cur_video_start": video.video_start,
            "cur_video_length": video.video_length,
            **cameras,
        }
        # Resolve placeholders
        resolved_runtime = self.cfg_loader.resolve(runtime_config, runtime_ctx, ignore_auto_and_global=True)

        # TODO: nasty quickfix, remove empty camera names if they aren't available for this dataset
        # please add a better solution for camera handling without patching configs
        resolved_runtime.__dict__["all_camera_names"] = list(set(resolved_runtime.all_camera_names) - {""})
        for algo in resolved_runtime.detectors_config.algorithms.values():
            if hasattr(algo, "camera_names"):
                algo.camera_names = list(set(algo.camera_names) - {""})

        return resolved_runtime

    # -------------------------------------------------------------------------
    # Static Queries (don't depend on runtime context)
    # -------------------------------------------------------------------------

    def save_experiment_config(self, output_folder) -> None:
        # we save current auto_placeholders for reproduction purposes
        code_config = CodeConfig(**self.auto_placeholders)
        # save all experiment configurations
        config = DetectorsExperimentConfig(
            project_folder=self.project_folder,
            project_config_path=self.project_config_path,
            machine_specific_path=self.machine_specific_path,
            run_config_file_path=self.run_config_file_path,
            code_config=code_config,
            machine_specific_config=self.machine_specific_config,
            project_config=self.project_config,
            run_config=self.run_config,
            dataset_config=self.dataset_properties,
            detector_config=self.detectors_config,
            predictions_mapping=self.predictions_mapping,
        )
        save_config(model_to_dict(config), output_folder / f"config_{code_config.time}.toml")

    @staticmethod
    def save_video_config(video_config, output_folder) -> None:
        save_config(model_to_dict(video_config), output_folder / "video_config.toml")

    @property
    def visualize(self) -> bool:
        return self.run_config.visualize

    @property
    def save_csv(self) -> bool:
        return self.run_config.save_csv

    @property
    def error_level(self) -> str:
        return self.run_config.error_level

    @property
    def check_missing_detectors_dependencies(self) -> bool:
        return self.run_config.check_missing_detectors_dependencies

    @property
    def log_level(self) -> LoggingLevelEnum:
        return self.run_config.log_level

    @property
    def log_file(self) -> Path:
        return self.run_config.io.out_folder / "nicetoolbox.log"
