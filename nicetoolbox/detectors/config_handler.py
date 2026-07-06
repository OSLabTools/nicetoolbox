import copy
from pathlib import Path
from typing import Any, Generator

from ..configs.project_config_handler import ProjectConfigHandler
from ..configs.schemas.dataset_properties import DatasetConfig, DatasetProperties
from ..configs.schemas.detectors_config import DetectorsConfig
from ..configs.schemas.detectors_run_file import (
    DetectorsRunFile,
    LoggingLevelEnum,
    ResolvedSubsequenceMeta,
    RunConfigVideo,
)
from ..configs.schemas.experiment_config import CodeConfig, DetectorsExperimentConfig
from ..configs.schemas.machine_specific_paths import MachineSpecificConfig
from ..configs.schemas.predictions_mapping import PredictionsMappingConfig
from ..configs.utils import model_to_dict, resolve_filter
from ..utils.config import save_config
from .data import SequenceData
from .subsequence_context import SubsequenceContext


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

    def iter_sequence_contexts(self) -> Generator[SubsequenceContext, None, None]:
        """
        Iterate over all videos and yield frozen runtime configurations.

        Each yielded SequenceRuntimeConfig is fully resolved and immutable.
        It should be discarded after the video is processed.

        Yields:
            SequenceRuntimeConfig for each video defined in the run configuration
        """
        for dataset_name, videos_run_config in self.run_config.run.items():
            # Get dataset properties
            dataset_config = self.dataset_properties[dataset_name]

            for video in videos_run_config.videos:
                yield self._create_video_runtime_config(
                    dataset_name=dataset_name,
                    video=video,
                    dataset_config=dataset_config,
                )

    def _create_video_runtime_config(
        self,
        dataset_name: str,
        video: RunConfigVideo,
        dataset_config: DatasetConfig,
    ) -> SubsequenceContext:
        """
        Create a fully resolved, frozen SequenceRuntimeConfig.

        All placeholders are resolved before constructing the frozen model.
        """
        # Collect all camera names from the dataset's video tracks
        # We process all cameras all the time, no matter if any detector actually use them
        # This important for data consistency for visualizer and audio detectors
        all_camera_names = list(dataset_config.video.cameras.keys())
        all_track_names = list(dataset_config.audio.tracks.keys())

        # TODO: move it to some more general system for handling optional input block deps
        # for now it's hardcoded to specific attributes names
        # Resolve per-detector camera_names / track_names filters against the available tracks.
        resolved_detectors = copy.deepcopy(self.detectors_config)
        for algo in resolved_detectors.algorithms.values():
            if hasattr(algo, "camera_names"):
                algo.camera_names = resolve_filter(algo.camera_names, all_camera_names)
            if hasattr(algo, "track_names"):
                algo.track_names = resolve_filter(algo.track_names, all_track_names)

        # Construct frozen model with all resolved values
        runtime_config = SubsequenceContext(
            dataset_name=dataset_name,
            video_config=video,
            log_file=self.log_file,
            machine=self.machine_specific_config,
            run=self.run_config,
            dataset_properties=dataset_config,
            detectors_config=resolved_detectors,
            predictions_mapping=self.predictions_mapping,
        )
        # Build runtime context for this video
        runtime_ctx = {
            "cur_dataset_name": dataset_name,
            "cur_session_ID": video.session_ID,
            "cur_sequence_ID": video.sequence_ID,
            "cur_video_start": video.video_start,
            "cur_video_length": video.video_length,
        }
        # Resolve placeholders (rebuilds every nested model via model_validate)
        try:
            res_runtime = self.cfg_loader.resolve(runtime_config, runtime_ctx, ignore_auto_and_global=True)
        except (ValueError, KeyError) as e:
            raise type(e)(
                "Failed to resolve runtime config SequenceRuntimeConfig for "
                f"dataset='{dataset_name}', sequence='{video.sequence_ID}':\n{e}"
            ) from e

        return res_runtime

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
    def save_subsequence_meta(sequence_context: SubsequenceContext, data: SequenceData) -> None:
        """
        Build and persist per-sequence resolved facts to `subsequence_meta.toml`
        in the sequence's output sub-folder.

        Uses identifiers from the user-declared video config plus frame-resolved
        values measured by the video handler during data prep.
        """
        declared = sequence_context.video_config
        subsequence_meta = ResolvedSubsequenceMeta(
            session_ID=declared.session_ID,
            sequence_ID=declared.sequence_ID,
            video_start=data.video_start_frame_index,
            video_length=data.video_length_frames,
            fps=data.fps,
        )
        path = sequence_context.io.out_sub_folder / "subsequence_meta.toml"
        save_config(model_to_dict(subsequence_meta), path)

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
