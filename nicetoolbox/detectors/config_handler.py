import copy
import fnmatch
from pathlib import Path
from typing import Any, Generator

from ..configs.project_config_handler import ProjectConfigHandler
from ..configs.schemas.dataset_properties import DatasetProperties, SequenceConfig
from ..configs.schemas.detectors_config import DetectorsConfig
from ..configs.schemas.detectors_run_file import (
    DetectorsRunFile,
    LoggingLevelEnum,
    ResolvedSubsequenceConfig,
    SubsequenceConfig,
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

        # Expand wildcard sequence_ids in run config against known dataset sequences.
        # Mutates self.run_config.run in place, so downstream iteration and the saved
        # experiment config both see the fully-expanded sequence list.
        self._expand_sequence_wildcards()

    def _expand_sequence_wildcards(self) -> None:
        """
        Expand any sequence entries in the run config whose `sequence_id` contains a
        glob wildcard (e.g. `*` for all sequences, `S1_*` for sequences starting with
        `S1_`) against the sequences declared in dataset properties.

        Non-wildcard entries pass through unchanged. Duplicates (from overlapping
        patterns or explicit entries) are deduplicated, keeping the first occurrence.
        Raises if a pattern matches no sequences or references an unknown dataset.
        """
        for dataset_name, run_ds in self.run_config.run.items():
            if dataset_name not in self.dataset_properties:
                raise ValueError(
                    f"Run config dataset '{dataset_name}' not found in dataset properties. "
                    f"Available: {list(self.dataset_properties.keys())}"
                )
            all_ids = [seq.sequence_id for seq in self.dataset_properties[dataset_name].sequences]

            expanded: list[SubsequenceConfig] = []
            seen_ids: set[str] = set()
            for entry in run_ds.sequences:
                if any(ch in entry.sequence_id for ch in "*?["):
                    matched = [sid for sid in all_ids if fnmatch.fnmatchcase(sid, entry.sequence_id)]
                    if not matched:
                        raise ValueError(
                            f"Wildcard sequence_id '{entry.sequence_id}' in dataset "
                            f"'{dataset_name}' matched no sequences. Available: {all_ids}"
                        )
                    for sid in matched:
                        if sid in seen_ids:
                            continue
                        seen_ids.add(sid)
                        expanded.append(entry.model_copy(update={"sequence_id": sid}))
                else:
                    if entry.sequence_id in seen_ids:
                        continue
                    seen_ids.add(entry.sequence_id)
                    expanded.append(entry)

            run_ds.sequences = expanded

    # -------------------------------------------------------------------------
    # Factory Method for Video Runtime Configurations
    # -------------------------------------------------------------------------

    def iter_sequence_contexts(self) -> Generator[SubsequenceContext, None, None]:
        """
        Iterate over all configured sequences and yield frozen runtime configurations.

        Each yielded SubsequenceContext is fully resolved and immutable.
        It should be discarded after the sequence is processed.

        Yields:
            SubsequenceContext for each sequence resolved from the run configuration
        """
        for dataset_name, run_ds in self.run_config.run.items():
            sequences_by_id = {seq.sequence_id: seq for seq in self.dataset_properties[dataset_name].sequences}

            for run_seq in run_ds.sequences:
                if run_seq.sequence_id not in sequences_by_id:
                    raise ValueError(
                        f"Run config sequence_id '{run_seq.sequence_id}' in dataset "
                        f"'{dataset_name}' not found. Available: {list(sequences_by_id)}"
                    )
                yield self._create_video_runtime_config(
                    dataset_name=dataset_name,
                    run_sequence=run_seq,
                    sequence_properties=sequences_by_id[run_seq.sequence_id],
                )

    def _create_video_runtime_config(
        self,
        dataset_name: str,
        run_sequence: SubsequenceConfig,
        sequence_properties: SequenceConfig,
    ) -> SubsequenceContext:
        """
        Create a fully resolved, frozen SubsequenceContext.

        All placeholders are resolved before constructing the frozen model.
        """
        # Collect all camera names from the sequence's video tracks
        # We process all cameras all the time, no matter if any detector actually use them
        # This important for data consistency for visualizer and audio detectors
        all_camera_names = list(sequence_properties.video.cameras.keys())
        all_track_names = list(sequence_properties.audio.tracks.keys())

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
            run_sequence=run_sequence,
            sequence_properties=sequence_properties,
            log_file=self.log_file,
            machine=self.machine_specific_config,
            run=self.run_config,
            detectors_config=resolved_detectors,
            predictions_mapping=self.predictions_mapping,
        )
        # Build runtime context for this sequence
        runtime_ctx = {
            "cur_dataset_name": dataset_name,
            "cur_sequence_id": run_sequence.sequence_id,
            "cur_video_start": run_sequence.video_start,
            "cur_video_length": run_sequence.video_length,
        }
        # Resolve placeholders (rebuilds every nested model via model_validate)
        try:
            res_runtime = self.cfg_loader.resolve(runtime_config, runtime_ctx, ignore_auto_and_global=True)
        except (ValueError, KeyError) as e:
            raise type(e)(
                "Failed to resolve runtime config SubsequenceContext for "
                f"dataset='{dataset_name}', sequence='{run_sequence.sequence_id}':\n{e}"
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
        declared = sequence_context.run_sequence
        subsequence_meta = ResolvedSubsequenceConfig(
            sequence_id=declared.sequence_id,
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
