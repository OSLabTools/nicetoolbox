"""
Parsing and handling of configuration files for evaluation.
Provides loop over evaluation tasks.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import toml
from pydantic import ValidationError

from ..utils import config as cfg
from ..utils.logging_utils import log_configs
from .config_schema import (
    AggregationConfig,
    DatasetProperties,
    DatasetPropertiesEvaluation,
    EvaluationConfig,
    GlobalEvalConfig,
    IOConfig,
    MetricTypeConfig,
    RunConfig,
)


class ConfigHandler:
    def __init__(self, eval_config_file: str, machine_specifics_file: str) -> None:
        """
        Handles loading and parsing of configuration files for evaluation.

        Args:
            eval_config_file (str): Path to the evaluation configuration TOML file.
            machine_specifics_file (str): Path to the machine-specifics TOML file.
        """
        # ---- (1) Load raw TOML files ----
        eval_config_path = Path(eval_config_file)
        machine_specifics_path = Path(machine_specifics_file)

        self.raw_evaluation_config = toml.load(eval_config_path)
        self.raw_machine_specifics = toml.load(machine_specifics_path)

        # ---- (2) Parse evaluation_config.toml related configs/dataclasses ----
        self.merged_config = {
            **self.raw_evaluation_config,
            **self.raw_machine_specifics,
        }
        self.io_config: IOConfig = self._parse_io_config()
        self.metric_type_configs: Dict[str, MetricTypeConfig] = (
            self._parse_metric_type_configs()
        )
        self.summaries_configs: Dict[str, AggregationConfig] = (
            self._parse_summaries_config()
        )
        self.global_settings: GlobalEvalConfig = self._parse_global_settings()

        # --- (3) Parse detector experiment run config ---
        # Load the latest run configuration from the experiment folder
        # It contains the run_config and dataset_properties
        self.experiment_run_config_raw = self._load_latest_experiment_config()

        # self.detector_config = self._parse_detector_config()
        self.component_algorithm_mapping = self._parse_component_algorithm_mapping()

        # Parsed from experiment run config_raw
        self.all_run_configs: Dict[str, RunConfig] = self._parse_run_configs()
        self.all_dataset_properties: Dict[str, DatasetProperties] = (
            self._parse_dataset_properties()
        )

    def _localize(
        self, config: Dict, fill_io: bool = True, fill_data: Optional[Dict] = None
    ) -> Dict:
        """
        Localize placeholders in the given config dictionary.
        The localization is done in multiple passes to ensure all placeholders are
        resolved correctly.

        Args:
            config (Dict): The configuration dictionary to localize.
            fill_io (bool): Whether to fill IO-related placeholders.
            fill_data (Optional[Dict]): Additional reference/baseline data for
                placeholder replacement.

        Returns:
            Dict: The localized configuration dictionary.
        """
        # 1) auto placeholders: git_hash, today, etc. (uses cfg.config_fill_auto)
        c1 = cfg.config_fill_auto(config)

        # 2) machine placeholders
        c2 = cfg.config_fill_placeholders(c1, self.raw_machine_specifics)

        # 3) IO placeholders (conditionally)
        if fill_io and self.io_config is not None:
            io_dict = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.io_config.__dict__.items()
            }
            c2 = cfg.config_fill_placeholders(c2, io_dict)

        # 4) Additional placeholders (e.g., merged_config, experiment_run_config_raw)
        if fill_data is not None:
            # Ensure fill_data values are strings if they are Path objects
            safe_fill_data = {
                k: str(v) if isinstance(v, Path) else v for k, v in fill_data.items()
            }
            c2 = cfg.config_fill_placeholders(c2, safe_fill_data)

        # 5) recursive fill (self-referential within the config being processed)
        c3 = cfg.config_fill_placeholders(c2, c2)
        return cfg.config_fill_placeholders(c3, c3)

    def _parse_global_settings(self) -> GlobalEvalConfig:
        """
        Parse the global evaluation settings from the merged configuration.

        Returns:
            GlobalEvalConfig: The parsed global evaluation settings dataclass.
        """
        # Extract global settings using the dataclass for validation/defaults
        try:
            return GlobalEvalConfig(
                **self.merged_config, metric_types=self.metric_type_configs
            )
        except ValidationError as e:
            logging.error(f"Error parsing GlobalEvalConfig: {e}")
            raise

    def _parse_io_config(self) -> IOConfig:
        """
        Parse the IO configuration from the merged configuration.

        Returns:
            IOConfig: The parsed IO configuration dataclass.
        """
        raw_io_cfg = self.merged_config.get("io", {})
        localized_io_cfg = self._localize(raw_io_cfg, fill_io=False)
        try:
            return IOConfig.model_validate(localized_io_cfg)
        except ValidationError as e:
            logging.error(f"Error parsing IOConfig: {e}", exc_info=True)
            raise

    def _parse_metric_type_configs(self) -> Dict[str, MetricTypeConfig]:
        """
        Parse the metric type configurations from the merged configuration.

        Returns:
            Dict[str, MetricTypeConfig]: A dictionary mapping metric type names to
                their configurations.
        """
        metric_types = dict()
        metric_type_configs_raw = self.merged_config.get("metrics", {})
        for metric_type, metric_type_config in metric_type_configs_raw.items():
            metric_type_config.update({"metric_type": metric_type})
            try:
                metric_types[metric_type] = MetricTypeConfig.model_validate(
                    metric_type_config
                )
            except ValidationError as e:
                logging.error(f"Error parsing MetricTypeConfig for {metric_type}: {e}")
        return metric_types

    def _parse_summaries_config(self) -> Dict[str, AggregationConfig]:
        """
        Parse the summaries configuration from the merged configuration.

        Returns:
            Dict[str, AggregationConfig]: A dictionary mapping summary names to
                their aggregation configurations.
        """
        summaries = dict()
        summaries_config_raw = self.merged_config.get("summaries", {})
        for summary_name, summary_config in summaries_config_raw.items():
            summary_config.update({"summary_name": summary_name})
            try:
                summaries[summary_name] = AggregationConfig.model_validate(
                    summary_config
                )
            except ValidationError as e:
                logging.error(f"Error parsing AggregationConfig of {summary_name}: {e}")
        return summaries

    def _load_latest_experiment_config(self) -> dict:
        """
        Load the latest experiment run configuration from the detector
        experiment folder.

        Returns:
            dict: The loaded experiment run configuration.
        """
        exp_folder = self.io_config.experiment_folder

        if not exp_folder.is_dir():
            logging.error(
                f"Experiment folder does not exist or is not a directory: {exp_folder}"
            )
            # Fallback or specific error handling might be needed if this is critical
            raise FileNotFoundError(f"Experiment folder not found: {exp_folder}")

        try:
            config_files = sorted(list(exp_folder.glob("config_*.toml")))
            if not config_files:
                logging.error(
                    f"No 'config_*.toml' files found in experiment folder: {exp_folder}"
                )
                raise RuntimeError(f"No 'config_*.toml' files found in {exp_folder}")

            latest_cfg_path = config_files[-1]
            logging.info(
                f"Loading latest experiment run configuration from: {latest_cfg_path}"
            )
            return toml.load(latest_cfg_path)

        except Exception as e:
            logging.error(f"Error loading latest experiment config: {e}")
            raise RuntimeError(f"Failed to load experiment config: {e}") from e

    def _parse_component_algorithm_mapping(self) -> Dict[str, str]:
        """
        Parse the component to algorithm mapping from the experiment run config.

        Returns:
            Dict[str, str]: A dictionary mapping component names to algorithm names.
        """
        return self.experiment_run_config_raw.get("run_config", {}).get(
            "component_algorithm_mapping", {}
        )

    def _parse_run_configs(self) -> Dict[str, RunConfig]:
        """
        Parse the run configurations from the experiment run config.

        Returns:
            Dict[str, RunConfig]: A dictionary mapping dataset names to their
                run configurations (Which components/videos have been processed).
        """
        run_configs = dict()
        run_config_raw = self.experiment_run_config_raw.get("run_config", {}).get(
            "run", {}
        )
        for dataset_name, dataset_run_config in run_config_raw.items():
            try:
                run_configs[dataset_name] = RunConfig(
                    dataset_name=dataset_name, **dataset_run_config
                )
            except ValidationError as e:
                logging.error(f"Error parsing run configurations: {e}")
                raise e
        return run_configs

    def _parse_dataset_properties(self) -> Dict[str, DatasetProperties]:
        """
        Parse the dataset properties from the experiment run config.

        Returns:
            Dict[str, DatasetProperties]: A dictionary mapping dataset names to their
                properties.
        """
        all_properties: Dict[str, DatasetProperties] = dict()
        all_dataset_properties_raw = self.experiment_run_config_raw.get(
            "dataset_config", {}
        )
        for name, props_dict_raw in all_dataset_properties_raw.items():
            try:
                # Base for localization:
                reference = {**self.merged_config, **props_dict_raw}
                props_localized = self._localize(props_dict_raw, fill_data=reference)
                # Create the DatasetProperties dataclass
                props_localized["dataset_name"] = name
                all_properties[name] = DatasetProperties.model_validate(props_localized)
            except ValidationError as e:
                logging.error(
                    f"Error parsing DatasetProperties for {name}: {e}."
                    " Skipping this dataset. This might be due to the"
                    " config saved inside the experiment folder."
                )
                continue
        return all_properties

    def get_combined_experiment_io_config(self) -> IOConfig:
        """
        Get the IO configuration, optionally including experiment-specific IO.

        Returns:
            IOConfig: The combined IO configuration dataclass.
        """
        combined_io_config = self.io_config.model_dump()
        experiment_io = self._localize(
            self.experiment_run_config_raw["run_config"]["io"]
        )
        combined_io_config.update({"experiment_io": experiment_io})
        return IOConfig.model_validate(combined_io_config)

    def save_experiment_config(self, output_folder: Path) -> None:
        """
        Save the effective configuration for the overall evaluation experiment run.
        This includes the evaluation config, run config, dataset properties,
        detector config, and machine-specific config.

        Args:
            output_folder (Path): The folder where the configuration will be saved.
        """
        log_configs(
            dict(
                machine_specific_config=self.raw_machine_specifics,
                io_config=self.get_combined_experiment_io_config().model_dump(),
                global_eval_config=self.global_settings.model_dump(),
                run_configs={
                    name: config.model_dump()
                    for name, config in self.all_run_configs.items()
                },
                dataset_properties={
                    name: props.model_dump()
                    for name, props in self.all_dataset_properties.items()
                },
                component_algorithm_mapping=self._parse_component_algorithm_mapping(),
            ),
            str(output_folder),
            file_name="config_<time>",
        )

    def get_evaluation_and_dataset_configs(self) -> Iterator[Tuple]:
        """
        Generator that yields tuples of (RunConfig, DatasetProperties, EvaluationConfig)
        for each dataset defined in the run configurations.

        Each tuple defines a task for the EvaluationEngine to process. This method is
        thus the main loop over datasets and their evaluation settings and tasks.

        Yields:
            Iterator[Tuple[RunConfig, DatasetProperties, EvaluationConfig]]:
                Tuples containing the run configuration, dataset properties, and
                evaluation configuration for each dataset.
        """
        for dataset_name, run_config in self.all_run_configs.items():
            if dataset_name not in self.all_dataset_properties:
                logging.error(
                    f"Dataset '{dataset_name}' from run_configs not found in"
                    " dataset_properties. Skipping."
                )
                continue

            # (1) Dataset properties and run config per dataset
            dataset_properties: DatasetProperties = self.all_dataset_properties[
                dataset_name
            ]

            # (2) Build evaluation configs based on dataset properties
            evaluation_entries: List[DatasetPropertiesEvaluation] = (
                dataset_properties.evaluation
            )

            # (3) Prepare evaluation config
            metric_types_dict = {}
            prediction_components = {}
            annotation_components = {}

            for eval_entry_obj in evaluation_entries:
                annot_components = eval_entry_obj.annotation_components

                for metric_type in eval_entry_obj.metric_types:
                    if metric_type not in self.metric_type_configs:
                        logging.warning(
                            f"Metric type '{metric_type}' for dataset '{dataset_name}'"
                            " not in global metric_type_configs. Skipping."
                        )
                        continue

                    metric_type_config = self.metric_type_configs[metric_type]
                    gt_required = metric_type_config.gt_required

                    # Add MetricTypeConfig to the dictionary
                    metric_types_dict[metric_type] = metric_type_config

                    if gt_required:
                        prediction_components[metric_type] = annot_components
                        annotation_components[metric_type] = annot_components
                    else:
                        gt_components = metric_type_config.gt_components
                        prediction_components[metric_type] = gt_components
                        annotation_components[metric_type] = ["none"]

            # (4) Finalize evaluation config
            evaluation_config = EvaluationConfig(
                device=self.global_settings.device,
                verbose=self.global_settings.verbose,
                batchsize=self.global_settings.batchsize,
                prediction_components=prediction_components,
                annotation_components=annotation_components,
                metric_types=metric_types_dict,
                component_algorithm_mapping=self.component_algorithm_mapping,
            )

            yield run_config, dataset_properties, evaluation_config
