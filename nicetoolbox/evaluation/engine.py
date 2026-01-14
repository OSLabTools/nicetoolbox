"""
Evaluation engine to manage the evaluation process across multiple datasets.
"""

import logging

from ..configs.schemas.dataset_properties import DatasetConfig
from ..configs.schemas.detectors_run_file import DetectorsRunConfig
from .config_handler import ConfigHandler
from .config_schema import FinalEvaluationConfig
from .in_out import IO
from .runner import DatasetRunner


class EvaluationEngine:
    def __init__(self, config_handler: ConfigHandler, io_manager: IO) -> None:
        """
        Initialize the EvaluationEngine with configuration and IO manager.
        """
        self.config_handler = config_handler
        self.io_manager = io_manager  # Global IO manager

    def run(self) -> None:
        """
        Run the evaluation process.

        This includes:
        1. Main loop over configurations
        2. Calls dataset-specific runner.
        """
        logging.info("EvaluationEngine: Starting processing all datasets.")

        for (
            run_config_instance,
            dataset_properties_instance,
            evaluation_config_instance,
        ) in self.config_handler.get_evaluation_and_dataset_configs():
            run_config: DetectorsRunConfig = run_config_instance
            dataset_properties: DatasetConfig = dataset_properties_instance
            evaluation_config: FinalEvaluationConfig = evaluation_config_instance

            dataset_name = dataset_properties._dataset_name
            logging.info(f"\n\n{'-' * 80}\n" f"Processing dataset: {dataset_name.upper()}" f"\n{'-' * 80}\n\n")

            runner = DatasetRunner(
                io_manager=self.io_manager,
                run_config=run_config,
                dataset_properties=dataset_properties,
                evaluation_config=evaluation_config,
            )

            try:
                runner.run()
            except Exception as e:
                logging.error(f"Error processing dataset {dataset_name}: {e}", exc_info=True)
                logging.warning(f"Skipping to next dataset due to error in {dataset_name}.")
                continue

        logging.info("EvaluationEngine: Finished processing all datasets.")
