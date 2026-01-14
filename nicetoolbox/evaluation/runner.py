"""
Dataset runner for executing evaluations on a single dataset.
"""

import logging
from typing import List

from torch.utils.data import DataLoader

from ..configs.schemas.dataset_properties import DatasetConfig
from ..configs.schemas.detectors_run_file import DetectorsRunConfig
from .config_schema import FinalEvaluationConfig
from .data.dataset import EvaluationDataset
from .data.discovery import ChunkWorkItem, DiscoveryEngine
from .data.loaders import AnnotationLoader, PredictionLoader
from .in_out import IO
from .metrics.evaluate import EvalResults, MetricRunner


class DatasetRunner:
    """
    Runner for evaluating a single dataset with given configurations.
    """

    def __init__(
        self,
        io_manager: IO,
        run_config: DetectorsRunConfig,
        dataset_properties: DatasetConfig,
        evaluation_config: FinalEvaluationConfig,
    ) -> None:
        """Initialize the DatasetRunner.

        Args:
            io_manager (IO): IO manager for file operations.
            run_config (RunConfig): Configuration for the current run.
            dataset_properties (DatasetProperties): Properties of the dataset.
            evaluation_config (EvaluationConfig): Current evaluation task config.
        """
        self.io_manager = io_manager
        self.run_config = run_config
        self.dataset_properties = dataset_properties
        self.evaluation_config = evaluation_config
        self.dataset_name = self.dataset_properties._dataset_name

    def run(self) -> None:
        """
        Execute the evaluation process for the dataset.

        This includes:
        1. Initializing the dataset in the IO manager.
        2. Discovering work items using the DiscoveryEngine.
        3. Setting up the EvaluationDataset and DataLoader.
        4. Running the MetricRunner to compute metrics.
        5. Saving the evaluation results using the IO manager.
        """
        logging.info(f"Starting run for dataset: {self.dataset_name}")
        self.io_manager.init_dataset(self.dataset_properties)

        # === (1) Discover work items ===
        try:
            discovery_engine = DiscoveryEngine(
                io_manager=self.io_manager,
                run_config=self.run_config,
                dataset_properties=self.dataset_properties,
                evaluation_config=self.evaluation_config,
            )
            work_items: List[ChunkWorkItem] = discovery_engine.discover_work_items()
            if not work_items:
                logging.error(f"No work items found for dataset {self.dataset_name}.")
                return
        except Exception as e:
            logging.error(
                f"Error during discovery for {self.dataset_name}: {e}. Skipping.",
                exc_info=True,
            )
            return

        # === (2) Initialize dataset and loaders ===
        try:
            pred_loader = PredictionLoader()
            annot_loader = None
            gt_needed = any(cfg.gt_required for cfg in self.evaluation_config.metric_types.values())
            if gt_needed:
                annot_loader = AnnotationLoader(self.io_manager.path_to_annotations)

            dataset = EvaluationDataset(
                work_items=work_items,
                prediction_loader=pred_loader,
                annotation_loader=annot_loader,
            )
            loader = DataLoader(
                dataset,
                batch_size=self.evaluation_config.batchsize,
                shuffle=False,
                collate_fn=EvaluationDataset.collate_fn,
            )
        except Exception as e:
            logging.error(
                f"Error initializing dataset {self.dataset_name}: {e}. Skipping.",
                exc_info=True,
            )
            return

        # === (3) Run metrics ===
        try:
            metric_runner = MetricRunner(loader=loader, eval_cfg=self.evaluation_config)
            eval_results: EvalResults = metric_runner.evaluate()
            eval_results.save(self.io_manager)
        except Exception as e:
            logging.error(
                f"Error running metric for dataset {self.dataset_name}: {e}. Skipping.",
                exc_info=True,
            )
        finally:
            pred_loader.close_files()
            if annot_loader:
                annot_loader.close_files()

        logging.info(f"Finished run for dataset: {self.dataset_name}")
