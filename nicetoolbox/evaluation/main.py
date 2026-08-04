"""
Main script to run the NICE Toolbox evaluation.
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

from ..utils import logging_utils as log_ut
from ..utils.system import system_capability_check
from .config_handler import ConfigHandler
from .data.results_saver import save_results
from .metrics.audio.transcription_error_rate import TranscriptionErrorRateMetric
from .metrics.base_metric import BaseMetric
from .metrics.categorical.confusion_matrix import ConfusionMatrixMetric
from .metrics.categorical.pr_curve import PrCurveMetric
from .metrics.categorical.roc_auc import RocAucMetric
from .metrics.joints.bone_length import BoneLengthMetric
from .metrics.joints.distance_error import DistanceErrorMetric
from .metrics.joints.missing_points import MissingPointsMetric
from .metrics.joints.pck import PCKMetric

ALL_METRICS: dict[str, type[BaseMetric]] = dict(
    bone_length=BoneLengthMetric,
    distance_error=DistanceErrorMetric,
    missing_points=MissingPointsMetric,
    pck=PCKMetric,
    confusion_matrix=ConfusionMatrixMetric,
    pr_curve=PrCurveMetric,
    roc_auc=RocAucMetric,
    transcription_error_rate=TranscriptionErrorRateMetric,
)


def main(project_folder_path: Path, machine_specifics: Path, eval_config: Path) -> None:
    """Run the NICE Toolbox evaluation pipeline end-to-end.

    Args:
        project_folder_path (Path): Path to the project folder containing nice_project.toml.
        machine_specifics (Path): Path to machine_specific_paths.toml.
        eval_config (Path): Path to evaluation_config.toml
    """
    # initialize default console logging and perform system check
    log_ut.init_console_logging()
    system_capability_check()

    # load all configs
    config = ConfigHandler(project_folder_path, machine_specifics, eval_config)

    # create output folder for experiment storage
    output_dir = config.eval_config.output_folder
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # setup file logging based on config
    log_ut.init_file_logging(config.eval_config.log_file_path, config.eval_config.log_level)
    log_ut.log_main_banner("NICE TOOLBOX EVALUATION STARTED")
    logging.info(f"Project path: '{config.project_folder}'")
    logging.info(f"Machine specific path: '{config.machine_specific_path}'")
    logging.info(f"Evaluation config path: '{config.eval_config_file_path}'")
    logging.info(f"Log file saved at: '{config.eval_config.log_file_path}'")
    logging.info(f"Output path: '{output_dir}'")

    config_dump_path = config.save_experiment_config(output_dir)
    logging.info(f"Full config dump saved at: '{config_dump_path}'")

    log_ut.log_banner("Metrics calculation")
    for metric_name, metric_config in config.eval_config.metrics_to_run().items():
        metric_cls = ALL_METRICS[metric_config.metric_type]
        log_ut.log_with_underscore(f"RUNNING '{metric_name}' ({metric_config.metric_type})")

        start_time = time.time()
        metric = metric_cls(metric_config, config)
        res = metric.compute()
        save_results(res, output_dir)
        end_time = time.time() - start_time

        logging.info(f"FINISHED '{metric_name}' in {end_time:.3}s.\n\n")


def entry_point():
    """Entry point for the NICE Toolbox evaluation script."""
    parser = argparse.ArgumentParser(description="Run NICE Toolbox Evaluation")
    parser.add_argument(
        "--project_folder_path",
        default=Path("."),
        type=Path,
        required=False,
        help="Path to the NICE Toolbox project folder containing nice_project.toml config",
    )
    parser.add_argument(
        "--machine_specifics",
        default=Path("machine_specific_paths.toml"),
        type=Path,
        required=False,
        help="Path to machine_specific_paths.toml config",
    )
    parser.add_argument(
        "--eval_config",
        default=Path("<configs_folder_path>/evaluation_config.toml"),
        type=Path,
        required=False,
        help="Path to evaluation_config.toml, supports placeholders",
    )
    args = parser.parse_args()

    main(args.project_folder_path, args.machine_specifics, args.eval_config)


if __name__ == "__main__":
    entry_point()
