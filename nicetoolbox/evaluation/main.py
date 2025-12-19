"""
Main script to run the NICE Toolbox evaluation.
"""

import argparse
import cProfile
import logging
from pathlib import Path

from ..evaluation.auto_summaries import create_auto_summaries
from ..evaluation.results_wrapper.core import EvaluationResults
from ..utils.to_csv import results_to_csv
from .config_handler import ConfigHandler
from .engine import EvaluationEngine
from .in_out import IO


def main_evaluation_run(eval_config: str, machine_specifics: str) -> None:
    """Main function to set up and run the evaluation."""
    # Configure root logger and a common formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Initialize Configuration
    try:
        config_handler = ConfigHandler(eval_config, machine_specifics)
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}. Exiting.")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}. Exiting.")
        raise

    # Initialize IO manager
    io_manager = IO(
        config_handler.io_config,
        config_handler.experiment_io,
        config_handler.cfg_loader,
    )

    # Write log file to the output folder
    log_file_path: Path = io_manager.output_folder / "nice_evaluation.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if not config_handler.global_settings.skip_evaluation:
        logging.info(f"\n{'#' * 80}\n\nNICE TOOLBOX EVALUATION\n\n{'#' * 80}\n\n")
        logging.info(f"Using evaluation config: {eval_config}")
        logging.info(f"Using machine specifics: {machine_specifics}")
        logging.info(f"Output will be saved to base folder: {io_manager.output_folder}")
        logging.info(f"Running on device: {config_handler.global_settings.device}")

        # Save the effective configuration for the overall experiment run
        config_handler.save_experiment_config(io_manager.output_folder)

        # Instantiate and run the EvaluationEngine
        engine = EvaluationEngine(config_handler, io_manager)
        engine.run()

        logging.info("All evaluations have been successfully completed.")

        # Evaluation already done. Postprocessing: CSV export
        if config_handler.global_settings.verbose:
            logging.info("Converting results to CSV format.")
            results_to_csv(io_manager.get_out_folder(), io_manager.get_csv_folder())
            logging.info("CSV conversion completed.")

    # Evaluation already done. Postprocessing: Automatic summary creation
    if config_handler.global_settings.verbose:
        logging.info("Loading evaluation results for automatic summary reports.")
        try:
            results: EvaluationResults = EvaluationResults(
                root=io_manager.get_out_folder()
            )
            create_auto_summaries(io_manager, results, config_handler.summaries_configs)
            logging.info("Automatic summaries created and exported.")
        except Exception as err:
            logging.error(
                f"Automatic summary creation failed with error: {err}", exc_info=True
            )


def entry_point():
    """Entry point for the NICE Toolbox evaluation script."""
    parser = argparse.ArgumentParser(description="Run NICE Toolbox Evaluation")
    parser.add_argument(
        "--eval_config",
        default="configs/evaluation_config.toml",
        help="Path to evaluation config",
    )
    parser.add_argument(
        "--machine_specifics",
        default="machine_specific_paths.toml",
        help="Path to machine specifics config",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable memory profiling"
    )
    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        main_evaluation_run(args.eval_config, args.machine_specifics)
        profiler.disable()
        profiler.dump_stats("evaluation.prof")
    else:
        main_evaluation_run(args.eval_config, args.machine_specifics)


if __name__ == "__main__":
    entry_point()
