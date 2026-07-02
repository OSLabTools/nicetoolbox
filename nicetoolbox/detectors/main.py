"""
Run the NICE toolbox inference pipeline. The main script imports various modules and
classes to run method detectors and feature detectors on the provided datasets.
"""

import argparse
import logging
import time
from pathlib import Path

from nicetoolbox_core.errors import ErrorLevel

from ..configs.schemas.detectors_instances_configs import BaseAlgorithmConfig
from ..configs.schemas.detectors_run_file import LoggingLevelEnum
from ..detectors.base_detector import BaseDetector
from ..download_manager.manager import AssetManager
from ..utils import logging_utils as log_ut
from ..utils import to_csv as csv
from ..utils.dependency_sort import sort_detectors_order
from ..utils.error_handling import manage_error_scope
from ..utils.system import system_capability_check
from . import config_handler as confh
from .data import SequenceData
from .feature_detectors.gaze_interaction.gaze_distance import GazeDistance
from .feature_detectors.gaze_multiview.gaze_fusion import GazeFusion
from .feature_detectors.kinematics.velocity_body import VelocityBody
from .feature_detectors.proximity.body_distance import BodyDistance
from .in_out import SequenceIO
from .method_detectors.crisper_whisper.crisper_whisper_detector import CrisperWhisper
from .method_detectors.eth_xgaze.eth_xgaze_detector import EthXgaze
from .method_detectors.mmpose.mmpose_framework_2d import MMPose2D
from .method_detectors.mmpose.mmpose_framework_3d import MotionBERT
from .method_detectors.py_feat.py_feat import PyFeat
from .method_detectors.sam_3d_body.sam_3d_body_detector import Sam3dBody
from .method_detectors.spiga.spiga_detector import Spiga
from .method_detectors.whisperx.whisperx_detector import WhisperX

# set of all detectors implemented by NICE Toolbox
# put your detector class here to register it
DETECTOR_CLASSES: set[BaseDetector] = {
    MMPose2D,
    MotionBERT,
    EthXgaze,
    PyFeat,
    Spiga,
    WhisperX,
    CrisperWhisper,
    Sam3dBody,
    VelocityBody,
    BodyDistance,
    GazeDistance,
    GazeFusion,
}
ALL_DETECTORS = {cls.algorithm_type: cls for cls in DETECTOR_CLASSES}


def get_algo_components(algo_cfg: BaseAlgorithmConfig) -> list[str]:
    """Return the component list for an algorithm config.

    For configs that declare components per instance, reads from the config.
    For all other detectors, falls back to the class-level components attribute.
    """
    components = getattr(algo_cfg, "components", None)
    if not components:
        components = list(ALL_DETECTORS[algo_cfg.algorithm_type].components)
    return components


def main(project_folder_path: Path, machine_specifics_file: Path, run_config_file: Path):
    """
    Main entry point for the NICE Toolbox detectors pipeline.

    Args:
        project_folder_path (Path): Path to the project folder containing nice_project.toml.
        machine_specifics_file (Path): The path to the machine specifics file.
        run_config_file (Path): The path to the run configuration file.
    """
    # initialize default console logging and perform system check
    log_ut.init_console_logging()
    system_capability_check()

    # ==================================
    # PHASE 1: Load Static Configuration
    # ==================================
    config = confh.Configuration(project_folder_path, machine_specifics_file, run_config_file)

    error_level = ErrorLevel(config.error_level)
    log_level = LoggingLevelEnum(config.log_level)

    main_output_folder = config.run_config.io.out_folder
    main_output_folder.mkdir(parents=True, exist_ok=True)

    log_file = main_output_folder / "nicetoolbox.log"

    log_ut.init_file_logging(log_file, log_level.name)
    log_ut.log_main_banner(f"NICE TOOLBOX STARTED. Saving results to '{main_output_folder}'.")

    # asset download manager
    # checks user selected algorithms and download assets for it
    # some algorithms can be skipped (i.e. missing hf key)
    manager = AssetManager(config)
    runnable_algorithms = manager.ensure_assets_for_config()
    if not runnable_algorithms:
        log_ut.log_with_underscore("No valid algorithms remain after asset checks. Nothing to do.")
        return

    # update config file
    config.run_config.algorithms = runnable_algorithms
    # save experiment config for visualizer and evaluation
    config.save_experiment_config(main_output_folder)

    # ==========================
    # PHASE 2: Process Sequences
    # ==========================
    for sequence_context in config.iter_sequence_contexts():  # for each sequence
        # get sequence meta information for logging
        sequence_str = str(sequence_context.video_config)
        log_ut.log_banner(f"RUNNING {sequence_str}")
        with manage_error_scope(error_level, ErrorLevel.SEQUENCE, sequence_str):
            # Create IO paths specific for the sequence
            io = SequenceIO(sequence_context)
            # Init sequences data and dump resolved info (actual fps, number of frames, etc.)
            data = SequenceData(sequence_context, io)
            config.save_subsequence_meta(sequence_context, data)

            # Topologically sort the selected algorithms by their dependencies
            ordered_detectors = sort_detectors_order(
                sequence_context.detectors_config, runnable_algorithms, config.check_missing_detectors_dependencies
            )

            # ======================
            # PHASE 3: RUN DETECTORS
            # ======================
            for algorithm_instance in ordered_detectors:  # for each detector instance
                with manage_error_scope(error_level, ErrorLevel.DETECTOR, algorithm_instance):
                    log_ut.log_with_underscore(f"STARTING '{algorithm_instance}'.")
                    start_time = time.time()

                    cfg = sequence_context.detectors_config.algorithms[algorithm_instance]
                    detector_class = ALL_DETECTORS[cfg.algorithm_type]
                    detector = detector_class(io, data, sequence_context, algorithm_instance)

                    result_data = detector.run()

                    if config.visualize and detector.visualize:
                        detector.visualization(result_data)

                    logging.info(f"FINISHED '{algorithm_instance}' in {time.time() - start_time}s.\n\n")

            # Convert results to .csv
            if config.save_csv:
                csv.results_to_csv(io.out_sub_folder, io.csv_folder)
                logging.info("Converting current sequence results to CSV successful.")

    log_ut.log_with_underscore("Detectors finished.")


def entry_point():
    """Entry point for running NICE toolbox detectors."""
    parser = argparse.ArgumentParser()
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
        "--run_config",
        default="<configs_folder_path>/detectors_run_file.toml",
        type=Path,
        required=False,
        help="Path to detectors_run_file.toml, supports placeholders",
    )
    args = parser.parse_args()

    main(args.project_folder_path, args.machine_specifics, args.run_config)


if __name__ == "__main__":
    entry_point()
