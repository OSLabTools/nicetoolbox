"""
Run the NICE toolbox inference pipeline. The main script imports various modules and
classes to run method detectors and feature detectors on the provided datasets.
"""

import argparse
import logging
import time

from ..utils import logging_utils as log_ut
from . import config_handler as confh
from .data import Data
from .feature_detectors.gaze_interaction.gaze_distance import GazeDistance
from .feature_detectors.kinematics.velocity_body import VelocityBody
from .feature_detectors.leaning.body_angle import BodyAngle
from .feature_detectors.proximity.body_distance import BodyDistance
from .in_out import IO
from .method_detectors.body_joints.mmpose_framework import HRNetw48, VitPose
from .method_detectors.emotion_individual.py_feat import PyFeat
from .method_detectors.gaze_individual.XGaze_3cams import XGaze3cams
from .result_conversion import to_csv as csv

all_method_detectors = dict(
    xgaze_3cams=XGaze3cams, hrnetw48=HRNetw48, vitpose=VitPose, py_feat=PyFeat
)


all_feature_detectors = dict(
    velocity_body=VelocityBody,
    body_distance=BodyDistance,
    gaze_distance=GazeDistance,
    body_angle=BodyAngle,
)


def main(run_config_file, machine_specifics_file):
    """
    The main function of the NICE Toolbox.

    Args:
        run_config_file (str): The path to the run configuration file.
        detector_config_file (str): The path to the detector configuration file.
        machine_specifics_file (str): The path to the machine specifics file.

    This function is the entry point of the NICE toolbox. It performs the following
    steps:

    1. Initializes the configuration handler with the provided configuration files.
    2. Initializes the IO module with the IO configuration from the configuration
        handler.
    3. Sets up logging and logs the start of the NICE toolbox.
    4. Checks the configuration consistency and saves the experiment configuration.
    5. Runs the datasets specified in the configuration.
    6. For each dataset, initializes the IO module and prepares the data.
    7. Runs the method detectors specified in the configuration for each dataset.
    8. Runs the feature extraction pipeline specified in the configuration for each
        dataset.
    """
    # CONFIG I
    config_handler = confh.Configuration(run_config_file, machine_specifics_file)

    # IO
    io = IO(config_handler.get_io_config())

    # LOGGING
    log_ut.setup_logging(*io.get_log_file_level())
    logging.info(
        f"\n{'#' * 80}\n\nNICE TOOLBOX STARTED. Saving results to "
        f"'{io.out_folder}'.\n\n{'#' * 80}\n\n"
    )

    # check and save experiment configs
    config_handler.checker()
    config_handler.save_experiment_config(io.get_config_file())

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # RUNNING
    for dataset_config, component_dict in config_handler.get_dataset_configs():
        logging.info(
            f"\n{'=' * 80}\nRUNNING dataset {dataset_config['dataset_name']} and "
            f"{dataset_config['session_ID']}.\n{'=' * 80}\n\n"
        )
        algorithm_names = list(set(confh.flatten_list(list(component_dict.values()))))
        method_names = [alg for alg in algorithm_names if alg in all_method_detectors]
        feature_names = [alg for alg in algorithm_names if alg in all_feature_detectors]

        # IO
        io.initialization(dataset_config, config_handler.get_all_detector_names())

        # DATA preparation
        data = Data(
            dataset_config,
            io,
            config_handler.get_all_input_data_formats(algorithm_names),
            config_handler.get_all_camera_names(algorithm_names),
            config_handler.get_all_dataset_names(),
        )

        # RUN method detectors
        for method_config, method_name in config_handler.get_method_configs(
            method_names
        ):
            start_time = time.time()
            logging.info(f"STARTING method '{method_name}'.\n{'-' * 80}")
            detector = all_method_detectors[method_name](method_config, io, data)
            detector.run_inference()
            if method_config["visualize"]:
                detector.visualization(data)
            logging.info(
                f"FINISHED method '{method_name}' in {time.time() - start_time}s.\n\n"
            )

        # RUN feature detectors
        for feature_config, feature_name in config_handler.get_feature_configs(
            feature_names
        ):
            start_time = time.time()
            logging.info(f"STARTING feature '{feature_name}'.\n{'-' * 80}")
            feature = all_feature_detectors[feature_name](feature_config, io, data)
            feature_data = feature.compute()
            if feature_config["visualize"]:
                feature.visualization(feature_data)
            logging.info(
                f"FINISHED feature '{feature_name}' in {time.time() - start_time}s.\n\n"
            )

    # convert results
    logging.info(f"Detectors finished.\n{'-' * 80}")
    if config_handler.save_csv():
        logging.info("START converting results to CSV-files.")
        csv.results_to_csv(io.get_output_folder("main"), io.get_output_folder("csv"))
        logging.info("FINISHED converting results to CSV-files.")


def entry_point():
    """Entry point for running NICE toolbox detectors."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_config",
        default="configs/detectors_run_file.toml",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--machine_specifics",
        default="machine_specific_paths.toml",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    main(args.run_config, args.machine_specifics)
