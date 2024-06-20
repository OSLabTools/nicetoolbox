"""

"""
import copy
import logging
import os
import glob
import argparse
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from in_out import IO
from data import Data
import oslab_utils.logging_utils as log_ut

from method_detectors.gaze_individual.XGaze_3cams import XGaze3cams
from method_detectors.body_joints.mmpose import HRNetw48, VitPose
from method_detectors.facial_expression.emoca import Emoca
from method_detectors.active_speaker.speaking_detector import SPELL
from feature_detectors.kinematics.velocity_body import VelocityBody
from feature_detectors.proximity.body_distance import BodyDistance
from feature_detectors.leaning.body_angle import BodyAngle
from feature_detectors.gaze_interaction.gaze_distance import GazeDistance
import configs.config_handler as confh


all_method_detectors = dict(
    xgaze_3cams=XGaze3cams,
    hrnetw48=HRNetw48,
    vitpose=VitPose
    )

all_feature_detectors = dict(
    velocity_body=VelocityBody,
    body_distance=BodyDistance,
    gaze_distance=GazeDistance,
    body_angle=BodyAngle
    )


def main(run_config_file, detector_config_file, machine_specifics_file):
    # CONFIG I
    config_handler = confh.Configuration(run_config_file, detector_config_file, machine_specifics_file)

    # IO
    io = IO(config_handler.get_io_config())

    # LOGGING
    log_ut.setup_logging(*io.get_log_file_level())
    logging.info(f"\n{'#' * 80}\n\nISA-TOOL STARTED. Saving results to '{io.out_folder}'.\n\n{'#' * 80}\n\n")
    
    # CONFIG II
    # check config consistency
    #config_handler.check_config_consistency(
    #        io.get_output_folder('config', 'output'))

    # check and save experiment configs
    config_handler.checker()
    config_handler.save_experiment_config(io.get_config_file())

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # RUNNING
    for (dataset_config, component_dict) in config_handler.get_dataset_configs():
        logging.info(f"\n{'=' * 80}\nRUNNING dataset {dataset_config['dataset_name']} and "\
                     f"{dataset_config['session_ID']}.\n{'=' * 80}\n\n")
        algorithm_names = list(set(confh.flatten_list(list(component_dict.values()))))
        method_names = [alg for alg in algorithm_names if alg in all_method_detectors.keys()]
        feature_names = [alg for alg in algorithm_names if alg in all_feature_detectors.keys()]

        # IO
        io.initialization(
            dataset_config, config_handler.get_all_detector_names())

        # DATA preparation
        data = Data(
            dataset_config, io, 
            config_handler.get_all_input_data_formats(algorithm_names),
            config_handler.get_all_camera_names(algorithm_names)
            )

        # RUN detectors
        for (method_config, method_name) in config_handler.get_method_configs(method_names):
            logging.info(f"STARTING method '{method_name}'.\n{'-' * 80}")
            detector = all_method_detectors[method_name](method_config, io, data)
            detector.run_inference()
            if method_config['visualize']:
                detector.visualization(data)
            logging.info(f"FINISHED method '{method_name}'.\n\n")

        # RUN feature extractions pipeline
        for (feature_config, feature_name) in config_handler.get_feature_configs(feature_names):
            logging.info(f"STARTING feature '{feature_name}'.\n{'-' * 80}")
            feature = all_feature_detectors[feature_name](feature_config, io, data)
            feature_data = feature.compute()
            if feature_config['visualize']:
                feature.visualization(feature_data)
            logging.info(f"FINISHED feature '{feature_name}'.\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", default="configs/run_file.toml", type=str, required=False)
    parser.add_argument("--detectors_config", default="configs/detectors_config.toml", type=str, required=False)
    parser.add_argument("--machine_specifics", default="configs/machine_specific_paths.toml", type=str, required=False)
    args = parser.parse_args()

    main(args.run_config, args.detectors_config, args.machine_specifics)
