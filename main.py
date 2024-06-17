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
from oslab_utils.in_out import create_tmp_folder
from oslab_utils.annotations import CustomEaf

#from method_detectors.nodding.noddingpigeon import NoddingPigeon
#from method_detectors.gazeDistance.ETH_XGaze import ETHXGaze
from method_detectors.gaze.XGaze_3cams import XGaze3cams
from method_detectors.human_pose.mmpose import HRNetw48, VitPose
from method_detectors.facial_expression.emoca import Emoca
from method_detectors.active_speaker.speaking_detector import SPELL
from feature_detectors.kinematics.kinematics import Kinematics
from feature_detectors.proximity.proximity import Proximity
from feature_detectors.leaning.leaning import Leaning
from feature_detectors.gazeDistance.gazeDistance import GazeDistance
import configs.config_handler as confh
import oslab_utils.logging_utils as log_ut


all_methods = dict(
#        nodding_pigeon=NoddingPigeon,
#        ethXgaze=ETHXGaze,
        xgaze_3cams=XGaze3cams,
        hrnetw48=HRNetw48,
        vitpose=VitPose
#        emoca=Emoca,
#        active_speaker=SPELL
)

all_features = dict(velocity_body=Kinematics,
                    body_distance=Proximity,
                    gaze_distance=GazeDistance,
                    body_angle=Leaning)


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
                     f"{dataset_config['participant_ID']}.\n{'=' * 80}\n\n")
        algorithm_names = list(set(confh.flatten_list(list(component_dict.values()))))
        method_names = [alg for alg in algorithm_names if alg in all_methods.keys()]
        feature_names = [alg for alg in algorithm_names if alg in all_features.keys()]

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
            detector = all_methods[method_name](method_config, io, data)
            detector.run_inference()
            if method_config['visualize']:
                detector.visualization(data)
            logging.info(f"FINISHED method '{method_name}'.\n\n")

        # RUN feature extractions pipeline
        for (feature_config, feature_name) in config_handler.get_feature_configs(feature_names):
            logging.info(f"STARTING feature '{feature_name}'.\n{'-' * 80}")
            feature = all_features[feature_name](feature_config, io, data)
            feature_data = feature.compute()
            if feature_config['visualize']:
                feature.visualization(feature_data)
            logging.info(f"FINISHED feature '{feature_name}'.\n\n")


def sync_dataset_configs(filename='dataset_properties.toml'):
    def listdir_absolut_paths(path):
        return [name for name in sorted(os.listdir(path))
                if os.path.isdir(os.path.join(path, name))]

    def get_user_input(text, options):
        valid_user_input = False
        while not valid_user_input:
            user_input = input(text)
            valid_user_input = user_input in options

        return user_input

    machine_specifics_file = "configs/machine_specific_paths.toml"
    machine_config = confh.load_config(machine_specifics_file)
    local_dataset_path = machine_config['datasets_folder_path']
    remote_dataset_path = machine_config['remote_datasets_folder_path']

    local_dataset_names = set(listdir_absolut_paths(local_dataset_path))
    remote_dataset_names = set(listdir_absolut_paths(remote_dataset_path))
    dataset_names_cut = local_dataset_names.intersection(remote_dataset_names)

    for dataset_name in dataset_names_cut:
        print(f"\nSynchronizing dataset {dataset_name}...")
        remote_file = os.path.join(remote_dataset_path, dataset_name, filename)
        if not os.path.exists(remote_file):
            print(f"{dataset_name}: Found no {filename} file.")
            continue

        local_file = os.path.join(local_dataset_path, dataset_name, filename)
        copy_remote = False
        if not os.path.exists(local_file):
            print(f"{dataset_name}: File {filename} was found on the remote "
                  f"but not locally.")
            user_input = get_user_input("Copy it (y/n)?", ['y', 'n'])
            copy_remote = user_input == 'y'

        else:
            remote = confh.load_config(remote_file)
            local = confh.load_config(local_file)
            keys_differ, values_differ = confh.compare_configs(
                    remote, local, log_fct=print, config_names=filename)
            if keys_differ or values_differ:
                print(f"{dataset_name}: Detected differences in the remote "
                      f"and local {filename} files.")
                user_input = get_user_input("Update the local file (y/n)?", ['y', 'n'])
                copy_remote = user_input == 'y'
            else:
                print(f"{dataset_name}: remote & local files {filename} match.")

        if copy_remote:
            os.system(f'cp {remote_file} {os.path.dirname(local_file)}')

    print(f"\nSynchronization completed.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", default="configs/run_file.toml", type=str, required=False)
    parser.add_argument("--detectors_config", default="configs/detectors_config.toml", type=str, required=False)
    parser.add_argument("--machine_specifics", default="configs/machine_specific_paths.toml", type=str, required=False)
    args = parser.parse_args()

    #sync_dataset_configs()
    main(args.run_config, args.detectors_config, args.machine_specifics)
