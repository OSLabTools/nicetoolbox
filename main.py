"""

"""
import copy
import logging
import os
import subprocess
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from in_out import IO
from data import Data
from oslab_utils.in_out import create_tmp_folder
from oslab_utils.annotations import CustomEaf

#from detectors.nodding.noddingpigeon import NoddingPigeon
#from detectors.gazeDistance.ETH_XGaze import ETHXGaze
from detectors.gaze.XGaze_3cams import XGaze3cams
from detectors.human_pose.pose_detector import PoseDetector
from detectors.facial_expression.emoca import Emoca
from features.kinematics.kinematics import Kinematics
from features.proximity.proximity import Kinematics as Proximity
from features.gazeDistance.gazeDistance import GazeDistance
from configs.config_handler import Configuration
import oslab_utils.logging_utils as log_ut


all_methods = dict(
#        nodding_pigeon=NoddingPigeon,
#        ethXgaze=ETHXGaze,
        xgaze_3cams=XGaze3cams,
        mmpose=PoseDetector,
        emoca=Emoca,
)

all_features = dict(kinematics=Kinematics,
                    proximity=Proximity,
                    gazeDistance=GazeDistance)

# def run(settings, Model, data, eaf):
#     """Main function to run the method
#
#     Parameters
#     ----------
#     settings : _type_
#         _description_
#
#     Returns
#     -------
#     _type_
#         _description_
#     """
#
#     # create method instance
#     Detector = Model(settings)
#
#     # not allow to create multiple annotations with equal (author, annotator,
#     # tier) as pympi.Elan.Eaf overrides existing annotations while adding new
#     # timeslots -> unwanted behavior
#     assert not (eaf.adocument['AUTHOR'] == settings['git_hash'] and
#             Detector.behavior in eaf.get_tier_names() and
#             eaf.get_parameters_for_tier(Detector.behavior)['ANNOTATOR']
#             == Detector.name), \
#         f"for author/git-hash '{settings['git_hash']}', " \
#         f"annotator/method '{Detector.name}' has annotated " \
#         f"tier/behavior '{Detector.behavior}' already."
#
#     # run method
#     detections = Detector.inference(data)
#
#     ### save results
#     # add tier for nodding detection
#     eaf.add_detection(detections['values'], Detector.name)
#
#     # save eaf instance to file
#     eaf.save()
#
#     print(f"\n\nSuccessfully ran '{Detector.name}'! \n"
#           f"\tSaved results to '{eaf.out_file}'.\n\n")


def flatten_dict(dictionary):
    output_dict = copy.deepcopy(dictionary)
    for key, value in dictionary.items():
        if isinstance(value, dict):
            del output_dict[key]
    return output_dict


def main():
    # CONFIG
    config_abstract_file = "configs/config.toml"
    machine_specifics_file = "configs/machine_specific_paths.toml"
    config_handler = Configuration(config_abstract_file, machine_specifics_file)
    config = config_handler.get_localized_config()

    # # ANNOTATIONS
    # # create an ELAN annotation file - an EAF file
    # eaf = CustomEaf(config['git_hash'],
    #                 config['video_file'],
    #                 config['out_folder'])
    # # run validity checks that these annotations were not done before

    # IO
    io = IO(config['io'],
            config['methods']['names'] + config['features']['names'] +
            [config['features'][name]['input_detector_names']
             for name in config['features']['names']])

    # save experiment configs
    config_handler.save_experiment_config(
            io.get_output_folder('config', 'output'))

    #initialize log
    log_path = os.path.join(io.get_output_folder('config', 'output'), "ISA-Tool.log")
    log_ut.setup_logging(log_path, level=logging.INFO)

    # -> clean up all /tmp/ even if the code crashes or triggers an assertion
    with create_tmp_folder(io.get_all_tmp_folders()):

        # DATA preparation
        data = Data(config, io)

        # RUN detectors
        for method_name in config['methods']['names']:
            # prepare the part of the config relevant for the detector
            method_config = flatten_dict(config['methods'][method_name])
            if 'algorithm' in method_config.keys():
                method_config.update(config['methods'][method_name][method_config['algorithm']])
            method_config["video_start"] = config["video_start"]
            detector = all_methods[method_name](method_config, io, data)
            inference_returncode = detector.run_inference()
            log_ut.assert_and_log(inference_returncode == 0, f"INFERENCE Pipeline {method_name} - FAILURE - See method log for details")
            logging.info(f"INFERENCE Pipeline {method_name} - SUCCESS - See method log for details")
            detector.visualization(data)

        # RUN feature extractions pipeline
        for feature_name in config['features']['names']:
            # prepare the part of the config relevant for the feature
            feature_config = copy.deepcopy(config['features'][feature_name])
            feature = all_features[feature_name](feature_config, io)
            feature_data = feature.compute()
            feature.visualization(feature_data)
            logging.info(f"Finished feature '{feature_name}'.")

        # create ELAN annotation file


if __name__ == '__main__':
    main()
