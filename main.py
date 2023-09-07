"""

"""
import copy
import os
import subprocess
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from in_out import IO
from data import Data
from oslab_utils.in_out import create_tmp_folder
from oslab_utils.annotations import CustomEaf

#from detectors.nodding.noddingpigeon import NoddingPigeon
from detectors.gaze.ETH_XGaze import ETHXGaze
from detectors.gaze.XGaze_3cams import XGaze3cams
from detectors.human_pose.pose_detector import PoseDetector
from configs.config_handler import Configuration


all_methods = dict(
#        nodding_pigeon=NoddingPigeon,
        ethXgaze=ETHXGaze,
        xgaze_3cam=XGaze3cams,
        mmpose=PoseDetector,
)


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
    io = IO(config['io'], config['methods']['names'])

    # save experiment configs
    config_handler.save_experiment_config(
            io.get_output_folder('config', 'output'))

    # -> clean up all /tmp/ even if the code crashes or triggers an assertion
    with create_tmp_folder(io.get_all_tmp_folders()):

        # DATA preparation
        data = Data(config, io)

        # RUN detectors
        for method_name in config['methods']['names']:
            # prepare the part of the config relevant for the detector
            method_config = copy.deepcopy(config['methods'][method_name])
            if 'algorithm' in method_config.keys():
                method_config.update(method_config[method_config['algorithm']])

            detector = all_methods[method_name](method_config, io)
            detector.inference(data)

            detector.visualization(data)

        # RUN feature extractions

        # create ELAN annotation file


if __name__ == '__main__':
    main()
