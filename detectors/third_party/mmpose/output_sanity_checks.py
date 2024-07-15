import sys
import os
import numpy as np
import random
import logging

# internal imports
import detectors.method_detectors.body_joints.output_parser as parser
import utils.filehandling as fh


def compare_data_values_with_saved_json(predictions_folder,intermediate_results_folder, camera_list, person_threshold):
    """
    Sanity check function. Compare saved hdf5 values with algorithm original results. For each camera it selects randomly 5 image and 5 keypoint.
    And for each person compare the saved 2d keypoints in hdf5 file with the algorithm's original output.

    Parameters
    ----------
    predictions_folder: str
        the folder where the algorithms output is saved
    intermediate_results_folder: str
        results folder where the hdf5 files are saved
    camera_list: list
        the list of camera names
    person_threshold: float
        the threshold the determine left and right bbox.

    Returns
    -------
    asserts if there is a dismatch

    """
    logging.info("CHECKING saved 2d data...")

    i =0
    while i < 5:
        for camera_name in camera_list:
            camera_folder = predictions_folder[camera_name]
            filelist = sorted(os.listdir(camera_folder))
            # Randomly select a filename
            random_filename = random.choice(filelist)
            random_filename_path = os.path.join(camera_folder, random_filename)
            # Find the index of the randomly selected filename
            index_of_random_filename = filelist.index(random_filename)
            # get the name of hdf5 file
            hdf5_file = [f for f in sorted(os.listdir(intermediate_results_folder)) if camera_name in f][0]
            mmpose_parser = parser.MMPoseParser("coco_wholebody")  # TODO get framework name
            number_of_keypoints = mmpose_parser.get_number_of_keypoints(random_filename_path)

            # find the keypoint location in original json
            for person in ["personL", "personR"]:
                random_keypoint_index = random.randint(0, number_of_keypoints-1)
                xy_json = mmpose_parser.get_keypoint_location(
                    random_filename_path, random_keypoint_index,person,person_threshold
                )
                #find the keypoint location from hdf5 file
                hdf5_parser = parser.HDF5Parser("coco_wholebody")
                xy_data = hdf5_parser.get_keypoint_location(
                    input_file =os.path.join(intermediate_results_folder, hdf5_file),
                    frame_index= index_of_random_filename,
                    keypoint_index = random_keypoint_index,
                    person = person)

                # Compare the data at the random index between JSON and HDF5
                fh.assert_and_log(np.allclose(xy_json, xy_data, atol=1e5), "Data MISMATCH!")

                filename_in_data = fh.get_attribute_by_index(index_of_random_filename,
                                                                              os.path.join(intermediate_results_folder,
                                                                                           hdf5_file), person)
                # Compare the data at the random index between JSON and HDF5
                fh.assert_and_log(os.path.basename(filename_in_data).split('.')[0] == random_filename.split('.')[0], "Filenames MISMATCH!")
        i += 1
    logging.info("2d Data MATCH!")
    logging.info("Filenames MATCH!")