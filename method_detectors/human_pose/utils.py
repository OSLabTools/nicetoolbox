import sys
import os
import numpy as np
import random
import cv2
import logging
import scipy.interpolate as interp
from method_detectors.human_pose.output_parser import MMPoseParser, HDF5Parser
import oslab_utils.logging_utils as log_ut


def compare_saved3d_data_values_with_triangulation_through_json(prediction_folders, results_folder, camera_names,
                                                                calibration_params, person_threshold):
    """
    Sanity check function. Compare saved 3d hdf5 values with triangulating algorithm original results. It selects randomly 5 images and 5 keypoints.
    And for each person compare the saved 3d keypoints in hdf5 file with the triangulated algorithm original results.

    Parameters
    ----------
    prediction_folders: dict
                    prediction folder dictionary - camera name is the key [str:detector_out/method_output/predictions/camera_name]
    results_folder: str
                    results folder where the hdf5 files are saved [str:detector_out/]
    camera_names: lst
                    list of camera name
    calibration_params: dict
                    calibration dictionary - camera name is the key
    person_threshold: float
                    the threshold the determine left and right bbox.

    Returns
    -------
    asserts if there is a dismatch

    """
    # Todo: add to the log
    print("CHECKING saved 3d data & triangulation...")
    i = 0
    while i < 5:
        camera1_path = prediction_folders[camera_names[0]]
        camera2_path=prediction_folders[camera_names[1]]
        filelist_cam1 = sorted(os.listdir(camera1_path))
        # Randomly select a filename
        random_filename_cam1 = random.choice(filelist_cam1)
        # get file from each camera
        random_filename_path_cam1 = os.path.join(camera1_path, random_filename_cam1)
        random_filename_path_cam2 = os.path.join(camera2_path, random_filename_cam1.replace(f"{camera_names[0]}", f"{camera_names[1]}"))
        # Find the index of the randomly selected filename
        index_of_random_filename = filelist_cam1.index(random_filename_cam1)
        # get the name of hdf5 file
        hdf5_file = [f for f in sorted(os.listdir(results_folder)) if "3d" in f][0]
        mmpose_parser = MMPoseParser("coco_wholebody")  # TODO get framework name
        number_of_keypoints = mmpose_parser.get_number_of_keypoints(random_filename_path_cam1)
        # find the keypoint location in original json
        for person in ["personL", "personR"]: # ToDO hardcoded
            random_keypoint_index = random.randint(0, number_of_keypoints-1)
            xy_json_cam1 = mmpose_parser.get_keypoint_location(
                random_filename_path_cam1, random_keypoint_index, person, person_threshold
            )

            xy_json_cam2 = mmpose_parser.get_keypoint_location(
                random_filename_path_cam2, random_keypoint_index, person, person_threshold
            )
            # find the keypoint location from hdf5 file
            hdf5_parser = HDF5Parser("coco_wholebody")
            xyz_data = hdf5_parser.get_keypoint_location(
                input_file=os.path.join(results_folder, hdf5_file),
                frame_index=index_of_random_filename,
                keypoint_index=random_keypoint_index,
                person=person, xyz = True)

            # undistort data
            cam1_undistorted = cv2.undistortPoints(np.array(xy_json_cam1),
                                         np.array(calibration_params[camera_names[0]]["intrinsic_matrix"]),
                                         np.array(calibration_params[camera_names[0]]["distortions"]),
                                        P=np.array(calibration_params[camera_names[0]]["intrinsic_matrix"]))
            cam2_undistorted = cv2.undistortPoints(np.array(xy_json_cam2),
                                         np.array(calibration_params[camera_names[1]]["intrinsic_matrix"]),
                                         np.array(calibration_params[camera_names[1]]["distortions"]),
                                         P=np.array(calibration_params[camera_names[1]]["intrinsic_matrix"]))

            # triangulate data
            d3_homogeneous = cv2.triangulatePoints(
                np.array(calibration_params[camera_names[0]]["projection_matrix"]),
                np.array(calibration_params[camera_names[1]]["projection_matrix"]),
                np.squeeze(cam1_undistorted),
                np.squeeze(cam2_undistorted))

            d3_euclidean = d3_homogeneous / d3_homogeneous[3]
            xyz_cal = np.squeeze(d3_euclidean[:3])
            # Compare the data at the random index between JSON and HDF5
            log_ut.assert_and_log(np.allclose(xyz_data, xyz_cal, atol=1e5), "Data MISMATCH!")
        i += 1
    logging.info("3d Data MATCH!")

# Function to interpolate missing data using scipy
def interpolate_data(data, is_3d= True, max_empty=10):
    num_people, num_cameras, num_frames, num_keypoints, _ = data.shape
    for i in range(num_people):
        for j in range(num_cameras):
            for k in range(num_keypoints):
                x = data[i, j, :, k, 0]
                y = data[i, j, :, k, 1]
                z = None
                if is_3d:
                    z = data[i, j, :, k, 2]

                # Check for NaNs and only proceed if there are any
                if np.isnan(x).any() or np.isnan(y).any():
                    valid = ~np.isnan(x)
                    valid_idx = np.where(valid)[0]

                    # Check gaps in valid indices and filter out large gaps
                    if valid_idx.size > 1:
                        gaps = np.diff(valid_idx)
                        small_gaps_idx = np.where(gaps <= max_empty)[0]
                        small_gaps_valid_idx = valid_idx[small_gaps_idx]

                        if small_gaps_valid_idx.size > 0:
                            small_gaps_valid_idx = np.append(small_gaps_valid_idx, valid_idx[small_gaps_idx[-1] + 1])

                        if small_gaps_valid_idx.size > 1:  # Need at least two points to interpolate
                            # Create interpolation functions for bounded regions
                            f_x = interp.interp1d(small_gaps_valid_idx, x[small_gaps_valid_idx],
                                                  kind='linear', bounds_error=False, fill_value=np.nan)
                            f_y = interp.interp1d(small_gaps_valid_idx, y[small_gaps_valid_idx],
                                                  kind='linear', bounds_error=False, fill_value=np.nan)
                            f_z = None
                            if is_3d:
                                f_z = interp.interp1d(small_gaps_valid_idx, z[small_gaps_valid_idx],
                                                      kind='linear', bounds_error=False, fill_value=np.nan)

                            # Apply interpolation only within the gaps
                            for gap_start, gap_end in zip(small_gaps_valid_idx[:-1], small_gaps_valid_idx[1:]):
                                data[i, j, gap_start:gap_end + 1, k, 0] = f_x(np.arange(gap_start, gap_end + 1))
                                data[i, j, gap_start:gap_end + 1, k, 1] = f_y(np.arange(gap_start, gap_end + 1))
                                if is_3d:
                                    data[i, j, gap_start:gap_end + 1, k, 2] = f_z(np.arange(gap_start, gap_end + 1))

    return data
