"""

"""
import os
import numpy as np
import logging
import scipy.signal as signal
from detectors.base_detector import BaseDetector
import detectors.human_pose.utils as utils
import oslab_utils.triangulation as tri
import oslab_utils.filehandling as fh
import subprocess
import oslab_utils.check_and_exception as check
from detectors.human_pose.filters import SGFilter


class PoseDetector(BaseDetector):
    """
    """
    name = 'mmpose'
    behavior = 'keypoints'
    name2 = 'hrnetw48'

    def __init__(self, config, io, data):
        """ Initialize PoseHRNet class.

        Parameters
        ----------
        config : dict
            the method-specific configurations dictionary
        io: class
            a class instance that handles in-output folders
        """

        self.algorithm_name = config['algorithm']
        self.name += '_' + self.algorithm_name
        logging.info(f"STARTING Inference... - {self.name}")

        self.camera_names = config["camera_names"]
        self.frame_list = data.frames_list
        #self.person_threshold = (config["resolution"][0]) / 2 * 0.80  # multiply 0.80 bec. rarely one person's bbox cross the x/2
        self.video_start = data.video_start
        #input
        self.data_folder = io.get_data_folder()
        #output
        self.main_out = io.get_output_folder('config', 'output')
        self.intermediate_results = io.get_output_folder(self.name, 'additional')
        self.method_out_folder = io.get_output_folder(self.name, 'output')
        self.prediction_folders = self.get_prediction_folders(make_dirs=True)
        self.image_folders = self.get_image_folders(make_dirs=config["save_images"])
        self.filtered = config["filtered"]
        if self.filtered:
            self.filter_window_length = config["window_length"]
            self.filter_polyorder = config["polyorder"]

        # first, make additions to the method/detector's config:
        # extract the relevant data input files from the data class
        #log_ut.assert_and_log(data.all_camera_names == set(config['camera_names']),
        #                      f"camera_names do not match! all loaded cameras = " \
        #                      f"'{data.all_camera_names}' and {self.name} requires cameras " \
        #                      f"'{config['camera_names']}'."
        #                      )

        config['input_data_folder'] = data.create_symlink_input_folder(
                config['input_data_format'], config['camera_names'])

        #config['frame_indices_list'] = data.frame_indices_list
        #config['person_threshold'] = self.person_threshold
        config['data_folder'] = self.data_folder
        config['intermediate_results'] = self.intermediate_results
        config['prediction_folders'] = self.prediction_folders
        config['image_folders'] = self.image_folders
        #config['threshold'] = self.threshold

        # then, call the base class init
        super().__init__(config, io, data)
        self.result_folder = config['result_folder']
        self.calibration = config['calibration']

    def visualization(self, data):
        """

        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        logging.info(f"VISUALIZING the method output {self.name}")

        success = True
        for camera in self.camera_names:
            if os.listdir(self.image_folders[camera]) == []:
                logging.error("Image folder is empty")

            image_base = os.path.join(self.image_folders[camera],"%05d.png")
            output_path = os.path.join(self.viz_folder, f"{self.name}_{camera}.mp4")

            ##TODO read fps directly and put this function under oslab_utils video
            cmd = f"ffmpeg -framerate {str(30)} -start_number {int(self.video_start)} -i {image_base} -c:v libx264 -pix_fmt yuv420p -y {output_path}"
            # Use the subprocess module to execute the command
            cmd_result = subprocess.run(cmd, shell=True)
            if cmd_result.returncode != 0:
                logging.error(f"FFMPEG video creation failed. Return code {cmd_result.returncode}")

            success *= 1 if os.path.isfile(output_path) else 0

        if success:
            logging.info(f"VISUALIZATION {self.name} - SUCCESS")
        else:
            logging.error(f"VISUALIZATION {self.name} - FAILURE - Video file was not created")

    def post_inference(self):
        """
        The triangulation pipeline that first gets the pose estimation results (2d points) of each person
            in two camera, then undistorts the 2d points using calibration parameters, and apply stereo triangulation

        Parameters
        ----------
        data: class
            a class instance that stores all input file locations

        Returns
        -------
        saves 3d positions as hdf5 file and returns the 3d positions as a list of np.arrays[personL, personR]
        """

        prediction_file = os.path.join(self.result_folder, f"{self.name2}.npz")
        prediction = np.load(prediction_file, allow_pickle=True)
        data_description = prediction['data_description'].item()
        results_2d = prediction['2d']
        results_2d_bbox = prediction['bbox_2d']

        ## Apply filter
        if self.filtered:
            logging.info("APPLYING filtering to 3d data...")
            results_2d_filtered = results_2d.copy()
            filter = SGFilter(self.filter_window_length, self.filter_polyorder)
            results_2d_filtered = filter.apply(results_2d_filtered)


        ## if apply filter 2d interpolation is done on filtered data
        if self.filtered:
            results_2d_interpolated = results_2d_filtered.copy()
        else:
            results_2d_interpolated = results_2d.copy()
        keypoint_conf_threshold = 0.60
        # Creating the mask where the confidence score is below the threshold
        low_confidence_mask = results_2d_interpolated[:, :, :, :, 2] < keypoint_conf_threshold
        # Applying the mask to set the first and second values of num_estimates to NaN where the confidence is low
        results_2d_interpolated[low_confidence_mask, 0:2] = np.nan
        print(f"HAS NAN Values? : {np.isnan(results_2d_interpolated).any()}")
        results_2d_interpolated = utils.interpolate_data(results_2d_interpolated, is_3d=False)

        data_desr_2d_interpolated = {
            'axis0': data_description['2d']['axis0'],
            'axis1': data_description['2d']['axis0'],
            'axis2': data_description['2d']['axis2'],
            'axis3': data_description['2d']['axis3'],
            'axis4': data_description['2d']['axis4']
        }
        data_description['2d_interpolated'] = data_desr_2d_interpolated

        if len(self.camera_names) != 2:
            if self.filtered:

                data_description['2d_filtered'] = data_desr_2d_interpolated
                # save results
                results_dict = {
                    '2d': results_2d,
                    '2d_filtered': results_2d_filtered,
                    '2d_interpolated': results_2d_interpolated,
                    'bbox_2d': results_2d_bbox,
                    'data_description': data_description
                }
                np.savez_compressed(prediction_file, **results_dict)

            else:
                 # save results
                results_dict = {
                    '2d': results_2d,
                    '2d_interpolated': results_2d_interpolated,
                    'bbox_2d': results_2d_bbox,
                    'data_description': data_description
                }
                np.savez_compressed(prediction_file, **results_dict)

            if len(self.camera_names) >2:
                logging.WARNING("WARNING - Currently No 3d implementation for more than 2 camera")


        elif len(self.camera_names) == 2:
            logging.info("COMPUTING 3d position of the keypoints...")

            ### It is using interpolated_2d results instead of original 2d
            cam1_data, cam2_data = results_2d_interpolated[:, 0], results_2d_interpolated[:, 1]

            if results_2d.shape[0] != len(data_description['2d']['axis0']) != len(self.subjects_descr):
                logging.error("Loaded prediction results differ in the number of persons.")

            person_data_list = []
            for i in range(len(self.subjects_descr)):
                person_cam1 = cam1_data[i]
                person_cam2 = cam2_data[i]
                # log_ut.assert_and_log(len(camera_frames_list) == person_cam1.shape[0], \
                #     f"Different number of frames in frames list and frames in data. "
                #      f"camera_name:{self.camera_names[0]}, person: {person}")
                # log_ut.assert_and_log(len(camera_frames_list) == person_cam2.shape[0], \
                #     f"Different number of frames in frames list and frames in data. "
                #      f"camera_name:{self.camera_names[1]}, person: {person}")

                # Extract the x and y values
                xy_points_cam1 = person_cam1[:, :, :2].reshape(-1, 1, 2)
                xy_points_cam2 = person_cam2[:, :, :2].reshape(-1, 1, 2)

                # Since it is using interpolated data.
                # There might be some missing values.
                # Create a combined mask for NaN values in either camera's data
                nan_mask_cam1 = np.isnan(xy_points_cam1).any(axis=2)
                nan_mask_cam2 = np.isnan(xy_points_cam2).any(axis=2)
                combined_nan_mask = nan_mask_cam1 | nan_mask_cam2  # Combine masks

                # Filter out rows with NaNs for processing
                filtered_xy_points_cam1 = xy_points_cam1[~combined_nan_mask]
                filtered_xy_points_cam2 = xy_points_cam2[~combined_nan_mask]

                # undistort data
                cam1_undistorted = np.squeeze(
                    tri.undistort_points_pinhole(filtered_xy_points_cam1,
                                                 np.array(self.calibration[self.camera_names[0]]["intrinsic_matrix"]),
                                                 np.array(self.calibration[self.camera_names[0]]["distortions"])))
                cam2_undistorted = np.squeeze(
                    tri.undistort_points_pinhole(filtered_xy_points_cam2,
                                                 np.array(self.calibration[self.camera_names[1]]["intrinsic_matrix"]),
                                                 np.array(self.calibration[self.camera_names[1]]["distortions"])))
                # triangulate data
                person_data_3d = tri.triangulate_stereo(
                    np.array(self.calibration[self.camera_names[0]]["projection_matrix"]),
                    np.array(self.calibration[self.camera_names[1]]["projection_matrix"]),
                    cam1_undistorted.T,
                    cam2_undistorted.T)

                # reshape 3d array
                # Create output arrays filled with NaNs
                output_shape = (xy_points_cam1.shape[0], 3)
                output_data_3d = np.full(output_shape, np.nan)
                # Insert the processed data back into the correct positions
                output_data_3d[~combined_nan_mask.reshape(-1)] = person_data_3d.T

                reshaped_3D_points = output_data_3d.reshape(person_cam1.shape[0], person_cam1.shape[1], 3)
                print(f"shape of reshaped_3D_points: {reshaped_3D_points.shape}")
                print(f"First cam interpolated - first 2 frames keypoint 3-9: {results_2d_interpolated[i, 0, :2, 3:9,:]}")
                print(f"Second cam interpolated - first 2 frames keypoint 3-9:: {results_2d_interpolated[i, 1, :2, 3:9, :]}")
                #print(f"3d: {reshaped_3D_points[:2, 3:9, :]}")

                person_data_list.append(reshaped_3D_points)

            # check if any [0,0,0] prediction
            for person_data in person_data_list:
                check.check_zeros(person_data)

            # update data description
            data_desr_3d = {
                'axis0': data_description['2d']['axis0'],
                'axis1': 'none',
                'axis2': data_description['2d']['axis2'],
                'axis3': data_description['2d']['axis3'],
                'axis4': ['coordinate_x', 'coordinate_y', 'coordinate_z']
                }
            data_description['3d'] = data_desr_3d

            # save results 
            results_dict = {
                '2d': results_2d,
                '2d_interpolated': results_2d_interpolated,
                'bbox_2d': results_2d_bbox,
                '3d': np.stack(person_data_list)[:, None],
                'data_description': data_description
                }
            np.savez_compressed(prediction_file, **results_dict)

            # check 3d data values
            # TODO: this check works only for videos with 2 subjects?
            # utils.compare_saved3d_data_values_with_triangulation_through_json(
            #     self.prediction_folders,
            #     self.result_folder,
            #     self.camera_names,
            #     self.calibration,
            #     self.person_threshold)


    def get_prediction_folders(self, make_dirs = False):
        out_kp = {}
        for camera in self.camera_names:
            out = os.path.join(self.method_out_folder, 'predictions', camera)
            out_kp[camera] = out
            if make_dirs:
                os.makedirs(out, exist_ok=True)
        return out_kp

    def get_image_folders(self, make_dirs = False):
        out_img = {}
        for camera in self.camera_names:
            out = os.path.join(self.method_out_folder, 'images', camera)
            out_img[camera] = out
            if make_dirs:
                os.makedirs(out, exist_ok=True)
        return out_img

    def get_camera_data(self):
        in_camera = {}
        for camera in self.camera_names:
            camera_folder = os.path.join(self.data_folder, camera)
            in_camera[camera] = camera_folder
        return in_camera
