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
import tests.test_data as test_data
import oslab_utils.logging_utils as log_ut
from detectors.human_pose.filters import SGFilter


class PoseDetector(BaseDetector):
    """
    """
    name = 'mmpose'
    behavior = 'keypoints'

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
        self.person_threshold = (config["resolution"][
            0]) / 2 * 0.80  # multiply 0.80 bec. rarely one person's bbox cross the x/2
        self.video_start = config["video_start"]
        #input
        self.data_folder = io.get_data_folder()
        #output
        self.main_out = io.get_output_folder('config', 'output')
        self.intermediate_results = io.get_output_folder(self.name, 'additional')
        self.method_out_folder = io.get_output_folder(self.name, 'output')
        self.prediction_folders = self.get_prediction_folders(make_dirs=True)
        self.image_folders = self.get_image_folders(make_dirs=config["save_images"])
        self.log = os.path.join(self.main_out, f"{self.name}_inference.log")
        self.filtered = config["filtered"]
        if self.filtered:
            self.filter_window_length = config["window_length"]
            self.filter_polyorder = config["polyorder"]

        # first, make additions to the method/detector's config:
        # extract the relevant data input files from the data class
        log_ut.assert_and_log(data.all_camera_names == set(config['camera_names']),
                              f"camera_names do not match! all loaded cameras = " \
                              f"'{data.all_camera_names}' and {self.name} requires cameras " \
                              f"'{config['camera_names']}'."
                              )

        config['input_data_folder'] = data.create_symlink_input_folder(
                config['input_data_format'], config['camera_names'])

        config['frame_indices_list'] = data.frame_indices_list
        config['person_threshold'] = self.person_threshold
        config['data_folder'] = self.data_folder
        config['intermediate_results'] = self.intermediate_results
        config['prediction_folders'] = self.prediction_folders
        config['image_folders'] = self.image_folders
        config['log_file'] = self.log

        # then, call the base class init
        super().__init__(config, io, data)
        self.result_folder = config['result_folder']
        self.calibration = config['calibration']
        self.subjects_descr = io.subjects_descr

    def visualization(self, data):
        """

        Parameters
        ----------
        data: class
            a class instance that stores all input file locations
        """
        logging.info(f"VISUALIZING the method output {self.name}")
        log_ut.assert_and_log(os.listdir(self.image_folders["cam4"]) != [], "Image folder is empty") #ToDo camera hardcoded
        image_base = os.path.join(self.image_folders["cam4"],"%05d.png")
        output_path = os.path.join(self.viz_folder, f"{self.name}.mp4")

        ##TODO read fps directly and put this function under oslab_utils video
        cmd = f"ffmpeg -framerate {str(30)} -start_number {int(self.video_start)} -i {image_base} -c:v libx264 -pix_fmt yuv420p -y {output_path}"
        # Use the subprocess module to execute the command
        cmd_result = subprocess.run(cmd, shell=True)

        log_ut.assert_and_log(cmd_result.returncode == 0, f"FFMPEG video creation failed. Return code {cmd_result.returncode}")
        if os.path.isfile(output_path):
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
        if len(self.camera_names) > 1:
            logging.info("COMPUTING 3d position of the keypoints...")

            camera_frames_list = [os.path.basename(f).split(".")[0] for sublist in self.frame_list for f in sublist if
                                  self.camera_names[0] in f]  # since each frame inside a list #ToDo read from camera_folder
            prediction_files = [os.path.join(self.intermediate_results,f) for f in os.listdir(self.intermediate_results) if "hdf5" in f]
            cam1_data_path = [f for f in prediction_files if self.camera_names[0] in f][0]
            cam2_data_path = [f for f in prediction_files if self.camera_names[1] in f][0]
            cam1_data, _ = fh.read_hdf5(cam1_data_path)
            cam2_data, _ = fh.read_hdf5(cam2_data_path)

            log_ut.assert_and_log(
                len(cam1_data) == len(cam2_data) == len(self.subjects_descr),
                "Loaded prediction results differ in the number of persons.")
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
                # undistort data
                cam1_undistorted = np.squeeze(
                    tri.undistort_points_pinhole(xy_points_cam1,
                                                 np.array(self.calibration[self.camera_names[0]]["intrinsic_matrix"]),
                                                 np.array(self.calibration[self.camera_names[0]]["distortions"])))
                cam2_undistorted = np.squeeze(
                    tri.undistort_points_pinhole(xy_points_cam2,
                                                 np.array(self.calibration[self.camera_names[1]]["intrinsic_matrix"]),
                                                 np.array(self.calibration[self.camera_names[1]]["distortions"])))
                # triangulate data
                person_data_3d = tri.triangulate_stereo(
                    np.array(self.calibration[self.camera_names[0]]["projection_matrix"]),
                    np.array(self.calibration[self.camera_names[1]]["projection_matrix"]),
                    cam1_undistorted.T,
                    cam2_undistorted.T)

                # reshape 3d array
                reshaped_3D_points = person_data_3d.T.reshape(person_cam1.shape[0], person_cam1.shape[1], 3)

                ## Apply filter
                if self.filtered == True:
                    logging.info("APPLYING filtering...")
                    filter = SGFilter(self.filter_window_length, self.filter_polyorder)
                    smooth_data = filter.apply(reshaped_3D_points)
                    person_data_list.append(smooth_data)
                else:
                    person_data_list.append(reshaped_3D_points)

            if len(self.subjects_descr) == 2:
                # check person data shape
                log_ut.assert_and_log(person_data_list[0].shape == person_data_list[
                    1].shape, f"Shape mismatch: Shapes for personL and personR are not the same.")

            # check if any [0,0,0] prediction
            for person_data in person_data_list:
                test_data.check_zeros(person_data)
            # save results
            filepath = os.path.join(self.result_folder, f"{self.name}_3d.hdf5")
            fh.save_to_hdf5(person_data_list, groups_list=self.subjects_descr,
                            output_file=filepath, index=camera_frames_list)

            # check 3d data values
            # TODO: this check works only for videos with 2 subjects?
            # utils.compare_saved3d_data_values_with_triangulation_through_json(
            #     self.prediction_folders,
            #     self.result_folder,
            #     self.camera_names,
            #     self.calibration,
            #     self.person_threshold)

            return person_data_list
        else:
            pass

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
