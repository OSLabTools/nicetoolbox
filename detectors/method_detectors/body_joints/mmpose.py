"""

"""
import os
import numpy as np
import logging
import scipy.signal as signal
import subprocess
from detectors.method_detectors.base_detector import BaseDetector
import detectors.method_detectors.body_joints.utils as utils
from detectors.method_detectors.body_joints.filters import SGFilter
import utils.triangulation as tri
import utils.filehandling as fh
import utils.config as cfg
import utils.check_and_exception as check
import configs.config_handler as confh


class MMPose(BaseDetector):
    """
    The MMPose class is a method detector for pose estimation using the MMPose framework.

    This class is designed to perform pose estimation on video data using the MMPose framework, 
    which is an open-source toolbox for comprehensive pose estimation. It provides the necessary 
    preparations and post-inference utilities to integrate the MMPose framework into our pipeline.

    The MMPose class supports 2D pose estimation and can handle data from multiple cameras. It also 
    has the capability to perform 3D pose estimation by triangulating the 2D pose data from different 
    camera views. Additionally, it offers the option to filter the 3D pose data to reduce noise and 
    smooth out the results.
    
    Available components: body_joints, hand_joints, face_landmarks

    Attributes:
        camera_names (list of str): Names of the cameras used to capture the video data. 
        video_start (int): The frame number from which to start pose estimation. 
        filtered (bool): Indicates whether the 3D pose data should be filtered.
        filter_window_length (int): Specifies the length of the window used for filtering the 3D data. 
        filter_polyorder (int): Defines the order of the polynomial used in the filtering process. 
        data_folder (str): The file path to the folder containing the input data.
        out_folder (str): The file path to the folder where the pose estimation results will be saved.
        prediction_folders (dict of str: str): A dictionary mapping camera names to their respective
            folders for storing the pose estimation predictions.
        image_folders (dict of str: str): A dictionary mapping camera names to their respective folders 
            containing the input images or frames for pose estimation.
        result_folders (dict of str: str): A dictionary mapping camera names to their respective folders
            for storing the final pose estimation results.
        calibration (dict): Contains the calibration parameters for the cameras used in pose estimation. 
    """

    def __init__(self, config, io, data):
        """ 
        Initializes the MMPose class with configuration settings and IO handling capabilities.
        
        This constructor takes care of all inference preparation steps, including setting up
        the input and output folders, configuring the keypoint mapping, and preparing the calibration 
        parameters for 3D pose estimation.
        
        Args:
            config (dict): A dictionary containing the configuration settings for the method detector.
            io (class): An instance of the IO class for input-output operations.
            data (class): An instance of the Data class for accessing data.
        """

        logging.info(f"Prepare Inference for '{self.algorithm}' and components {self.components}.")

        self.camera_names = config["camera_names"]
        self.video_start = data.video_start
        # person_threshold: multiply 0.80 bec. rarely one person's bbox cross the x/2
        # self.person_threshold = (config["resolution"][0]) / 2 * 0.80  
        # config['person_threshold'] = self.person_threshold
        self.filtered = config["filtered"]
        if self.filtered:
            self.filter_window_length = config["window_length"]
            self.filter_polyorder = config["polyorder"]

        #input
        self.data_folder = io.get_data_folder()
        config['input_data_folder'] = data.create_symlink_input_folder(
                config['input_data_format'], config['camera_names'])
        
        #output
        main_component = self.components[0]
        self.out_folder = io.get_detector_output_folder(main_component, self.algorithm, 'output')
        self.prediction_folders = self.get_prediction_folders(make_dirs=True)
        self.image_folders = self.get_image_folders(make_dirs=config["visualize"])
        config['prediction_folders'] = self.prediction_folders
        config['image_folders'] = self.image_folders
        config['data_folder'] = self.data_folder
        #config['intermediate_results'] = io.get_detector_output_folder(main_component, self.algorithm, 'additional')

        # keypoints mapping
        keypoints_indices = cfg.load_config("./detectors/configs/predictions_mapping.toml")[
                "human_pose"][config["keypoint_mapping"]]['keypoints_index']
        mapping = self.get_per_component_keypoint_mapping(keypoints_indices)
        config['keypoints_indices'], config['keypoints_description'] = mapping

        # then, call the base class init
        super().__init__(config, io, data)
        self.result_folders = config['result_folders']
        self.calibration = config['calibration']

        logging.info(f"Inference Preparation completed.\n")

    # TODO: Somehow mark this method as abstract and to be implemented by the derived classes (algorithms)
    def get_per_component_keypoint_mapping(self, keypoints_indices):
        """
        This method extracts the keypoint indices and descriptions for each pose estimation component.
        
        It has to be implemented by the derived classes assoicated to the available pose estimation
        algorithms. (See HRNetw48 and Vitpose classes below)
        
        Available algorithms are: hrnetw48, vitpose

        Args:
            keypoints_indices (_type_): _description_
        """
        pass

    def visualization(self, data):
        """
        Generates a visualization video for each camera from the processed image frames.

        This method takes the processed image frames for each camera and compiles them into a 
        video file. It uses the `ffmpeg` command-line tool to create a video with a specified 
        frame rate. For each camera, it checks if the image folder is empty and logs an error 
        if so. Otherwise, it constructs the video file path and executes the `ffmpeg` command 
        to generate the video. The success of the video creation is tracked, and the method logs 
        the outcome of the visualization process for each camera.

        Args:
            data (class): An instance of a class that stores all data related information, 
                including the frame rate (`fps`) for the video and the starting frame number 
                (`video_start`).

        Raises:
            FileNotFoundError: If the image folder for any camera is empty, indicating that there 
                are no processed frames to visualize, a FileNotFoundError is logged as an error.

        Note:
            - The method assumes that the processed image frames are named in a specific format (`%05d.png`), 
            where each frame's name is a zero-padded five-digit number representing its sequence in the video.
        """
        logging.info(f"VISUALIZING the method detector output of {self.components} and {self.algorithm}.")

        success = True
        for camera in self.camera_names:
            if os.listdir(self.image_folders[camera]) == []:
                logging.error("Image folder is empty")

            image_base = os.path.join(self.image_folders[camera],"%05d.png")
            output_path = os.path.join(self.viz_folder, f"{self.algorithm}_{camera}.mp4")

            ##TODO put this function under utils video
            cmd = f"ffmpeg -framerate {data.fps} -start_number {int(self.video_start)} -i {image_base} -c:v libx264 -pix_fmt yuv420p -y {output_path}"
            # Use the subprocess module to execute the command
            cmd_result = subprocess.run(cmd, shell=True)
            if cmd_result.returncode != 0:
                logging.error(f"FFMPEG video creation failed. Return code {cmd_result.returncode}")

            success *= 1 if os.path.isfile(output_path) else 0

        if success:
            logging.info(f"VISUALIZATION {self.components}, {self.algorithm} - SUCCESS")
        else:
            logging.error(f"VISUALIZATION {self.components}, {self.algorithm} - FAILURE - Video file was not created")

    # TODO: Split this method into smaller functions for better readability
    def post_inference(self):
        """
        Post-inference processing for pose estimation components such as body_joints, hand_joints,
        and face_landmarks.

        This method takes the raw 2D pose estimation results and applies a series of processing 
        steps. They include optional filtering to smooth the results, interpolation to fill in 
        missing values, undistortion using camera calibration parameters, and 3D triangulation from 
        multiple camera views. The final processed results are saved for further analysis 
        and visualization for each of the components.
        
        Steps:
        1. **Filtering**: Applies a smoothing filter to the 2D pose estimation results if filtering
            is enabled. This step reduces noise and improves the consistency of the pose data over 
            time.
        2. **Interpolation**: Fills in missing values in the 2D pose estimation results. This is 
            crucial for maintaining the integrity of the pose data, especially in cases where 
            occlusion or poor lighting conditions may lead to incomplete detections.
        3. **Undistortion**: Corrects the 2D pose estimation results for lens distortion using the 
            camera's calibration parameters.
        4. **3D Triangulation**: Uses the undistorted 2D pose estimation results from at least two
            camera views to reconstruct the 3D positions of the pose keypoints.
        5. **Saving Results**: The processed 3D pose data is saved to a .npz file with the following 
            structure:
                - '2d': A numpy array containing the 2D pose estimation results.
                - '2d_filtered': A numpy array containing the filtered 2D pose estimation results.
                - '2d_interpolated': A numpy array containing the interpolated 2D pose estimation results.
                - 'bbox_2d': A numpy array containing the 2D bounding box coordinates.
                - '3d': A numpy array containing the 3D pose estimation results.
                - 'data_description': A dictionary containing the data description for the above output numpy arrays. See the documentation of the output for more details.
                
        Returns:
            None. The processed results are saved to the output folder (See step 5).
        """

        for component, result_folder in self.result_folders.items():

            prediction_file = os.path.join(result_folder, f"{self.algorithm}.npz")
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
            results_2d_interpolated = utils.interpolate_data(results_2d_interpolated, is_3d=False)

            data_description['2d_interpolated'] = data_description['2d']
            if len(self.camera_names) != 2:
                if self.filtered:

                    data_description['2d_filtered'] = data_description['2d']
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
                    #print(f"3d: {reshaped_3D_points[:2, 3:9, :]}")

                    person_data_list.append(reshaped_3D_points)

                # check if any [0,0,0] prediction
                for person_data in person_data_list:
                    check.check_zeros(person_data)

                # save results 
                descr_2d = data_description['2d']
                data_description.update({
                    '3d': dict(
                        axis0=descr_2d["axis0"],
                        axis1=None,
                        axis2=descr_2d["axis2"],
                        axis3=descr_2d["axis3"],
                        axis4=['coordinate_x', 'coordinate_y', 'coordinate_z']
                    )
                })
                if self.filtered:
                    data_description['2d_filtered'] = data_description['2d']
                    # save results
                    results_dict = {
                        '2d': results_2d,
                        '2d_filtered': results_2d_filtered,
                        '2d_interpolated': results_2d_interpolated,
                        'bbox_2d': results_2d_bbox,
                        '3d': np.stack(person_data_list)[:, None],
                        'data_description': data_description
                    }
                    np.savez_compressed(prediction_file, **results_dict)
                else:
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
                #     result_folder,
                #     self.camera_names,
                #     self.calibration,
                #     self.person_threshold)

    def get_prediction_folders(self, make_dirs = False):
        """
        Generate and return a dictionary of prediction folders for each camera.

        Args:
            make_dirs (bool): If True, the function will create the directories if they do not exist.

        Returns:
            dict: A dictionary where the keys are the camera names and the values are the 
                corresponding prediction folder paths.
        """
        out_kp = {}
        for camera in self.camera_names:
            out = os.path.join(self.out_folder, 'predictions', camera)
            out_kp[camera] = out
            if make_dirs:
                os.makedirs(out, exist_ok=True)
        return out_kp

    def get_image_folders(self, make_dirs = False):
        """
        Generate and return a dictionary of image folders for each camera.

        Args:
            make_dirs (bool): If True, the function will create the directories if they do not exist.

        Returns:
            dict: A dictionary where the keys are the camera names and the values are the 
                corresponding image folder paths.
        """
        out_img = {}
        for camera in self.camera_names:
            out = os.path.join(self.out_folder, 'images', camera)
            out_img[camera] = out
            if make_dirs:
                os.makedirs(out, exist_ok=True)
        return out_img

    def get_camera_data(self):
        """
        Retrieves the camera data for each camera name.
    
        This function constructs a dictionary where the keys are the camera names
        and the values are the corresponding camera folders. The camera folders are
        obtained by joining the `data_folder` with the camera name.
        
        Returns:
            dict: A dictionary containing the camera data, where the keys are the camera names
                    and the values are the corresponding camera folders.
        """
        in_camera = {}
        for camera in self.camera_names:
            camera_folder = os.path.join(self.data_folder, camera)
            in_camera[camera] = camera_folder
        return in_camera


def extract_key_per_value(input_dict):
    """
    Extracts keys from a dictionary based on the type of their values.

    If all values in the dictionary are integers, it returns a list of keys.
    If any value is a list, it appends an index to the key to create a unique key.

    Args:
        input_dict (dict): The input dictionary to extract keys from.

    Returns:
        return_keys (list): A list of keys extracted from the input dictionary.

    Raises:
        NotImplementedError: If a value in the dictionary is neither an integer nor a list.
    """
    if all(isinstance(l, int) for l in list(input_dict.values())):
        return list(input_dict.keys())
    else:
        return_keys = []
        for key, value in input_dict.items():
            if isinstance(value, int):
                return_keys.append(value)
            elif isinstance(value, list):
                for idx, _ in enumerate(value):
                    return_keys.append(f"{key}_{idx}")
            else:
                raise NotImplementedError
        return return_keys            


class HRNetw48(MMPose):
    """
    HRNetw48 is a subclass of MMPose specialized for pose estimation using the HRNetw48 model.

    The HRNetw48 class is designed to utilize the HRNetw48 model within the MMPose framework 
    for pose estimation tasks. It provides the necessary component keypoint indices
    and descriptions for the body joints component. The base_detector class starts the algorithm 
    as a subprocess within the base_detector's 'run_inference' method.
    
    HRNetw48, or High-Resolution Network, is a deep neural network designed for human pose estimation
    that maintains high-resolution representations through the whole process. It is particularly
    effective for tasks requiring precise localization of body joints. 
    
    Components: body_joints, hand_joints, face_landmarks
    """
    components = ['body_joints', 'hand_joints', 'face_landmarks']
    algorithm = 'hrnetw48'

    def get_per_component_keypoint_mapping(self, keypoints_indices):
        """
        Extracts and returns the indices and descriptions of keypoints for each component.

        Args:
            keypoints_indices (dict): A dictionary containing the indices of keypoints for each 
                component. The keys of the dictionary are the component names ('body_joints', 
                'hand_joints', 'face_landmarks'), and the values are dictionaries containing the 
                indices of keypoints for each keypoint.

        Returns:
            tuple: A tuple containing two dictionaries.
                - The first dictionary contains the indices of keypoints for each component.
                - The second dictionary contains the descriptions of keypoints for each component.

        """
        indices = dict(
            body_joints = confh.flatten_list(list(keypoints_indices['body'].values()) + 
                                             list(keypoints_indices['foot'].values())),
            hand_joints = confh.flatten_list(list(keypoints_indices['hand'].values())),
            face_landmarks = confh.flatten_list(list(keypoints_indices['face'].values())),
            )
        
        description = dict(
            body_joints = confh.flatten_list(extract_key_per_value(keypoints_indices['body']) + 
                                             extract_key_per_value(keypoints_indices['foot'])),
            hand_joints = confh.flatten_list(extract_key_per_value(keypoints_indices['hand'])),
            face_landmarks = confh.flatten_list(extract_key_per_value(keypoints_indices['face'])),
            )
        
        return indices, description


class VitPose(MMPose):
    """
    VitPose is a subclass of MMPose specialized for pose estimation using the Vision Transformer (ViT) model.

    The VitPose class is designed to utilize the ViT model within the MMPose framework for pose 
    estimation tasks, focusing on body joints. It provides the necessary component keypoint indices
    and descriptions for the body joints component. The base_detector class starts the algorithm 
    as a subprocess within the base_detector's 'run_inference' method.
    
    VitPose leverages the Vision Transformer architecture, a model that applies the transformer mechanism to 
    image processing tasks, including pose estimation. 

    Component: body_joints
    """
    components = ['body_joints']
    algorithm = 'vitpose'

    def get_per_component_keypoint_mapping(self, keypoints_indices):
        """
        Extracts and returns the indices and descriptions of keypoints for each component.

        Args:
            keypoints_indices (dict): A dictionary containing the indices of keypoints for each 
                component. The keys of the dictionary are the component names ('body_joints', 
                'hand_joints', 'face_landmarks'), and the values are dictionaries containing the 
                indices of keypoints for each keypoint.
                Note: This algorithm only supports the 'body_joints' component.

        Returns:
            tuple: A tuple containing two dictionaries.
                - The first dictionary contains the indices of keypoints for each component.
                - The second dictionary contains the descriptions of keypoints for each component.

        """
        indices = dict(
            body_joints = confh.flatten_list(list(keypoints_indices['body'].values()))
            )
        
        description = dict(
            body_joints = confh.flatten_list(extract_key_per_value(keypoints_indices['body']))
            )
        
        return indices, description

