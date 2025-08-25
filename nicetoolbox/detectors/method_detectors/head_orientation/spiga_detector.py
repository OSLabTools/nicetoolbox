"""
SPIGA method detector class.
"""

import logging
import os

import cv2
import numpy as np

from ....utils import filehandling as fh
from ....utils import video as vd
from ... import config_handler as confh
from ..base_detector import BaseDetector


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
        NotImplementedError: If a value in the dictionary is neither an integer nor a
        list.
    """
    if all(isinstance(val, int) for val in list(input_dict.values())):
        return list(input_dict.keys())
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


class Spiga(BaseDetector):
    """
    SPIGA is a method detector that computes the head_orientation component.

    Component: head_orientation

    Attributes:
        components (list): A list containing the name of the component: head_orientation
        algorithm (str): Algorithm name used to compute the head_orientation component.
        camera_names (list): List of camera names used to capture original input data.
    """

    components = ["head_orientation"]
    algorithm = "spiga"

    def __init__(self, config, io, data) -> None:
        """
        Initialize the SPIGA method detector with all inference preparation.

        Args:
            config (dict): Configuration settings for SPIGA.
            io (class): IO class instance for input/output operations.
            data (class): Data class instance for frame and subject data.
        """
        logging.info(
            f"Prepare Inference for '{self.algorithm}' and component {self.components}."
        )

        self.frames_list = data.frames_list
        config["frames_list"] = self.frames_list
        config["frame_indices_list"] = data.frame_indices_list
        self.video_start = data.video_start
        self.keypoints_indices = fh.load_config("./configs/predictions_mapping.toml")[
            self.components[0]
        ][self.algorithm]["keypoints_index"]
        config["face_landmarks_description"] = confh.flatten_list(
            extract_key_per_value(self.keypoints_indices["face"])
        )

        # Call the base class constructor
        super().__init__(config, io, data, requires_out_folder=config["visualize"])

        self.camera_names = [n for n in config["camera_names"] if n]
        self.cam_sees_subjects = config["cam_sees_subjects"]
        self.results_folder = config["result_folders"][self.components[0]]
        self.camera_order = config["camera_order"]

    def post_inference(self):
        """
        Calculate head orientation in 2D image after SPIGA inference.
        """
        n_subjects = len(self.subjects_descr)
        n_cams = len(self.camera_names)
        n_frames = len(self.frames_list)
        spiga_vectors = np.zeros((n_subjects, n_cams, n_frames, 4))

        prediction_file = os.path.join(self.results_folder, f"{self.algorithm}.npz")
        prediction = np.load(prediction_file, allow_pickle=True)
        predictions_dict = {key: prediction[key] for key in prediction.files}
        data_description = predictions_dict["data_description"].item()

        headposes = prediction["headpose"]
        face_landmarks = prediction["face_landmark_2d"]

        # Todo: vectorize
        for subj_idx in range(n_subjects):
            for cam_idx in range(n_cams):
                for frame_idx in range(len(self.frames_list)):
                    # Extract headpose
                    headpose = headposes[subj_idx][cam_idx][frame_idx]  # shape (6,)
                    landmarks = face_landmarks[subj_idx][cam_idx][frame_idx]
                    euler_yzx = np.array(
                        headpose[:3]
                    )  # first three value is euler angles

                    # Rotation matrix
                    rotation_matrix = self._euler_to_rotation_matrix(euler_yzx)

                    # 2D nose projection
                    nose_down = self.keypoints_indices["face"]["nose_down"]
                    # select middle point of nose_down landmarks
                    nose_down_index = nose_down[int(len(nose_down) / 2)]
                    nose_org = np.array(
                        [
                            (landmarks[nose_down_index][0]),
                            (landmarks[nose_down_index][1]),
                        ],
                        dtype=np.float32,
                    )
                    direction3D = np.array(
                        [100.0, 0, 0]
                    )  # Rotation order Y-Z-X, body’s forward axis is +X.
                    nose_direction_2D = rotation_matrix @ direction3D.reshape(3, 1)
                    nose_direction_2D = nose_direction_2D[:2].flatten()
                    nose_tip = nose_org + nose_direction_2D

                    # Optional: logging or boundary checks
                    if subj_idx >= len(self.subjects_descr):
                        logging.warning(f"Subject index {subj_idx} out of bounds")
                        continue
                    spiga_vectors[subj_idx, cam_idx, frame_idx, :] = [
                        nose_org[0],
                        nose_org[1],
                        nose_tip[0],
                        nose_tip[1],
                    ]
        predictions_dict["head_orientation_2d"] = spiga_vectors
        data_description.update(
            {
                "head_orientation_2d": {
                    "axis0": self.subjects_descr,
                    "axis1": self.camera_order,
                    "axis2": data_description["headpose"]["axis2"],
                    "axis3": ["start_x", "start_y", "end_x", "end_y"],
                }
            }
        )

        np.savez_compressed(prediction_file, **predictions_dict)
        logging.info("SPIGA post-processing result saved successfully.")

    def visualization(self, data):
        n_subj = len(self.subjects_descr)

        prediction_file = os.path.join(self.results_folder, f"{self.algorithm}.npz")
        predictions = np.load(prediction_file, allow_pickle=True)
        head_data = predictions["head_orientation_2d"]
        data_decr_arr = predictions["data_description"]
        camera_order = data_decr_arr.item()["head_orientation_2d"]["axis1"]

        # per camera and frame, visualize each subject's gaze
        success = True
        for camera_name in camera_order:
            cam_idx = camera_order.index(camera_name)
            os.makedirs(os.path.join(self.viz_folder, camera_name), exist_ok=True)

            for frame_idx in range(head_data.shape[2]):
                # load the original input image
                image_file = [
                    file for file in self.frames_list[frame_idx] if camera_name in file
                ][0]
                image = cv2.imread(image_file)

                colors = [(300, 30, 60), (0, 128, 0)]

                for subject_idx in range(n_subj):
                    if subject_idx not in self.cam_sees_subjects[camera_name]:
                        continue

                    head_orientation = head_data[subject_idx, cam_idx, frame_idx]
                    start = (int(head_orientation[0]), int(head_orientation[1]))
                    end = (int(head_orientation[2]), int(head_orientation[3]))

                    cv2.arrowedLine(
                        image,
                        start,
                        end,
                        colors[subject_idx],
                        thickness=3,
                        tipLength=0.1,
                    )

                cv2.imwrite(
                    os.path.join(
                        self.viz_folder,
                        camera_name,
                        f"{frame_idx + int(self.video_start):05d}.jpg",
                    ),
                    image,
                )

            # create and save video
            success *= vd.frames_to_video(
                os.path.join(self.viz_folder, camera_name),
                os.path.join(self.viz_folder, f"{camera_name}.mp4"),
                fps=data.fps,
                start_frame=int(self.video_start),
            )

        logging.info(
            f"Detector {self.components}: visualization finished with code "
            f"{success}."
        )

        # Note taken from spiga.demo.visualize.layouts.plot_headpose

    def _euler_to_rotation_matrix(self, headpose):
        # Change coordinates system
        euler = np.array([-(headpose[0] - 90), -headpose[1], -(headpose[2] + 90)])
        # Convert to radians
        rad = euler * (np.pi / 180.0)
        cy = np.cos(rad[0])
        sy = np.sin(rad[0])
        cp = np.cos(rad[1])
        sp = np.sin(rad[1])
        cr = np.cos(rad[2])
        sr = np.sin(rad[2])
        # labels in original Spiga function corrected,
        # the rotation in y-axis would named pitch, and z-axis yaw.
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])  # yaw
        Rp = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])  # pitch
        Rr = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])  # roll
        return np.matmul(np.matmul(Ry, Rp), Rr)
