"""
Components module for defining various visual components.

Classes:
    GazeIndividualComponent: Class for visualizing individual gaze data.
    BodyJointsComponent: Class for visualizing body joints data.
    HandJointsComponent: Class for visualizing hand joints data.
    FaceLandmarksComponent: Class for visualizing face landmarks data.
    GazeInteractionComponent: Class for visualizing gaze interaction data.
    ProximityComponent: Class for visualizing proximity data.
    KinematicsComponent: Class for visualizing kinematics data.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import rerun as rr

from ...configs.config_handler import load_validated_config_raw
from ...configs.schemas.predictions_mapping import PredictionsMappingConfig

PREDICTIONS_MAPPING_FILE = "configs/predictions_mapping.toml"
PREDICTIONS_MAPPING = load_validated_config_raw(
    PREDICTIONS_MAPPING_FILE, PredictionsMappingConfig
)


class Component(ABC):
    """
    Abstract class for defining visual components.

    Attributes:
        visualizer_config (dict): Configuration settings for the visualizer.
        component_name (str): The name of the component.
        logger (viewer.Viewer): The viewer object for logging the visualizations.
        algorithm_list (list): The list of algorithms used for the component.
        component_prediction_folder (str): The path to the component prediction folder.
        canvas_list (list): The list of canvases for the component.
        algorithms_results (list): The list of algorithm results for the component.
        canvas_data (dict): The dictionary of canvas data for the component.
    """

    def __init__(self, visualizer_config, io, logger, component_name):
        self.visualizer_config = visualizer_config
        self.component_name = component_name
        self.logger = logger
        self.component_prediction_folder = io.get_component_results_folder(
            visualizer_config["media"]["video_name"], component_name=component_name
        )
        self.algorithm_list = self.visualizer_config["media"][self.component_name][
            "algorithms"
        ]

        # get canvas list from visualizer_config
        canvas_list = []
        for canvases in self.visualizer_config["media"][self.component_name][
            "canvas"
        ].values():
            canvas_list.extend(canvases)
        self.canvas_list = list(set(canvas_list))

        # load algorithm results
        self.algorithms_results = []
        for alg in self.algorithm_list:
            alg_path = io.get_algorithm_result(self.component_prediction_folder, alg)
            try:
                self.algorithms_results.append(np.load(alg_path, allow_pickle=True))
            except FileNotFoundError:
                print(f"ERROR: {alg}.npz file is not found in {self.component_name} folder."
                      f"It will not be visualized\n  "
                      f"Remove {alg} or {self.component_name} in the visualizer_config.toml file")
                raise

        # create canvas data dictionary - key is data name, and value is algorithms data
        # (lists of algorithms results)
        self.canvas_data = {}
        for data_name, canvas in self.visualizer_config["media"][self.component_name][
            "canvas"
        ].items():
            if canvas != []:
                self.algorithms_data = []
                if self.algorithms_results:
                    for i, _alg in enumerate(self.algorithm_list):
                        self.algorithms_data.append(self.algorithms_results[i][data_name])
                    self.canvas_data[data_name] = self.algorithms_data

    def _parse_alg_color(self, alg_idx: int) -> List[int]:
        """
        Parse the color for the algorithm index.

        Args:
            alg_idx (int): The index of the algorithm.

        Returns:
            list: The color for the algorithm index.
        """
        return self.visualizer_config["media"][self.component_name]["appearance"][
            "colors"
        ][alg_idx]

    def _parse_radii(self, type: str) -> float:
        """
        Parse the radii for the type. Type is one of '3d' or 'camera_view'.

        Args:
            type (str): The type of radii.

        Returns:
            float: The radii for the type.
        """
        if type == "3d":
            return self.visualizer_config["media"][self.component_name]["appearance"][
                "radii"
            ]["3d"]
        if type == "camera_view":
            return self.visualizer_config["media"][self.component_name]["appearance"][
                "radii"
            ]["camera_view"]
        raise ValueError("Invalid type. Use either '3d' or 'camera_view'")

    @abstractmethod
    def _get_algorithms_labels(self):
        """
        Abstract method to get the labels for the algorithms.
        """
        pass

    @abstractmethod
    def _log_data(self):
        """
        Abstract method to log the data.
        """
        pass

    @abstractmethod
    def visualize(self):
        """
        Abstract method to visualize the component.
        """
        pass


class BodyJointsComponent(Component):
    """
    Class for visualizing body joint data.
    """

    def __init__(self, visualizer_config: Dict, io, logger, component_name: str):
        super().__init__(visualizer_config, io, logger, component_name)
        # note: All these numpy arrays share a common structure in their first 3
        # dimension : [number_of_subjects, number_of_cameras, number_of_frames]
        # by design all algorithms in same component shares the same cameras and
        # subjects -- therefore the camera_names and subject_names results will be
        # read from first algorithm data description axis0 gives subject information
        self.subject_names = self.algorithms_results[0]["data_description"].item()[
            "2d"
        ]["axis0"]
        # data description axis1 gives camera information
        self.camera_names = self.algorithms_results[0]["data_description"].item()["2d"][
            "axis1"
        ]

    def calculate_middle_eyes(self, dimension: int) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate the middle of the eyes for the given dimension.

        Args:
            dimension (int): The dimension for the middle eyes.

        Returns:
            Tuple[np.ndarray, List[str]]: The middle eyes data and the camera names.
        """
        # we will use first algorithm results
        labels = self._get_algorithms_labels()[0]
        right_eye_idx = labels.index("right_eye")
        left_eye_idx = labels.index("left_eye")
        dim = f"{dimension}d"
        if (dimension < 2) | (dimension > 3):
            assert "supported dimensions are: 2 or 3"
        elif dimension == 3:
            if dim not in self.canvas_data.keys():
                print(
                    f"{dim} results could not found in selected canvas data.\n"
                    f"If you don't have 3d results, set multi-view false in visualizer_config.toml.\n"
                    f"If you have 3d results, add '3D_Canvas' into body_joint.canvas in visualizer_config.toml")
                return (None, None)
        data = self.algorithms_results[0][dim]
        mean_value = np.mean(data[:, :, :, [right_eye_idx, left_eye_idx], :dimension], axis=3)
        return (mean_value, self.camera_names)

    def _get_algorithms_labels(self) -> List[List[str]]:
        """
        Get the labels for the algorithms.

        Returns:
            List[List[str]]: The labels for the algorithms.
        """
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, _alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]["data_description"].item()["2d"]["axis3"]
            )
        return algorithm_labels

    def _get_skeleton_connections(
        self, alg_idx: int, predictions_mapping: Dict
    ) -> List[List[str]]:
        """
        Get the skeleton connections for the algorithm index from the predictions
        mapping.

        Args:
            alg_idx (int): The index of the algorithm.
            predictions_mapping (Dict): The predictions mapping.

        Returns:
            List[List[str]]: The skeleton connections for the algorithm index.
        """
        alg_name = self.algorithm_list[alg_idx]
        # get algorithm keypoint type
        alg_type = self.visualizer_config["algorithms_properties"][alg_name][
            "keypoint_mapping"
        ]
        return predictions_mapping["human_pose"][alg_type]["connections"][
            self.component_name
        ]

    def _log_skeleton(
        self, entity_path: str, data_points: np.ndarray, dimension: int, alg_idx: int
    ) -> None:
        """
        Log the skeleton data points in rerun.

        Args:
            entity_path (str): The entity path.
            data_points (np.ndarray): The data points.
            dimension (int): The dimension.
            alg_idx (int): The algorithm index.
        """
        keypoints_dict = {
            label: i for i, label in enumerate(self._get_algorithms_labels()[alg_idx])
        }
        connections = self._get_skeleton_connections(alg_idx, PREDICTIONS_MAPPING)
        start_points, end_points = [], []
        for connect in connections:
            for k in range(len(connect) - 1):
                if (connect[k] in keypoints_dict) & (connect[k + 1] in keypoints_dict):
                    start = keypoints_dict[connect[k]]
                    end = keypoints_dict[connect[k + 1]]
                    start_points.append([data_points[start]])
                    end_points.append([data_points[end]])

        start_points = np.array(start_points).reshape(-1, dimension)
        end_points = np.array(end_points).reshape(-1, dimension)
        color = self._parse_alg_color(alg_idx)

        if dimension == 2:
            radii = self.visualizer_config["media"][self.component_name]["appearance"][
                "radii"
            ]["camera_view"]
            rr.log(
                entity_path,
                rr.LineStrips2D(
                    np.stack((start_points, end_points), axis=1),
                    colors=color,
                    radii=radii,
                ),
            )
        else:
            radii = self.visualizer_config["media"][self.component_name]["appearance"][
                "radii"
            ]["3d"]
            rr.log(
                entity_path,
                rr.LineStrips3D(
                    np.stack((start_points, end_points), axis=1),
                    colors=color,
                    radii=radii,
                ),
            )

    def _log_data(
        self, entity_path: str, data_points: np.ndarray, dimension: int, alg_idx: int
    ) -> None:
        """
        Log the data points in rerun.

        Args:
            entity_path (str): The entity path.
            data_points (np.ndarray): The data points.
            dimension (int): The dimension.
            alg_idx (int): The algorithm index.
        """
        color = self._parse_alg_color(alg_idx)
        if dimension == "2d":
            radii = self._parse_radii("camera_view")
            rr.log(
                entity_path,
                rr.Points2D(
                    data_points,
                    keypoint_ids=list(range(data_points.shape[0])),
                    colors=color,
                    radii=radii,
                ),
            )
        elif dimension == "3d":
            radii = self._parse_radii("3d")
            rr.log(
                entity_path,
                rr.Points3D(
                    data_points,
                    keypoint_ids=list(range(data_points.shape[0])),
                    colors=color,
                    radii=radii,
                ),
            )

    def visualize(self, frame_idx: int) -> None:
        """
        Visualize the body joints component.

        Combines the _log_data and _log_skeleton methods to visualize the body joints
        component in either 2D or 3D.

        Args:
            frame_idx (int): The frame index.
        """
        for canvas in self.canvas_list:
            if canvas == "3D_Canvas":
                for alg_idx, alg_data in enumerate(self.canvas_data["3d"]):
                    if frame_idx >= alg_data.shape[2]:  # number of frames
                        continue
                    alg_name = self.algorithm_list[alg_idx]
                    for subject_idx, subject in enumerate(self.subject_names):
                        subject_3d_points = alg_data[subject_idx, 0, frame_idx][:, :3]  # select first 3 values, 4rd is confidence score
                        entity_path = self.logger.generate_component_entity_path(
                            self.component_name,
                            is_3d=True,
                            alg_name=alg_name,
                            subject_name=subject,
                        )
                        self._log_data(entity_path, subject_3d_points, "3d", alg_idx)
                        self._log_skeleton(
                            f"{entity_path}/skeleton",
                            subject_3d_points,
                            dimension=3,
                            alg_idx=alg_idx,
                        )
            else:
                cam_name = canvas
                data_key = [c for c in self.canvas_data if c != "3d"]
                camera_index = self.camera_names.index(cam_name)
                for k in data_key:
                    for alg_idx, alg_data in enumerate(self.canvas_data[k]):
                        if frame_idx >= alg_data.shape[2]:  # number of frames
                            continue
                        alg_name = self.algorithm_list[alg_idx]
                        for subject_idx, subject in enumerate(self.subject_names):
                            subject_2d_points = alg_data[
                                subject_idx, camera_index, frame_idx
                            ][:, :2]  # select first 2 values, 3rd is confidence score
                            entity_path = self.logger.generate_component_entity_path(
                                self.component_name,
                                is_3d=False,
                                alg_name=alg_name,
                                subject_name=subject,
                                cam_name=cam_name,
                            )
                            self._log_data(
                                entity_path, subject_2d_points, "2d", alg_idx
                            )
                            self._log_skeleton(
                                f"{entity_path}/skeleton",
                                subject_2d_points,
                                dimension=2,
                                alg_idx=alg_idx,
                            )


class HandJointsComponent(BodyJointsComponent):
    """
    Class for visualizing hand joints data.
    """

    def __init__(self, visualizer_config, io, logger, component_name):
        """
        Initialize the HandJointsComponent by calling the BodyJointsComponent
        constructor.

        Args:
            visualizer_config (dict): The visualizer configuration settings.
            io: The input/output object.
            logger: The logger object.
            component_name (str): The name of the component.
        """
        super().__init__(visualizer_config, io, logger, component_name)


class FaceLandmarksComponent(BodyJointsComponent):
    """
    Class for visualizing face landmarks data.
    """

    def __init__(self, visualizer_config, io, logger, component_name):
        super().__init__(visualizer_config, io, logger, component_name)


class GazeIndividualComponent(Component):
    """
    Class for visualizing individual gaze data.

    Attributes:
        calib (dict): The calibration parameters.
        camera_names (List[str]): The camera names.
        subject_names (List[str]): The subject names.
        landmarks_2d (np.ndarray): The 2D landmarks data.
        eyes_middle_3d_data (np.ndarray): The 3D eyes middle data.
        camera_view_subjects_middle_point_dict (Dict): The camera view subjects middle
            point dictionary.
        look_at_data (np.ndarray): The look at data.
        look_at_labels (List[str]): The look at labels.
        projected_gaze_data_algs (List[Dict]): The projected gaze data for the
            algorithms.
    """

    def __init__(
        self,
        visualizer_config: Dict,
        io,
        logger,
        component_name: str,
        calib: Dict,
        eyes_middle_3d_data: np.ndarray = None,
        look_at_data_tuple: bool = None,
    ):
        """
        Initialize the GazeIndividualComponent.

        Args:
            visualizer_config (Dict): The visualizer configuration settings.
            io: The input/output object.
            logger (viewer.Viewer): The viewer rerun object.
            component_name (str): The name of the component.
            calib (Dict): The calibration parameters.
            eyes_middle_3d_data (np.ndarray, optional): The 3D eyes middle data.
                Defaults to None.
            look_at_data_tuple (bool, optional): The look at data tuple.
                Defaults to None.
        """
        super().__init__(visualizer_config, io, logger, component_name)
        self.calib = calib
        # the camera_names and subject_names results will be read from first algorithm
        # we are getting camera names from landmarks_2d because 3d doesn't have any
        # camera info
        self.camera_names = self.algorithms_results[0]["data_description"].item()[
            "landmarks_2d"
        ]["axis1"]  # axis1 gives camera info
        self.subject_names = self.algorithms_results[0]["data_description"].item()[
            "3d"
        ]["axis0"]  # axis0 gives subject info
        self.landmarks_2d = self.algorithms_results[0]["landmarks_2d"]

        # create subjects middle of face
        # 3d
        self.eyes_middle_3d_data, _ = eyes_middle_3d_data
        # camera view
        # create the camera view -- middle of subjects' face point dictionary
        mean_face = np.nanmean(self.landmarks_2d.astype(float)[:, :, :, :4, :], axis=3)
        self.camera_view_subjects_middle_point_dict = {}
        for cam_idx, cam_name in enumerate(self.camera_names):
            subjects_middle_points = []
            for subject_idx, _subject in enumerate(self.subject_names):
                subjects_middle_points.append(mean_face[subject_idx, cam_idx, :])
            self.camera_view_subjects_middle_point_dict[cam_name] = (
                subjects_middle_points
            )

        self.look_at_data = None
        self.look_at_labels = None
        if look_at_data_tuple:
            self.look_at_data = look_at_data_tuple[0]
            self.look_at_labels = look_at_data_tuple[1]

        # retrieve 3d data projected to 2d camera views
        key_2d = (
            "2d_projected_from_3d_filtered"
            if "3d_filtered" in self.canvas_data
            else "2d_projected_from_3d"
        )
        self.projected_gaze_data_algs = []
        for alg_idx, _alg in enumerate(self.algorithm_list):
            proj = {}
            for cam_name in [c for c in self.canvas_list if "3d" not in c.lower()]:
                cam_idx = self.camera_names.index(cam_name)
                proj[cam_name] = self.algorithms_results[alg_idx][key_2d][:, cam_idx]
            self.projected_gaze_data_algs.append(proj)

    def _get_algorithms_labels(self) -> List[List[str]]:
        """
        Get the labels for the algorithms.

        Returns:
            List[List[str]]: The labels for the algorithms.
        """
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, _alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]["data_description"].item()["3d"]["axis3"]
            )
        return algorithm_labels

    def _get_look_at_color(
        self, sub_idx: int, alg_idx: int, look_to_subject: str, frame_idx: int
    ) -> List[int]:
        """
        Get the look at color for the subject index, algorithm index, look to subject,
        and frame index.

        Args:
            sub_idx (int): The subject index.
            alg_idx (int): The algorithm index.
            look_to_subject (str): The look to subject.
            frame_idx (int): The frame index.

        Returns:
            List[int]: The look at color.
        """
        look_to_label = f"look_at_{look_to_subject}"
        look_to_ind = self.look_at_labels.index(look_to_label)
        is_look_at = self.look_at_data[sub_idx, 0, frame_idx, look_to_ind]
        color_index = 0 if is_look_at else 1
        return self.visualizer_config["media"]["gaze_interaction"]["appearance"][
            "colors"
        ][alg_idx][color_index]

    def _log_data(
        self,
        entity_path: str,
        head_points: np.ndarray,
        data_points: np.ndarray,
        color: List[int],
        dimension: str,
    ) -> None:
        """
        Log the gaze points and head points in rerun.

        Args:
            entity_path (str): The entity path.
            head_points (np.ndarray): The head points.
            data_points (np.ndarray): The gaze points.
            color (List[int]): The color.
            dimension (str): The dimension.
        """
        if dimension == "2d":
            radii = self._parse_radii("camera_view")
            rr.log(
                entity_path,
                rr.Arrows2D(
                    origins=np.array(head_points).reshape(-1, 2),
                    vectors=np.array(data_points).reshape(-1, 2),
                    colors=np.array(color),
                    radii=radii,
                ),
            )
            rr.components.DrawOrder(1)

        elif dimension == "3d":
            radii = self._parse_radii("3d")
            rr.log(
                entity_path,
                rr.Arrows3D(
                    origins=np.array(head_points).reshape(-1, 3),
                    vectors=np.array(data_points).reshape(-1, 3)
                    / 2,  # divided by two to make it shorter in visualization
                    colors=np.array(color).reshape(-1, 3),
                    radii=radii,
                ),
            )

    def visualize(self, frame_idx: int) -> None:
        """
        Visualize the gaze individual component.

        Combines the _log_data method to visualize the gaze individual component in
        either 2D or 3D.

        Args:
            frame_idx (int): The frame index.
        """
        dataname = "3d_filtered" if "3d_filtered" in self.canvas_data else "3d"
        for canvas in self.canvas_list:
            if canvas == "3D_Canvas":
                for alg_idx, alg_data in enumerate(self.canvas_data[dataname]):
                    if frame_idx >= alg_data.shape[2]:  # number of frames
                        continue
                    alg_name = self.algorithm_list[alg_idx]
                    for subject_idx, subject in enumerate(self.subject_names):
                        subject_gaze_individual = -alg_data[subject_idx, 0, frame_idx]
                        subject_eyes_middle_3d_data = self.eyes_middle_3d_data[
                            subject_idx, 0, frame_idx
                        ]
                        entity_path = self.logger.generate_component_entity_path(
                            self.component_name,
                            is_3d=True,
                            alg_name=alg_name,
                            subject_name=subject,
                        )
                        # gaze interaction defines color
                        if self.look_at_data is not None:
                            if subject_idx + 1 < len(self.subject_names) - 1:
                                # look at subject either one forward or one backward in
                                # index
                                look_to_subject = self.subject_names[subject_idx + 1]
                            else:
                                look_to_subject = self.subject_names[subject_idx - 1]
                            color = self._get_look_at_color(
                                subject_idx, alg_idx, look_to_subject, frame_idx
                            )
                        else:
                            color = self.visualizer_config["media"]["gaze_individual"][
                                "appearance"
                            ]["colors"][alg_idx]
                        self._log_data(
                            entity_path,
                            subject_eyes_middle_3d_data,
                            subject_gaze_individual,
                            color,
                            "3d",
                        )
            else:
                cam_name = canvas
                for alg_idx, alg_data in enumerate(self.canvas_data[dataname]):
                    if frame_idx >= alg_data.shape[2]:  # number of frames
                        continue
                    alg_name = self.algorithm_list[alg_idx]
                    for subject_idx, subject in enumerate(self.subject_names):
                        if (
                            subject_idx
                            in self.visualizer_config["dataset_properties"][
                                "cam_sees_subjects"
                            ][cam_name]
                        ):
                            camera_data = self.projected_gaze_data_algs[alg_idx][canvas]
                            if frame_idx >= camera_data.shape[1]:  # number of frames
                                continue
                            frame_data = -camera_data[subject_idx, frame_idx]
                            subject_eyes_mid = (
                                self.camera_view_subjects_middle_point_dict[canvas][
                                    subject_idx
                                ][frame_idx][:2]
                            )
                            entity_path = self.logger.generate_component_entity_path(
                                self.component_name,
                                is_3d=False,
                                alg_name=alg_name,
                                subject_name=subject,
                                cam_name=cam_name,
                            )
                            # gaze interaction defines color
                            if self.look_at_data is not None:
                                if subject_idx + 1 < len(self.subject_names) - 1:
                                    # look at subject either one forward or one
                                    # backward in index
                                    look_to_subject = self.subject_names[
                                        subject_idx + 1
                                    ]
                                else:
                                    look_to_subject = self.subject_names[
                                        subject_idx - 1
                                    ]
                                color = self._get_look_at_color(
                                    subject_idx, alg_idx, look_to_subject, frame_idx
                                )
                            else:
                                color = self.visualizer_config["media"][
                                    self.component_name
                                ]["appearance"]["colors"][alg_idx]
                            self._log_data(
                                entity_path, subject_eyes_mid, frame_data, color, "2d"
                            )


class GazeInteractionComponent(Component):
    """
    Class for visualizing gaze interaction data.
    """

    def __init__(self, visualizer_config, io, logger, component_name):
        """
        Initialize the GazeInteractionComponent.

        Args:
            visualizer_config (Dict): The visualizer configuration settings.
            io: The input/output object.
            logger(viewer.Viewer): The viewer rerun object.
            component_name (str): The name of the component.
        """
        super().__init__(visualizer_config, io, logger, component_name)
        # selects first key - it might be distance_gaze_2d or distance_gaze_3d
        keyname = list(self.algorithms_results[0]["data_description"].item().keys())[0]
        self.camera_names = self.algorithms_results[0]["data_description"].item()[
            keyname
        ]["axis1"]
        self.subject_names = self.algorithms_results[0]["data_description"].item()[
            keyname
        ]["axis0"]

    def get_lookat_data(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get the look at data.

        Returns:
            Tuple[np.ndarray, List[str]]: The look at data and the look at labels.
        """
        # read from first algorithm
        if "gaze_look_at_3d" in self.algorithms_results[0]["data_description"].item():
            data_name = "gaze_look_at_3d"
        else:
            data_name = "gaze_look_at_2d"
        data_labels = self._get_algorithms_labels(data_name)[0]  # 0 first algorithm
        data = self.canvas_data[data_name][0]  # 0 first alg
        return (data, data_labels)

    def _get_algorithms_labels(self, data_name: str) -> List[List[str]]:
        """
        Get the labels for the algorithms.

        Args:
            data_name (str): The data name.

        Returns:
            List[List[str]]: The labels for the algorithms.
        """
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, _alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]["data_description"].item()[data_name][
                    "axis3"
                ]
            )
        return algorithm_labels

    def _log_data(self):
        pass

    def visualize(self):
        pass


class EmotionIndividualComponent(Component):
    """
    Class for visualizing emotion individual data.
    """

    def __init__(self, visualizer_config: Dict, io, logger, component_name: str):
        """
        Initialize the EmotionIndividualComponent.

        Args:
            visualizer_config (Dict): The visualizer configuration settings.
            io: The input/output object.
            logger (viewer.Viewer): The viewer rerun object.
            component_name (str): The name of the component.
        """
        super().__init__(visualizer_config, io, logger, component_name)
        # the camera_names and subject_names results will be read from first algorithm
        # we are getting camera names from landmarks_2d because 3d doesn't have any
        # camera info
        self.camera_names = self.algorithms_results[0]["data_description"].item()[
            "emotions"
        ]["axis1"]  # axis1 gives camera info
        self.subject_names = self.algorithms_results[0]["data_description"].item()[
            "emotions"
        ]["axis0"]  # axis0 gives subject info
        self.algorithm_labels = self._get_algorithms_labels()

    def _get_algorithms_labels(self) -> List[List[str]]:
        """
        Get the labels for the algorithms.

        Returns:
            List[List[str]]: The labels for the algorithms.
        """
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, _alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]["data_description"].item()["emotions"][
                    "axis3"
                ]
            )
        return algorithm_labels

    def _log_data(
        self, entity_path: str, head_bbox: np.ndarray, colors: str, labels: str
    ) -> None:
        """
        Log the face bounding box and emotion.

        Args:
            entity_path (str): The entity path.
            head_points (np.ndarray): The head points.
            data_points (np.ndarray): The gaze points.
            color (List[int]): The color.
            dimension (str): The dimension.
        """
        rr.log(
            entity_path,
            rr.Boxes2D(
                array=head_bbox,
                array_format=rr.Box2DFormat.XYWH,
                labels=labels,
                colors=colors,
            ),
        )

    def visualize(self, frame_idx: int) -> None:
        """
        Visualize the emotion individual component.

        Combines the _log_data and _log_annotation_context method to visualize the
        emotion individual component in camera views.

        Args:
            frame_idx (int): The frame index.
        """
        dataname = "emotions"
        head_bbox = "faceboxes"
        for canvas in self.canvas_list:
            cam_name = canvas
            camera_index = self.camera_names.index(cam_name)
            for alg_idx, alg_data in enumerate(self.canvas_data[dataname]):
                alg_colors = self._parse_alg_color(alg_idx)
                if frame_idx >= alg_data.shape[2]:  # number of frames
                    continue
                alg_name = self.algorithm_list[alg_idx]
                for subject_idx, subject in enumerate(self.subject_names):
                    if (
                        subject_idx
                        in self.visualizer_config["dataset_properties"][
                            "cam_sees_subjects"
                        ][cam_name]
                    ):
                        subject_head_bbox = self.algorithms_results[alg_idx][head_bbox][
                            subject_idx, camera_index, frame_idx
                        ]
                        subject_emotion_probability = alg_data[
                            subject_idx, camera_index, frame_idx
                        ]
                        max_probability_idx = np.argmax(subject_emotion_probability)
                        entity_path = self.logger.generate_component_entity_path(
                            self.component_name,
                            is_3d=False,
                            alg_name=alg_name,
                            subject_name=subject,
                            cam_name=cam_name,
                        )
                        self._log_data(
                            entity_path,
                            subject_head_bbox,
                            labels=self.algorithm_labels[alg_idx][max_probability_idx],
                            colors=alg_colors[max_probability_idx],
                        )


class HeadOrientationComponent(Component):
    """
    Class for visualizing head orientation data.
    """

    def __init__(self, visualizer_config: Dict, io, logger, component_name: str):
        """
        Initialize the HeadOrientationComponent.

        Args:
            visualizer_config (Dict): The visualizer configuration settings.
            io: The input/output object.
            logger (viewer.Viewer): The viewer rerun object.
            component_name (str): The name of the component.
        """
        super().__init__(visualizer_config, io, logger, component_name)
        # the camera_names and subject_names results will be read from first algorithm
        # we are getting camera names from landmarks_2d because 3d doesn't have any
        # camera info
        self.camera_names = self.algorithms_results[0]["data_description"].item()[
            "headpose"
        ]["axis1"]  # axis1 gives camera info
        self.subject_names = self.algorithms_results[0]["data_description"].item()[
            "headpose"
        ]["axis0"]  # axis0 gives subject info
        self.algorithm_labels = self._get_algorithms_labels()

    def _get_algorithms_labels(self) -> List[List[str]]:
        """
        Get the labels for the algorithms.

        Returns:
            List[List[str]]: The labels for the algorithms.
        """
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, _alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]["data_description"].item()[
                    "head_orientation_2d"
                ]["axis3"]
            )
        return algorithm_labels

    def _log_data(
        self,
        entity_path: str,
        head_points: np.ndarray,
        data_points: np.ndarray,
        color: List[int],
        dimension: str,
    ) -> None:
        """
        Log the head orientation points into.

        Args:
            entity_path (str): The entity path.
            head_points (np.ndarray): The head points.
            data_points (np.ndarray): The gaze points.
            color (List[int]): The color.
            dimension (str): The dimension.
        """
        vectors_forward = data_points[0:2] - head_points
        if dimension == "2d":
            radii = self._parse_radii("camera_view")
            rr.log(
                entity_path,
                rr.Arrows2D(
                    origins=np.array(head_points).reshape(-1, 2),
                    vectors=np.array(vectors_forward).reshape(-1, 2),
                    colors=np.array(color),
                    radii=radii,
                ),
            )
            rr.components.DrawOrder(1)

    def visualize(self, frame_idx: int) -> None:
        """
        Visualize the head orientation component.

        Combines the _log_data method to visualize the head orientation component in
        either 2D.

        Args:
            frame_idx (int): The frame index.
        """
        for canvas in self.canvas_list:
            cam_name = canvas
            camera_index = self.camera_names.index(cam_name)
            for alg_idx, alg_data in enumerate(self.canvas_data["head_orientation_2d"]):
                num_frames = alg_data.shape[2]
                if frame_idx >= num_frames:  # number of frames
                    continue
                alg_name = self.algorithm_list[alg_idx]
                for subject_idx, subject in enumerate(self.subject_names):
                    if (
                        subject_idx
                        in self.visualizer_config["dataset_properties"][
                            "cam_sees_subjects"
                        ][cam_name]
                    ):
                        frame_data = alg_data[subject_idx, camera_index, frame_idx]
                        entity_path = self.logger.generate_component_entity_path(
                            self.component_name,
                            is_3d=False,
                            alg_name=alg_name,
                            subject_name=subject,
                            cam_name=cam_name,
                        )

                        color = self.visualizer_config["media"][self.component_name][
                            "appearance"
                        ]["colors"][alg_idx]
                        self._log_data(
                            entity_path, frame_data[:2], frame_data[2:], color, "2d"
                        )


class ProximityComponent(Component):
    """
    Class for visualizing proximity data.
    """

    def __init__(
        self,
        visualizer_config: Dict,
        io,
        logger,
        component_name: str,
        eyes_middle_3d_data: Tuple[np.ndarray, List[str]] = None,
        eyes_middle_2d_data: Tuple[np.ndarray, List[str]] = None,
    ):
        """
        Initialize the ProximityComponent.

        Args:
            visualizer_config (Dict): The visualizer configuration settings.
            io: The input/output object.
            logger (viewer.Viewer): The viewer rerun object.
            component_name (str): The name of the component.
            eyes_middle_3d_data (Tuple[np.ndarray, List[str]], optional):
                The 3D eyes middle data. Defaults to None.
            eyes_middle_2d_data (Tuple[np.ndarray, List[str]], optional):
                The 2D eyes middle data. Defaults to None.
        """
        super().__init__(visualizer_config, io, logger, component_name)
        self.camera_names = self.algorithms_results[0]["data_description"].item()[
            "body_distance_2d"
        ]["axis1"]
        self.subject_names = self.algorithms_results[0]["data_description"].item()[
            "body_distance_2d"
        ]["axis0"]

        # create subjects middle data
        # 3d
        self.eyes_middle_3d_data, _ = eyes_middle_3d_data
        if self.eyes_middle_3d_data is not None:
            first_subject_eyes_middle_data = self.eyes_middle_3d_data[0, 0, :].mean(
                axis=0
            )
            second_subject_eyes_middle_data = self.eyes_middle_3d_data[1, 0, :].mean(
                axis=0
            )
            self.middle_point_3d = (
                first_subject_eyes_middle_data + second_subject_eyes_middle_data
            ) / 2
        # 2d - camera view
        self.eyes_middle_2d_data, _ = eyes_middle_2d_data
        # create camera view - middle point dictionary
        self.camera_view_middle_point_dict = {}
        for cam in self.camera_names:
            camera_idx = self.camera_names.index(cam)
            first_subject_eyes_middle_data = self.eyes_middle_2d_data[
                0, camera_idx, :
            ].mean(axis=0)
            second_subject_eyes_middle_data = self.eyes_middle_2d_data[
                1, camera_idx, :
            ].mean(axis=0)
            middle_point = (
                first_subject_eyes_middle_data + second_subject_eyes_middle_data
            ) / 2
            self.camera_view_middle_point_dict[cam] = middle_point

    def _get_algorithms_labels(self) -> List[List[str]]:
        """
        Get the labels for the algorithms.

        Returns:
            List[List[str]]: The labels for the algorithms.
        """
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, _alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]["data_description"].item()[
                    "body_distance_2d"
                ]["axis3"]
            )
        return algorithm_labels

    def _log_data(
        self,
        entity_path: str,
        data_points: np.ndarray,
        alg_idx: int,
        mid_point: np.ndarray,
        dimension: str,
    ) -> None:
        """
        Logs the proximity score in rerun.

        Args:
            entity_path (str): The entity path.
            data_points (np.ndarray): The data points.
            alg_idx (int): The algorithm index.
            mid_point (np.ndarray): The middle point.
            dimension (str): The dimension.
        """
        color = self._parse_alg_color(alg_idx)
        if dimension == "2d":
            radii = self._parse_radii("camera_view")
            proximity_start = np.array(
                [mid_point[0] - (data_points / 2), mid_point[1] - 100]
            )
            proximity_end = np.array(
                [mid_point[0] + (data_points / 2), mid_point[1] - 100]
            )
            rr.log(
                entity_path,
                rr.LineStrips2D(
                    np.vstack((proximity_start, proximity_end)),
                    colors=color,
                    radii=radii,
                    labels="Proximity",
                ),
            )
        elif dimension == "3d":
            radii = self._parse_radii("3d")
            proximity_start = np.array(
                [mid_point[0] - (data_points / 2), mid_point[1] - 0.5, mid_point[2]]
            )
            proximity_end = np.array(
                [mid_point[0] + (data_points / 2), mid_point[1] - 0.5, mid_point[2]]
            )
            rr.log(
                entity_path,
                rr.LineStrips3D(
                    np.vstack((proximity_start, proximity_end)),
                    colors=color,
                    radii=radii,
                    labels="Proximity",
                ),
            )

    def visualize(self, frame_idx: int) -> None:
        """
        Visualize the proximity component.

        Uses the _log_data method to visualize the proximity component in either 2D or
        3D.

        Args:
            frame_idx (int): The frame index.
        """
        for canvas in self.canvas_list:
            if canvas == "3D_Canvas":
                for alg_idx, alg_data in enumerate(
                    self.canvas_data["body_distance_3d"]
                ):
                    alg_name = self.algorithm_list[alg_idx]
                    if frame_idx >= alg_data.shape[2]:  # number of frames
                        continue
                    frame_proximity = alg_data[:, 0, frame_idx, 0][0]
                    entity_path = self.logger.generate_component_entity_path(
                        self.component_name, is_3d=True, alg_name=alg_name
                    )
                    self._log_data(
                        entity_path,
                        frame_proximity,
                        alg_idx,
                        self.middle_point_3d,
                        "3d",
                    )
            else:
                cam_name = canvas
                for alg_idx, alg_data in enumerate(
                    self.canvas_data["body_distance_2d"]
                ):
                    alg_name = self.algorithm_list[alg_idx]
                    camera_idx = self.camera_names.index(canvas)
                    if frame_idx >= alg_data.shape[2]:  # number of frames
                        continue
                    frame_proximity = alg_data[:, camera_idx, frame_idx, 0][0]
                    entity_path = self.logger.generate_component_entity_path(
                        self.component_name,
                        is_3d=False,
                        alg_name=alg_name,
                        cam_name=cam_name,
                    )
                    mid_point = self.camera_view_middle_point_dict[cam_name]
                    self._log_data(
                        entity_path, frame_proximity, alg_idx, mid_point, "2d"
                    )


class KinematicsComponent(Component):
    """
    Class for visualizing kinematics data.
    """

    def __init__(self, visualizer_config: Dict, io, logger, component_name: str):
        """
        Initialize the KinematicsComponent.

        Args:
            visualizer_config (Dict): The visualizer configuration settings.
            io: The input/output object.
            logger (viewer.Viewer): The viewer rerun object.
            component_name (str): The name of the component.
        """
        super().__init__(visualizer_config, io, logger, component_name)
        self.camera_names = self.algorithms_results[0]["data_description"].item()[
            "velocity_body_2d"
        ]["axis1"]
        self.subject_names = self.algorithms_results[0]["data_description"].item()[
            "velocity_body_2d"
        ]["axis0"]

    def _get_algorithms_labels(self) -> List[List[str]]:
        """
        Get the labels for the algorithms.

        Returns:
            List[List[str]]: The labels for the algorithms.
        """
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, _alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]["data_description"].item()[
                    "velocity_body_2d"
                ]["axis3"]
            )
        return algorithm_labels

    def _get_joints_movement_by_bodypart(self, alg_idx: int, data_name: str) -> np.ndarray:
        """
        Get the joints movement by body part for the algorithm index.

        Args:
            alg_idx (int): The algorithm index.
        """
        bodypart_motion = []
        labels = self._get_algorithms_labels()[alg_idx]
        for _bodypart, joints in self.visualizer_config["media"][self.component_name][
            "joints"
        ].items():
            data = self.algorithms_data[alg_idx]
            joint_indices = [labels.index(joint) for joint in joints]
            # THRESHOLD = 0.3
            # data[..., :, 0][data[..., : , 0]< THRESHOLD] = 0
            selected_data = data[:, :, :, joint_indices, 0:1] # if use only 0, instead of 0:1, last dimension drops
            bodypart_motion.append(np.nanmean(selected_data, axis=-2))
        bodypart_motion = np.concatenate(bodypart_motion, axis=-1)
        return bodypart_motion

    def _log_data(self, entity_path: str, data_points: np.ndarray) -> None:
        """
        Log the data points in rerun.

        Args:
            entity_path (str): The entity path.
            data_points (np.ndarray): The data points.
        """
        rr.log(entity_path, rr.Scalar(np.round(data_points, decimals=2)))

    def visualize(self, frame_idx: int) -> None:
        """
        Visualize the kinematics component.

        Uses the _log_data method to visualize the kinematics component.

        Args:
            frame_idx (int): The frame index.
        """
        for data_name, data in self.canvas_data.items():
            for alg_idx, _alg_data in enumerate(data):
                alg_name = self.algorithm_list[alg_idx]
                for subject_idx, subject in enumerate(self.subject_names):
                    joints_bodypart_motion = self._get_joints_movement_by_bodypart(
                        alg_idx, data_name
                    )
                    for idx, bodypart in enumerate(
                        self.visualizer_config["media"][self.component_name][
                            "joints"
                        ].keys()
                    ):
                        if data_name == "velocity_body_3d":
                            if (
                                frame_idx >= joints_bodypart_motion.shape[2]
                            ):  # number of frames
                                continue
                            frame_bodypart_data = joints_bodypart_motion[
                                subject_idx, 0, frame_idx
                            ][idx]
                            entity_path = self.logger.generate_component_entity_path(
                                self.component_name,
                                is_3d=True,
                                alg_name=alg_name,
                                subject_name=subject,
                                bodypart=bodypart,
                            )
                            self._log_data(entity_path, frame_bodypart_data)

                        elif data_name == "velocity_body_2d":
                            for camera_idx, camera in enumerate(self.camera_names):
                                if (
                                    frame_idx >= joints_bodypart_motion.shape[2]
                                ):  # number of frames
                                    continue
                                frame_kinematic_2d = joints_bodypart_motion[
                                    subject_idx, camera_idx, frame_idx
                                ][idx]
                                entity_path = (
                                    self.logger.generate_component_entity_path(
                                        self.component_name,
                                        is_3d=False,
                                        alg_name=alg_name,
                                        subject_name=subject,
                                        cam_name=camera,
                                        bodypart=bodypart,
                                    )
                                )
                                self._log_data(entity_path, frame_kinematic_2d)
