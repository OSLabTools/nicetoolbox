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

from ...utils import visual_utils as vis_utils


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
            visualizer_config["io"]["video_name"], component_name=component_name
        )
        self.algorithm_list = self.visualizer_config["media"][self.component_name]["algorithms"]

        # get canvas list from visualizer_config
        canvas_list = []
        for canvases in self.visualizer_config["media"][self.component_name]["canvas"].values():
            canvas_list.extend(canvases)
        self.canvas_list = list(set(canvas_list))

        # load algorithm results
        self.algorithms_results = []
        for alg in self.algorithm_list:
            alg_path = io.get_algorithm_result(self.component_prediction_folder, alg)
            try:
                self.algorithms_results.append(np.load(alg_path, allow_pickle=True))
            except FileNotFoundError:
                print(
                    f"ERROR: {alg}.npz file is not found in {self.component_name} folder."
                    f"It will not be visualized\n  "
                    f"Remove {alg} or {self.component_name} in the visualizer_config.toml file"
                )
                raise

        # create canvas data dictionary - key is data name, and value is algorithms data
        # (lists of algorithms results)
        self.canvas_data = {}
        for data_name, canvas in self.visualizer_config["media"][self.component_name]["canvas"].items():
            if canvas != []:
                self.algorithms_data = []
                if self.algorithms_results:
                    for i, _alg in enumerate(self.algorithm_list):
                        try:
                            self.algorithms_data.append(self.algorithms_results[i][data_name])
                        except KeyError:
                            print(f"WARNING! {self.component_name}: '{data_name}' cannot be found, will be skipped.")
                    self.canvas_data[data_name] = self.algorithms_data

    def _parse_alg_color(self, alg_idx: int) -> List[int]:
        """
        Parse the color for the algorithm index.

        Args:
            alg_idx (int): The index of the algorithm.

        Returns:
            list: The color for the algorithm index.
        """
        return self.visualizer_config["media"][self.component_name]["appearance"]["colors"][alg_idx]

    def _parse_radii(self, type: str) -> float:
        """
        Parse the radii for the type. Type is one of '3d' or 'camera_view'.

        Args:
            type (str): The type of radii.

        Returns:
            float: The radii for the type.
        """
        if type == "3d":
            return self.visualizer_config["media"][self.component_name]["appearance"]["radii"]["3d"]
        if type == "camera_view":
            return self.visualizer_config["media"][self.component_name]["appearance"]["radii"]["camera_view"]
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
        self.subject_names_2d = self.algorithms_results[0]["data_description"].item()["2d"]["axis0"]
        if "3D_Canvas" in self.canvas_list:
            if "3d" in self.algorithms_results[0]:
                self.subject_names_3d = self.algorithms_results[0]["data_description"].item()["3d"]["axis0"]
            else:
                print("WARNING! velocity_body_3d cannot be found,will be skipped.")
                self.canvas_list.remove("3D_Canvas")
        # data description axis1 gives camera information
        self.camera_names = self.algorithms_results[0]["data_description"].item()["2d"]["axis1"]

    def calculate_middle_eyes(self) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Calculate the middle of the eyes for the both dimensions.
        If has only one camera, then 3d returns as None

        Returns:
            Tuple[np.ndarray, np.ndarray]: The middle eyes 2d and 3d data.
        """
        # we will use first algorithm results
        labels = self._get_algorithms_labels()[0]
        right_eye_idx = labels.index("right_eye")
        left_eye_idx = labels.index("left_eye")

        # compute 2d
        data_2d = self.algorithms_results[0]["2d"]
        mean_value_2d = np.mean(data_2d[:, :, :, [right_eye_idx, left_eye_idx], :2], axis=3)

        if len(self.camera_names) == 1:
            return mean_value_2d, None

        if "3d" not in self.canvas_data:
            print("3D results not found in canvas data — skipping 3D calculation\n")
            return mean_value_2d, None
        # compute 3d
        data_3d = self.algorithms_results[0]["3d"]
        mean_value_3d = np.mean(data_3d[:, :, :, [right_eye_idx, left_eye_idx], :3], axis=3)
        return mean_value_2d, mean_value_3d

    def _get_algorithms_labels(self) -> List[List[str]]:
        """
        Get the labels for the algorithms.

        Returns:
            List[List[str]]: The labels for the algorithms.
        """
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, _alg in enumerate(self.algorithm_list):
            algorithm_labels.append(self.algorithms_results[i]["data_description"].item()["2d"]["axis3"])
        return algorithm_labels

    def _get_skeleton_connections(self, alg_idx: int, predictions_mapping: Dict) -> List[List[str]]:
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
        alg_type = self.visualizer_config["algorithms_properties"][alg_name]["keypoint_mapping"]
        return predictions_mapping["human_pose"][alg_type]["connections"][self.component_name]

    def _log_skeleton(self, entity_path: str, data_points: np.ndarray, dimension: int, alg_idx: int) -> None:
        """
        Log the skeleton data points in rerun.

        Args:
            entity_path (str): The entity path.
            data_points (np.ndarray): The data points.
            dimension (int): The dimension.
            alg_idx (int): The algorithm index.
        """
        keypoints_dict = {label: i for i, label in enumerate(self._get_algorithms_labels()[alg_idx])}
        connections = self._get_skeleton_connections(alg_idx, self.visualizer_config["predictions_mapping"])
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
            radii = self.visualizer_config["media"][self.component_name]["appearance"]["radii"]["camera_view"]
            rr.log(
                entity_path,
                rr.LineStrips2D(
                    np.stack((start_points, end_points), axis=1),
                    colors=color,
                    radii=radii,
                ),
            )
        else:
            radii = self.visualizer_config["media"][self.component_name]["appearance"]["radii"]["3d"]
            rr.log(
                entity_path,
                rr.LineStrips3D(
                    np.stack((start_points, end_points), axis=1),
                    colors=color,
                    radii=radii,
                ),
            )

    def _log_data(self, entity_path: str, data_points: np.ndarray, dimension: int, alg_idx: int) -> None:
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
            if not canvas:
                continue
            if canvas == "3D_Canvas":
                for alg_idx, alg_data in enumerate(self.canvas_data["3d"]):
                    if frame_idx >= alg_data.shape[2]:  # number of frames
                        continue
                    alg_name = self.algorithm_list[alg_idx]
                    for subject_idx, subject in enumerate(self.subject_names_3d):
                        subject_3d_points = alg_data[subject_idx, 0, frame_idx][
                            :, :3
                        ]  # select first 3 values, 4th is confidence score
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
                        for subject_idx, subject in enumerate(self.subject_names_2d):
                            subject_2d_points = alg_data[subject_idx, camera_index, frame_idx][
                                :, :2
                            ]  # select first 2 values, 3rd is confidence score
                            entity_path = self.logger.generate_component_entity_path(
                                self.component_name,
                                is_3d=False,
                                alg_name=alg_name,
                                subject_name=subject,
                                cam_name=cam_name,
                            )
                            self._log_data(entity_path, subject_2d_points, "2d", alg_idx)
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


class GazeFusionComponent(Component):
    """
    Class for visualizing fused gaze.

    Each configured algorithm is a `gaze_fusion` instance (e.g. gaze_fusion_weighted,
    gaze_fusion_per_subject) — the component draws one overlay per instance so different
    fusion strategies can be compared side-by-side. All data (fused 3D direction,
    per-camera 2D reprojection, 2D face origin) is read from each fusion NPZ directly.

    Attributes:
        calib (dict): The calibration parameters.
        camera_names (List[str]): The camera names.
        subject_names (List[str]): The subject names.
        eyes_middle_3d_data (np.ndarray): The 3D eyes middle data (from body_joints).
        look_at_data (np.ndarray): The look at data.
        look_at_labels (List[str]): The look at labels.
        origins_per_alg (List[np.ndarray]): Per-camera 2D face origin per algorithm (S, C, F, 2).
        fused_gaze_3d_per_alg (List[np.ndarray]): Fused 3D gaze per algorithm (S, 1, F, 4).
        projected_gaze_2d_per_alg (List[np.ndarray]): Reprojected 2D gaze per algorithm (S, C, F, 3).
    """

    # Fallback arrow length (px) for drawing unit gaze directions when calibration is unavailable.
    DEFAULT_ARROW_LENGTH_PX = 200.0

    def __init__(
        self,
        visualizer_config: Dict,
        io,
        logger,
        component_name: str,
        calib: Dict,
        eyes_middle_3d_data: np.ndarray = None,
        look_at_data_tuples: List[Tuple[np.ndarray, List[str]]] = None,
    ):
        super().__init__(visualizer_config, io, logger, component_name)
        self.calib = calib

        # All arrays live in the fusion NPZ (each algorithm in self.algorithms_results
        # is one gaze_fusion instance). Cameras and subjects are stable across instances.
        first = self.algorithms_results[0]
        descr = first["data_description"].item()
        self.camera_names = descr["gaze_2d"]["axis1"]
        self.subject_names = descr["gaze_3d"]["axis0"]

        self.eyes_middle_3d_data = eyes_middle_3d_data

        # Per-algorithm look-at pair (aligned positionally with self.algorithm_list). Empty
        # list means every algorithm falls back to static color.
        self.look_at_per_alg: List[Tuple[np.ndarray, List[str]]] = []
        if look_at_data_tuples:
            if len(look_at_data_tuples) != len(self.algorithm_list):
                raise ValueError(
                    f"gaze_fusion has {len(self.algorithm_list)} algorithms but gaze_interaction "
                    f"provided {len(look_at_data_tuples)} look-at entries. The two algorithms lists "
                    f"must be the same length (each fusion paired with its own gaze_distance)."
                )
            self.look_at_per_alg = look_at_data_tuples

        # Cache per-algorithm arrays (each aligned to same subject/camera order via the
        # base component NPZ layout). Prefer filtered variants when the fusion emits them.
        self.origins_per_alg = []
        self.fused_gaze_3d_per_alg = []
        self.projected_gaze_2d_per_alg = []
        for alg_result in self.algorithms_results:
            files = set(alg_result.files)
            key_3d = "gaze_3d_filtered" if "gaze_3d_filtered" in files else "gaze_3d"
            key_2d = "gaze_2d_filtered" if "gaze_2d_filtered" in files else "gaze_2d"
            self.origins_per_alg.append(alg_result["gaze_origin_2d"][..., :2].astype(float))
            self.fused_gaze_3d_per_alg.append(alg_result[key_3d])
            self.projected_gaze_2d_per_alg.append(alg_result[key_2d])

    def _get_algorithms_labels(self) -> List[List[str]]:
        """Labels for the fused gaze axis3 (per algorithm)."""
        return [res["data_description"].item()["gaze_3d"]["axis3"] for res in self.algorithms_results]

    def _arrow_length(self, cam_name: str) -> float:
        """Pixel length for drawing a unit gaze direction in `cam_name`'s view.

        gaze_2d is stored normalized, so it needs a pixel scale to render as a visible arrow.
        Calibration is optional in the visualizer, so fall back to a fixed length when this
        camera's image size is unknown.
        """
        if self.calib and cam_name in self.calib:
            return vis_utils.gaze_arrow_pixel_length(self.calib[cam_name]["image_size"][0])
        return self.DEFAULT_ARROW_LENGTH_PX

    def _get_look_at_color(self, sub_idx: int, alg_idx: int, look_to_subject: str, frame_idx: int) -> List[int]:
        """
        Get the look at color for the subject/frame from the paired gaze_distance instance
        at position `alg_idx`.
        """
        look_at_data, look_at_labels = self.look_at_per_alg[alg_idx]
        look_to_ind = look_at_labels.index(look_to_subject)
        is_look_at = look_at_data[sub_idx, 0, frame_idx, look_to_ind]
        color_index = 0 if is_look_at else 1
        return self.visualizer_config["media"]["gaze_interaction"]["appearance"]["colors"][alg_idx][color_index]

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
        Visualize each configured gaze_fusion instance as its own overlay.

        Every algorithm in self.algorithm_list produces a full set of arrows (one per subject)
        drawn with that algorithm's color. Look-at coloring, when enabled, overrides the
        static color only for alg_idx=0 (there is a single gaze_distance instance driving it).

        Args:
            frame_idx (int): The frame index.
        """
        for canvas in self.canvas_list:
            if canvas == "3D_Canvas" and self.eyes_middle_3d_data is not None:
                for alg_idx, alg_name in enumerate(self.algorithm_list):
                    fused_gaze_3d = self.fused_gaze_3d_per_alg[alg_idx]
                    if frame_idx >= fused_gaze_3d.shape[2]:
                        continue
                    for subject_idx, subject in enumerate(self.subject_names):
                        subject_gaze = -fused_gaze_3d[subject_idx, 0, frame_idx, :3]
                        subject_eyes_middle_3d_data = self.eyes_middle_3d_data[subject_idx, 0, frame_idx]
                        entity_path = self.logger.generate_component_entity_path(
                            self.component_name,
                            is_3d=True,
                            alg_name=alg_name,
                            subject_name=subject,
                        )
                        color = self._pick_color(alg_idx, subject_idx, frame_idx)
                        self._log_data(
                            entity_path,
                            subject_eyes_middle_3d_data,
                            subject_gaze,
                            color,
                            "3d",
                        )
            else:
                cam_name = canvas
                for alg_idx, alg_name in enumerate(self.algorithm_list):
                    projected = self.projected_gaze_2d_per_alg[alg_idx]
                    origins = self.origins_per_alg[alg_idx]
                    if frame_idx >= projected.shape[2]:
                        continue
                    cam_idx = self.camera_names.index(cam_name)
                    for subject_idx, subject in enumerate(self.subject_names):
                        cam = self.visualizer_config["dataset_properties"]["video"]["cameras"][cam_name]
                        if subject_idx not in cam["sees_subjects"]:
                            continue
                        origin = origins[subject_idx, cam_idx, frame_idx, :2]
                        # gaze_2d carries [x, y, conf]; drop conf. It is a unit direction, so
                        # scale it to a pixel arrow for this camera's image width.
                        gaze_2d = projected[subject_idx, cam_idx, frame_idx, :2] * self._arrow_length(cam_name)
                        entity_path = self.logger.generate_component_entity_path(
                            self.component_name,
                            is_3d=False,
                            alg_name=alg_name,
                            subject_name=subject,
                            cam_name=cam_name,
                        )
                        color = self._pick_color(alg_idx, subject_idx, frame_idx)
                        self._log_data(entity_path, origin, gaze_2d, color, "2d")

    def _pick_color(self, alg_idx: int, subject_idx: int, frame_idx: int) -> List[int]:
        """Look-at color if a paired gaze_distance instance exists for this alg_idx; otherwise
        the algorithm's static color from this component's appearance block."""
        if alg_idx < len(self.look_at_per_alg):
            if subject_idx + 1 < len(self.subject_names) - 1:
                look_to_subject = self.subject_names[subject_idx + 1]
            else:
                look_to_subject = self.subject_names[subject_idx - 1]
            return self._get_look_at_color(subject_idx, alg_idx, look_to_subject, frame_idx)
        return self.visualizer_config["media"][self.component_name]["appearance"]["colors"][alg_idx]


class EyeClosureComponent(Component):
    """
    Class for visualizing eye closure (EAR) data.

    Draws a closed contour around each eye, mirroring the eye outlines the eye_closure_ear
    detector draws on its own mp4.

    Everything is read from the component NPZ: `eye_landmarks_2d` carries the eye points
    already sliced out by the detector, labelled `left_eye_*` / `right_eye_*` on axis3, so
    splitting the points by label prefix is enough — no keypoint mapping lookup needed.

    When an eye_closed_state instance is paired in (via closed_state_tuples), each contour is
    colored by that eye's closed/open state instead of the algorithm's static color.

    The `score` array is additionally plotted as a scalar timeseries, one plot per tracked
    (subject, camera) pair with a line per eye. Pairs the detector never tracked are all-NaN
    and are dropped rather than plotted empty.

    The contours are camera-view only; the detector produces no 3D eye data.

    Attributes:
        camera_names (List[str]): The camera names.
        subject_names (List[str]): The subject names.
        eye_indices_per_alg (List[Tuple[List[int], List[int]]]): Per algorithm, the
            (left, right) positions within the axis3 landmark labels.
        closed_state_per_alg (List[Tuple[np.ndarray, List[str]]]): Per-algorithm closed-state
            pair (data, eye_labels). Empty when no state component is wired.
        state_camera_names (List[str]): The closed-state camera axis, which may be a different
            subset than this component's own. Empty when no state component is wired.
    """

    LANDMARKS_KEY = "eye_landmarks_2d"
    SCORE_KEY = "score"

    def __init__(
        self,
        visualizer_config: Dict,
        io,
        logger,
        component_name: str,
        closed_state_tuples: List[Tuple[np.ndarray, List[str]]] = None,
        state_camera_names: List[str] = None,
    ):
        """
        Initialize the EyeClosureComponent.

        Args:
            visualizer_config (Dict): The visualizer configuration settings.
            io: The input/output object.
            logger (viewer.Viewer): The viewer rerun object.
            component_name (str): The name of the component.
            closed_state_tuples (List[Tuple[np.ndarray, List[str]]], optional): Per-algorithm
                (closed_state, eye_labels) from eye_closed_state, positionally aligned with
                this component's algorithms. Defaults to None (static coloring).
            state_camera_names (List[str], optional): The closed-state camera axis, used to
                index the state arrays by camera name. Defaults to None.
        """
        super().__init__(visualizer_config, io, logger, component_name)

        descr = self.algorithms_results[0]["data_description"].item()
        self.camera_names = descr[self.LANDMARKS_KEY]["axis1"]
        self.subject_names = descr[self.LANDMARKS_KEY]["axis0"]

        # Per-algorithm closed-state pair (aligned positionally with self.algorithm_list). An
        # empty list means every algorithm falls back to its static color.
        self.closed_state_per_alg: List[Tuple[np.ndarray, List[str]]] = []
        self.state_camera_names: List[str] = []
        if closed_state_tuples:
            if len(closed_state_tuples) != len(self.algorithm_list):
                raise ValueError(
                    f"eye_closure_score has {len(self.algorithm_list)} algorithms but "
                    f"eye_closed_state provided {len(closed_state_tuples)} entries. The two "
                    f"algorithms lists must be the same length (each score instance paired "
                    f"with its own threshold instance)."
                )
            self.closed_state_per_alg = closed_state_tuples
            self.state_camera_names = state_camera_names or []

        # Per-algorithm EAR score, plotted as a timeseries alongside the contours. Skipped
        # entirely when the score canvas is empty, and per-algorithm when a score producer
        # emits only landmarks.
        plot_scores = bool(self.visualizer_config["media"][self.component_name]["canvas"].get(self.SCORE_KEY))
        self.score_per_alg: List[np.ndarray | None] = [
            res[self.SCORE_KEY] if plot_scores and self.SCORE_KEY in res.files else None
            for res in self.algorithms_results
        ]
        self.score_eye_labels_per_alg: List[List[str]] = [
            res["data_description"].item()[self.SCORE_KEY]["axis3"] if self.score_per_alg[i] is not None else []
            for i, res in enumerate(self.algorithms_results)
        ]
        # A (subject, camera) pair a detector never saw is all-NaN -- e.g. the left subject on
        # the right-facing camera. Plotting it would add a permanently empty view, so the
        # tracked pairs are resolved once here and the rest are skipped.
        self.plotted_series_per_alg = [
            self._resolve_plotted_series(alg_idx) for alg_idx in range(len(self.score_per_alg))
        ]

        # The landmark array is already sliced to the eye points, so the eyes are split by
        # label prefix. Point order within each eye is the upstream landmark order, which walks
        # the eyelid ring corner-to-corner (upper lid, then back along the lower lid) for every
        # mapping currently shipped -- so connecting them in order closes the contour. A new
        # mapping that numbers eye points differently would need an explicit ring order here.
        self.eye_indices_per_alg = []
        for labels in self._get_algorithms_labels():
            left = [i for i, name in enumerate(labels) if name.startswith("left_eye")]
            right = [i for i, name in enumerate(labels) if name.startswith("right_eye")]
            self.eye_indices_per_alg.append((left, right))

    def _get_algorithms_labels(self) -> List[List[str]]:
        """Eye landmark labels (axis3) per algorithm."""
        return [res["data_description"].item()[self.LANDMARKS_KEY]["axis3"] for res in self.algorithms_results]

    def _resolve_plotted_series(self, alg_idx: int) -> List[Tuple[int, int]]:
        """
        Resolve which (subject_idx, camera_idx) score series are worth plotting.

        A pair whose score is NaN for every frame means the detector never tracked that subject
        on that camera -- e.g. the left-seated subject on the right-facing camera -- so it is
        dropped rather than logged as a permanently empty plot.

        Returns:
            List of (subject_idx, camera_idx) pairs that carry at least one real score.
        """
        score = self.score_per_alg[alg_idx]
        if score is None:
            return []

        # Any non-NaN across the frame and eye axes means the pair was tracked at some point.
        tracked = ~np.isnan(score).all(axis=(2, 3))  # (subjects, cameras)
        return [(int(s), int(c)) for s, c in zip(*np.nonzero(tracked))]

    def _log_score(self, entity_path: str, score: float) -> None:
        """
        Log one eye's EAR score as a scalar timeseries point in rerun.

        Args:
            entity_path (str): The entity path.
            score (float): The EAR score.
        """
        rr.log(entity_path, rr.Scalar(round(float(score), 3)))

    def _pick_color(self, alg_idx: int, subject_idx: int, camera_idx: int, frame_idx: int, eye_name: str) -> List[int]:
        """
        Closed/open color if a paired eye_closed_state instance exists for this alg_idx;
        otherwise the algorithm's static color from this component's appearance block.

        The state array is float-backed so a missing upstream score reads as NaN — neither
        closed nor open — and falls back to the static color rather than claiming "open".
        """
        if alg_idx >= len(self.closed_state_per_alg) or camera_idx < 0:
            return self._parse_alg_color(alg_idx)

        closed_state, eye_labels = self.closed_state_per_alg[alg_idx]
        if eye_name not in eye_labels or frame_idx >= closed_state.shape[2]:
            return self._parse_alg_color(alg_idx)

        eye_idx = eye_labels.index(eye_name)
        is_closed = closed_state[subject_idx, camera_idx, frame_idx, eye_idx]
        if np.isnan(is_closed):
            return self._parse_alg_color(alg_idx)

        color_index = 0 if is_closed else 1
        return self.visualizer_config["media"]["eye_closed_state"]["appearance"]["colors"][alg_idx][color_index]

    def _log_data(self, entity_path: str, eye_points: np.ndarray, color: List[int]) -> None:
        """
        Log one eye contour as a closed 2D line strip in rerun.

        Args:
            entity_path (str): The entity path.
            eye_points (np.ndarray): The eye's (n_points, 2) contour points.
            color (List[int]): The color.
        """
        # Repeat the first point so the eyelid ring closes.
        contour = np.vstack([eye_points, eye_points[:1]])
        rr.log(
            entity_path,
            rr.LineStrips2D(
                [contour],
                colors=color,
                radii=self._parse_radii("camera_view"),
            ),
        )

    def visualize(self, frame_idx: int) -> None:
        """
        Visualize the eye closure component on each camera view.

        Each eye is drawn as its own entity so the two contours stay independently
        addressable in the viewer, and each tracked (subject, camera) EAR score is logged as
        its own scalar timeseries.

        Args:
            frame_idx (int): The frame index.
        """
        self._plot_scores(frame_idx)

        for canvas in self.canvas_list:
            if not canvas or canvas == "3D_Canvas":
                continue
            cam_name = canvas
            if cam_name not in self.camera_names:
                continue
            camera_index = self.camera_names.index(cam_name)
            # The threshold detector has its own camera_names, so its camera axis may be a
            # different subset. Resolve the state's own index by name; -1 disables recoloring
            # for this camera and falls back to the static color.
            state_camera_index = self.state_camera_names.index(cam_name) if cam_name in self.state_camera_names else -1

            for alg_idx, alg_result in enumerate(self.algorithms_results):
                landmarks = alg_result[self.LANDMARKS_KEY]
                if frame_idx >= landmarks.shape[2]:  # number of frames
                    continue
                alg_name = self.algorithm_list[alg_idx]
                left_indices, right_indices = self.eye_indices_per_alg[alg_idx]

                for subject_idx, subject in enumerate(self.subject_names):
                    cam = self.visualizer_config["dataset_properties"]["video"]["cameras"][cam_name]
                    if subject_idx not in cam["sees_subjects"]:
                        continue

                    entity_path = self.logger.generate_component_entity_path(
                        self.component_name,
                        is_3d=False,
                        alg_name=alg_name,
                        subject_name=subject,
                        cam_name=cam_name,
                    )

                    for eye_name, indices in (("left_eye", left_indices), ("right_eye", right_indices)):
                        eye_points = landmarks[subject_idx, camera_index, frame_idx][indices, :2]
                        # Drop missing points; a partially tracked eye still draws.
                        eye_points = eye_points[~np.isnan(eye_points).any(axis=1)]
                        if len(eye_points) < 2:
                            continue

                        color = self._pick_color(alg_idx, subject_idx, state_camera_index, frame_idx, eye_name)
                        self._log_data(f"{entity_path}/{eye_name}", eye_points, color)

    def _plot_scores(self, frame_idx: int) -> None:
        """
        Log the EAR score timeseries for every tracked (subject, camera) pair.

        Independent of the canvas cameras -- a plot is a view of its own, not an overlay on a
        camera image -- so this runs once per frame rather than once per canvas. Both eyes go
        under the same entity prefix so they share one plot, one line each.

        Args:
            frame_idx (int): The frame index.
        """
        for alg_idx, score in enumerate(self.score_per_alg):
            if score is None or frame_idx >= score.shape[2]:
                continue
            alg_name = self.algorithm_list[alg_idx]
            eye_labels = self.score_eye_labels_per_alg[alg_idx]

            for subject_idx, camera_idx in self.plotted_series_per_alg[alg_idx]:
                for eye_idx, eye_name in enumerate(eye_labels):
                    value = score[subject_idx, camera_idx, frame_idx, eye_idx]
                    # Gaps stay gaps: logging NaN would draw a line through untracked frames.
                    if np.isnan(value):
                        continue
                    entity_path = self.logger.generate_metric_entity_path(
                        alg_name=alg_name,
                        subject_name=self.subject_names[subject_idx],
                        cam_name=self.camera_names[camera_idx],
                        metric=eye_name,
                    )
                    self._log_score(entity_path, value)


class EyeClosedStateComponent(Component):
    """
    Class for reading eye closed/open state.

    Carries no overlay of its own — like GazeInteractionComponent, it exists to hand its
    per-eye binary state to EyeClosureComponent, which recolors the eye contours with it.

    Attributes:
        camera_names (List[str]): The camera names.
        subject_names (List[str]): The subject names.
    """

    STATE_KEY = "per_camera_state"

    def __init__(self, visualizer_config: Dict, io, logger, component_name: str):
        """
        Initialize the EyeClosedStateComponent.

        Args:
            visualizer_config (Dict): The visualizer configuration settings.
            io: The input/output object.
            logger (viewer.Viewer): The viewer rerun object.
            component_name (str): The name of the component.
        """
        super().__init__(visualizer_config, io, logger, component_name)
        descr = self.algorithms_results[0]["data_description"].item()
        self.camera_names = descr[self.STATE_KEY]["axis1"]
        self.subject_names = descr[self.STATE_KEY]["axis0"]

    def get_closed_state_data(self) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Get the closed-state data for every configured eye_closure_threshold instance, in
        list order.

        Each entry corresponds positionally to `[media.eye_closed_state].algorithms[i]` and is
        intended to be paired with `[media.eye_closure_score].algorithms[i]` for coloring.

        Returns:
            List of (data, eye_labels) tuples, one per algorithm. Data is
            (subjects, cameras, frames, eyes) with 1 = closed, 0 = open, NaN = missing.
        """
        labels_per_alg = self._get_algorithms_labels()
        return [(res[self.STATE_KEY], labels_per_alg[i]) for i, res in enumerate(self.algorithms_results)]

    def _get_algorithms_labels(self) -> List[List[str]]:
        """Eye labels (axis3) per algorithm."""
        return [res["data_description"].item()[self.STATE_KEY]["axis3"] for res in self.algorithms_results]

    def _log_data(self, entity_path: str, val: float) -> None:
        pass

    def visualize(self, frame_idx: int) -> None:
        pass


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
        self.camera_names = self.algorithms_results[0]["data_description"].item()[keyname]["axis1"]
        self.subject_names = self.algorithms_results[0]["data_description"].item()[keyname]["axis0"]

    def get_lookat_data(self) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Get the look at data for every configured gaze_distance instance, in list order.

        Each entry corresponds positionally to `[media.gaze_interaction].algorithms[i]` and
        is intended to be paired with `[media.gaze_fusion].algorithms[i]` for coloring.

        Returns:
            List of (data, labels) tuples, one per algorithm.
        """
        if "gaze_look_at_3d" in self.algorithms_results[0]["data_description"].item():
            data_name = "gaze_look_at_3d"
        else:
            data_name = "gaze_look_at_2d"
        labels_per_alg = self._get_algorithms_labels(data_name)
        return [(self.canvas_data[data_name][i], labels_per_alg[i]) for i in range(len(self.algorithm_list))]

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
            algorithm_labels.append(self.algorithms_results[i]["data_description"].item()[data_name]["axis3"])
        return algorithm_labels

    def _log_data(self):
        pass

    def visualize(self):
        pass


class FaceBoundingBoxComponent(Component):
    """
    Class for visualizing face bounding box data.

    Draws one box per subject per camera view, labelled with the detection confidence.

    The boxes also anchor other face-level overlays: EmotionIndividualComponent recolors
    them by the dominant emotion rather than drawing boxes of its own, so the two never
    disagree about where a face is.

    Attributes:
        camera_names (List[str]): The camera names.
        subject_names (List[str]): The subject names.
    """

    BBOX_KEY = "bbox_2d"

    def __init__(self, visualizer_config: Dict, io, logger, component_name: str):
        """
        Initialize the FaceBoundingBoxComponent.

        Args:
            visualizer_config (Dict): The visualizer configuration settings.
            io: The input/output object.
            logger (viewer.Viewer): The viewer rerun object.
            component_name (str): The name of the component.
        """
        super().__init__(visualizer_config, io, logger, component_name)
        descr = self.algorithms_results[0]["data_description"].item()
        self.camera_names = descr[self.BBOX_KEY]["axis1"]  # axis1 gives camera info
        self.subject_names = descr[self.BBOX_KEY]["axis0"]  # axis0 gives subject info

    def get_bbox_data(self) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Get the bounding boxes for every configured algorithm, in list order.

        Each entry corresponds positionally to `[media.face_bounding_box].algorithms[i]` and
        is intended to be paired with a face-level component that draws on top of the boxes.

        Returns:
            List of (data, camera_names) tuples, one per algorithm. Data is
            (subjects, cameras, frames, x0/y0/x1/y1/conf), NaN where no face was assigned.
        """
        return [(res[self.BBOX_KEY], self.camera_names) for res in self.algorithms_results]

    def _get_algorithms_labels(self) -> List[List[str]]:
        """Bounding box corner labels (axis3) per algorithm."""
        return [res["data_description"].item()[self.BBOX_KEY]["axis3"] for res in self.algorithms_results]

    def _log_data(self, entity_path: str, box: np.ndarray, color: List[int], label: str) -> None:
        """
        Log one face bounding box in rerun.

        Args:
            entity_path (str): The entity path.
            box (np.ndarray): The box as [x0, y0, x1, y1].
            color (List[int]): The color.
            label (str): The box label.
        """
        rr.log(
            entity_path,
            rr.Boxes2D(
                array=box,
                array_format=rr.Box2DFormat.XYXY,
                labels=label,
                colors=color,
            ),
        )

    def visualize(self, frame_idx: int) -> None:
        """
        Visualize the face bounding boxes on each camera view.

        Args:
            frame_idx (int): The frame index.
        """
        for canvas in self.canvas_list:
            if not canvas or canvas == "3D_Canvas":
                continue
            cam_name = canvas
            if cam_name not in self.camera_names:
                continue
            camera_index = self.camera_names.index(cam_name)

            for alg_idx, alg_data in enumerate(self.canvas_data[self.BBOX_KEY]):
                if frame_idx >= alg_data.shape[2]:  # number of frames
                    continue
                alg_name = self.algorithm_list[alg_idx]
                color = self._parse_alg_color(alg_idx)

                for subject_idx, subject in enumerate(self.subject_names):
                    subjs = self.visualizer_config["dataset_properties"]["video"]["cameras"][cam_name]["sees_subjects"]
                    if subject_idx not in subjs:
                        continue
                    box = alg_data[subject_idx, camera_index, frame_idx]
                    # NaN marks a frame where no face was assigned to this subject slot.
                    if np.isnan(box[:4]).any():
                        continue
                    entity_path = self.logger.generate_component_entity_path(
                        self.component_name,
                        is_3d=False,
                        alg_name=alg_name,
                        subject_name=subject,
                        cam_name=cam_name,
                    )
                    self._log_data(entity_path, box[:4], color, f"{subject} {box[4]:.2f}")


class EmotionIndividualComponent(Component):
    """
    Class for visualizing emotion individual data.

    Draws each subject's face bounding box coloured by their strongest emotion and labelled
    with it. The boxes come from face_bounding_box (paired in via bbox_tuples) rather than
    from this component's own NPZ, so the emotion overlay and the box overlay always agree
    on where the face is.

    The emotion labels are read from the NPZ's axis3, so whatever set the detector predicts
    is what gets drawn - the count is not assumed.

    The `valence_arousal` array is additionally plotted as a scalar timeseries, one plot per
    tracked (subject, camera) pair with a line per dimension. Pairs the detector never tracked
    are all-NaN and are dropped rather than plotted empty.

    Attributes:
        camera_names (List[str]): The camera names.
        subject_names (List[str]): The subject names.
        bbox_per_alg (List[Tuple[np.ndarray, List[str]]]): Per-algorithm (boxes, camera_names).
            Empty when no bounding box component is wired.
        valence_arousal_per_alg (List[np.ndarray | None]): Per-algorithm valence/arousal, or
            None when the canvas is empty or the detector does not emit it.
    """

    EMOTIONS_KEY = "emotions"
    VALENCE_AROUSAL_KEY = "valence_arousal"

    def __init__(
        self,
        visualizer_config: Dict,
        io,
        logger,
        component_name: str,
        bbox_tuples: List[Tuple[np.ndarray, List[str]]] = None,
    ):
        """
        Initialize the EmotionIndividualComponent.

        Args:
            visualizer_config (Dict): The visualizer configuration settings.
            io: The input/output object.
            logger (viewer.Viewer): The viewer rerun object.
            component_name (str): The name of the component.
            bbox_tuples (List[Tuple[np.ndarray, List[str]]], optional): Per-algorithm
                (bbox_2d, camera_names) from face_bounding_box, positionally aligned with
                this component's algorithms. Defaults to None (nothing is drawn).
        """
        super().__init__(visualizer_config, io, logger, component_name)
        descr = self.algorithms_results[0]["data_description"].item()
        self.camera_names = descr[self.EMOTIONS_KEY]["axis1"]  # axis1 gives camera info
        self.subject_names = descr[self.EMOTIONS_KEY]["axis0"]  # axis0 gives subject info
        self.algorithm_labels = self._get_algorithms_labels()

        # Per-algorithm boxes to draw on (aligned positionally with self.algorithm_list). The
        # emotion is a property of a face, so with no box there is nowhere to draw it.
        self.bbox_per_alg: List[Tuple[np.ndarray, List[str]]] = []
        if bbox_tuples:
            if len(bbox_tuples) != len(self.algorithm_list):
                raise ValueError(
                    f"emotion_individual has {len(self.algorithm_list)} algorithms but "
                    f"face_bounding_box provided {len(bbox_tuples)} entries. The two algorithms "
                    f"lists must be the same length (each emotion instance paired with the "
                    f"boxes of the detector that produced it)."
                )
            self.bbox_per_alg = bbox_tuples
        else:
            print(
                "WARNING! emotion_individual needs face_bounding_box to draw on, but it is not "
                "in the components list. Emotions will not be visualized.\n  "
                "Add 'face_bounding_box' to the components list in the visualizer_config.toml file"
            )

        # Per-algorithm valence/arousal, plotted as a timeseries alongside the boxes. Skipped
        # entirely when the canvas is empty, and per-algorithm when a detector emits only emotions.
        plot_valence_arousal = bool(
            self.visualizer_config["media"][self.component_name]["canvas"].get(self.VALENCE_AROUSAL_KEY)
        )
        self.valence_arousal_per_alg: List[np.ndarray | None] = [
            res[self.VALENCE_AROUSAL_KEY] if plot_valence_arousal and self.VALENCE_AROUSAL_KEY in res.files else None
            for res in self.algorithms_results
        ]
        self.valence_arousal_labels_per_alg: List[List[str]] = [
            res["data_description"].item()[self.VALENCE_AROUSAL_KEY]["axis3"]
            if self.valence_arousal_per_alg[i] is not None
            else []
            for i, res in enumerate(self.algorithms_results)
        ]
        # A (subject, camera) pair a detector never saw is all-NaN -- e.g. the left subject on
        # the right-facing camera. Plotting it would add a permanently empty view, so the
        # tracked pairs are resolved once here and the rest are skipped.
        self.plotted_series_per_alg = [
            self._resolve_plotted_series(alg_idx) for alg_idx in range(len(self.valence_arousal_per_alg))
        ]

    def _get_algorithms_labels(self) -> List[List[str]]:
        """Emotion labels (axis3) per algorithm."""
        return [res["data_description"].item()[self.EMOTIONS_KEY]["axis3"] for res in self.algorithms_results]

    def _resolve_plotted_series(self, alg_idx: int) -> List[Tuple[int, int]]:
        """
        Resolve which (subject_idx, camera_idx) valence/arousal series are worth plotting.

        A pair that is NaN for every frame means the detector never tracked that subject on
        that camera -- e.g. the left-seated subject on the right-facing camera -- so it is
        dropped rather than logged as a permanently empty plot.

        Returns:
            List of (subject_idx, camera_idx) pairs that carry at least one real value.
        """
        valence_arousal = self.valence_arousal_per_alg[alg_idx]
        if valence_arousal is None:
            return []

        # Any non-NaN across the frame and dimension axes means the pair was tracked at some point.
        tracked = ~np.isnan(valence_arousal).all(axis=(2, 3))  # (subjects, cameras)
        return [(int(s), int(c)) for s, c in zip(*np.nonzero(tracked))]

    def _log_score(self, entity_path: str, value: float) -> None:
        """
        Log one valence/arousal value as a scalar timeseries point in rerun.

        Args:
            entity_path (str): The entity path.
            value (float): The valence or arousal value.
        """
        rr.log(entity_path, rr.Scalar(round(float(value), 3)))

    def _log_data(self, entity_path: str, box: np.ndarray, color: List[int], label: str) -> None:
        """
        Log the face bounding box coloured by its dominant emotion.

        Args:
            entity_path (str): The entity path.
            box (np.ndarray): The box as [x0, y0, x1, y1].
            color (List[int]): The colour of the dominant emotion.
            label (str): The emotion name and its score.
        """
        rr.log(
            entity_path,
            rr.Boxes2D(
                array=box,
                array_format=rr.Box2DFormat.XYXY,
                labels=label,
                colors=color,
            ),
        )

    def visualize(self, frame_idx: int) -> None:
        """
        Visualize the emotion individual component on each camera view.

        Args:
            frame_idx (int): The frame index.
        """
        self._plot_valence_arousal(frame_idx)

        for canvas in self.canvas_list:
            if not canvas or canvas == "3D_Canvas":
                continue
            cam_name = canvas
            if cam_name not in self.camera_names:
                continue
            camera_index = self.camera_names.index(cam_name)

            for alg_idx, alg_data in enumerate(self.canvas_data[self.EMOTIONS_KEY]):
                if alg_idx >= len(self.bbox_per_alg) or frame_idx >= alg_data.shape[2]:
                    continue
                alg_name = self.algorithm_list[alg_idx]
                # One colour per emotion, so the config carries a colour list per algorithm.
                emotion_colors = self._parse_alg_color(alg_idx)
                emotion_labels = self.algorithm_labels[alg_idx]
                boxes, bbox_camera_names = self.bbox_per_alg[alg_idx]
                # The box detector has its own camera axis, which may be a different subset.
                if cam_name not in bbox_camera_names:
                    continue
                bbox_camera_index = bbox_camera_names.index(cam_name)

                for subject_idx, subject in enumerate(self.subject_names):
                    subjs = self.visualizer_config["dataset_properties"]["video"]["cameras"][cam_name]["sees_subjects"]
                    if subject_idx not in subjs:
                        continue

                    box = boxes[subject_idx, bbox_camera_index, frame_idx]
                    scores = alg_data[subject_idx, camera_index, frame_idx]
                    # NaN marks a frame where no face was assigned to this subject slot.
                    if np.isnan(box[:4]).any() or np.isnan(scores).all():
                        continue

                    strongest = int(np.nanargmax(scores))
                    entity_path = self.logger.generate_component_entity_path(
                        self.component_name,
                        is_3d=False,
                        alg_name=alg_name,
                        subject_name=subject,
                        cam_name=cam_name,
                    )
                    self._log_data(
                        entity_path,
                        box[:4],
                        color=emotion_colors[strongest % len(emotion_colors)],
                        label=f"{emotion_labels[strongest]} {scores[strongest]:.2f}",
                    )

    def _plot_valence_arousal(self, frame_idx: int) -> None:
        """
        Log the valence/arousal timeseries for every tracked (subject, camera) pair.

        Independent of the canvas cameras -- a plot is a view of its own, not an overlay on a
        camera image -- so this runs once per frame rather than once per canvas. Both dimensions
        go under the same entity prefix so they share one plot, one line each.

        Args:
            frame_idx (int): The frame index.
        """
        for alg_idx, valence_arousal in enumerate(self.valence_arousal_per_alg):
            if valence_arousal is None or frame_idx >= valence_arousal.shape[2]:
                continue
            alg_name = self.algorithm_list[alg_idx]
            labels = self.valence_arousal_labels_per_alg[alg_idx]

            for subject_idx, camera_idx in self.plotted_series_per_alg[alg_idx]:
                for label_idx, label in enumerate(labels):
                    value = valence_arousal[subject_idx, camera_idx, frame_idx, label_idx]
                    # Gaps stay gaps: logging NaN would draw a line through untracked frames.
                    if np.isnan(value):
                        continue
                    entity_path = self.logger.generate_metric_entity_path(
                        alg_name=alg_name,
                        subject_name=self.subject_names[subject_idx],
                        cam_name=self.camera_names[camera_idx],
                        metric=label,
                    )
                    self._log_score(entity_path, value)


class HeadOrientationComponent(Component):
    """
    Class for visualizing head orientation data.
    """

    # The heading and the pixel anchor it is drawn from are separate arrays, so the origin is
    # read directly instead of through the canvas config.
    DIRECTION_KEY = "head_direction_2d"
    ORIGIN_KEY = "head_origin_2d"
    # Arrow length (px), since the direction is a unitless heading rather than pixels.
    ARROW_LENGTH_PX = 60.0

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
        # head_direction_2d is per-camera, so it carries both the camera and subject axes
        self.camera_names = self.algorithms_results[0]["data_description"].item()[self.DIRECTION_KEY][
            "axis1"
        ]  # axis1 gives camera info
        self.subject_names = self.algorithms_results[0]["data_description"].item()[self.DIRECTION_KEY][
            "axis0"
        ]  # axis0 gives subject info
        self.algorithm_labels = self._get_algorithms_labels()
        self.origins_per_alg = [
            alg_result[self.ORIGIN_KEY][..., :2].astype(float) for alg_result in self.algorithms_results
        ]

    def _get_algorithms_labels(self) -> List[List[str]]:
        """
        Get the labels for the algorithms.

        Returns:
            List[List[str]]: The labels for the algorithms.
        """
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, _alg in enumerate(self.algorithm_list):
            algorithm_labels.append(self.algorithms_results[i]["data_description"].item()[self.DIRECTION_KEY]["axis3"])
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
            head_points (np.ndarray): The arrow origin in pixels (the nose tip).
            data_points (np.ndarray): The head direction, a unitless image-plane heading.
            color (List[int]): The color.
            dimension (str): The dimension.
        """
        # The direction is a heading, not a second point, so it is scaled to a fixed on-screen
        # length rather than subtracted from the origin.
        vectors_forward = data_points[0:2] * self.ARROW_LENGTH_PX
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
            if not canvas:
                continue
            cam_name = canvas
            camera_index = self.camera_names.index(cam_name)
            for alg_idx, alg_data in enumerate(self.canvas_data[self.DIRECTION_KEY]):
                num_frames = alg_data.shape[2]
                if frame_idx >= num_frames:  # number of frames
                    continue
                alg_name = self.algorithm_list[alg_idx]
                origins = self.origins_per_alg[alg_idx]
                for subject_idx, subject in enumerate(self.subject_names):
                    cam = self.visualizer_config["dataset_properties"]["video"]["cameras"][cam_name]
                    subjs = cam["sees_subjects"]
                    if subject_idx in subjs:
                        direction = alg_data[subject_idx, camera_index, frame_idx]
                        origin = origins[subject_idx, camera_index, frame_idx]
                        # NaN marks a frame where no face was assigned to this subject slot.
                        if np.isnan(origin).any() or np.isnan(direction[:2]).any():
                            continue
                        entity_path = self.logger.generate_component_entity_path(
                            self.component_name,
                            is_3d=False,
                            alg_name=alg_name,
                            subject_name=subject,
                            cam_name=cam_name,
                        )

                        color = self.visualizer_config["media"][self.component_name]["appearance"]["colors"][alg_idx]
                        self._log_data(entity_path, origin, direction, color, "2d")


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

        descr = self.algorithms_results[0]["data_description"].item()
        # Detect which dim this run carries; axes are the same across algorithms of same dim.
        self.data_key = next(k for k in ("body_distance_2d", "body_distance_3d") if k in descr)
        self.is_3d = self.data_key.endswith("_3d")
        axes = descr[self.data_key]
        self.camera_names = axes["axis1"]
        self.subject_names = axes["axis0"]

        # 3D midpoint between the two subjects' eyes (only needed if the 3D_Canvas will show it).
        self.eyes_middle_3d_data = eyes_middle_3d_data
        if self.eyes_middle_3d_data is not None:
            first_subject_eyes_middle_data = self.eyes_middle_3d_data[0, 0, :].mean(axis=0)
            second_subject_eyes_middle_data = self.eyes_middle_3d_data[1, 0, :].mean(axis=0)
            self.middle_point_3d = (first_subject_eyes_middle_data + second_subject_eyes_middle_data) / 2

        # 2D midpoint per camera view.
        self.eyes_middle_2d_data = eyes_middle_2d_data
        self.camera_view_middle_point_dict = {}
        if self.eyes_middle_2d_data is not None:
            for cam in self.camera_names:
                camera_idx = self.camera_names.index(cam)
                first_subject_eyes_middle_data = self.eyes_middle_2d_data[0, camera_idx, :].mean(axis=0)
                second_subject_eyes_middle_data = self.eyes_middle_2d_data[1, camera_idx, :].mean(axis=0)
                middle_point = (first_subject_eyes_middle_data + second_subject_eyes_middle_data) / 2
                self.camera_view_middle_point_dict[cam] = middle_point

    def _get_algorithms_labels(self) -> List[List[str]]:
        """Distance labels (axis3) per algorithm."""
        return [res["data_description"].item()[self.data_key]["axis3"] for res in self.algorithms_results]

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
            proximity_start = np.array([mid_point[0] - (data_points / 2), mid_point[1] - 100])
            proximity_end = np.array([mid_point[0] + (data_points / 2), mid_point[1] - 100])
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
            proximity_start = np.array([mid_point[0] - (data_points / 2), mid_point[1] - 0.5, mid_point[2]])
            proximity_end = np.array([mid_point[0] + (data_points / 2), mid_point[1] - 0.5, mid_point[2]])
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
        Visualize the proximity component. 3D goes on the single ['3d'] slot; 2D uses per-camera slots.
        """
        for canvas in self.canvas_list:
            if canvas == "3D_Canvas":
                if not self.is_3d:
                    continue
                for alg_idx, alg_data in enumerate(self.canvas_data[self.data_key]):
                    alg_name = self.algorithm_list[alg_idx]
                    if frame_idx >= alg_data.shape[2]:
                        continue
                    frame_proximity = alg_data[:, 0, frame_idx, 0][0]
                    entity_path = self.logger.generate_component_entity_path(
                        self.component_name, is_3d=True, alg_name=alg_name
                    )
                    self._log_data(entity_path, frame_proximity, alg_idx, self.middle_point_3d, "3d")
            else:
                if self.is_3d:
                    continue
                cam_name = canvas
                for alg_idx, alg_data in enumerate(self.canvas_data[self.data_key]):
                    alg_name = self.algorithm_list[alg_idx]
                    camera_idx = self.camera_names.index(canvas)
                    if frame_idx >= alg_data.shape[2]:
                        continue
                    frame_proximity = alg_data[:, camera_idx, frame_idx, 0][0]
                    entity_path = self.logger.generate_component_entity_path(
                        self.component_name,
                        is_3d=False,
                        alg_name=alg_name,
                        cam_name=cam_name,
                    )
                    mid_point = self.camera_view_middle_point_dict[cam_name]
                    self._log_data(entity_path, frame_proximity, alg_idx, mid_point, "2d")


class KinematicsComponent(Component):
    """
    Class for visualizing kinematics data.
    """

    def __init__(self, visualizer_config: Dict, io, logger, component_name: str):
        """
        Initialize the KinematicsComponent.

        Reads the pre-aggregated `velocity_bodypart_{dim}` output produced by
        velocity_body (one detector instance per dim). Axes3 = bodypart labels,
        axis4 = [velocity, confidence].
        """
        super().__init__(visualizer_config, io, logger, component_name)

        descr = self.algorithms_results[0]["data_description"].item()
        # Detect which dim this run carries; axes are the same across algorithms of the same dim.
        self.data_key = next(k for k in ("velocity_bodypart_2d", "velocity_bodypart_3d") if k in descr)
        self.is_3d = self.data_key.endswith("_3d")
        axes = descr[self.data_key]
        self.camera_names = axes["axis1"]
        self.subject_names = axes["axis0"]
        self.bodypart_labels = axes["axis3"]

    def _get_algorithms_labels(self) -> List[List[str]]:
        """Bodypart labels (axis3) per algorithm."""
        return [res["data_description"].item()[self.data_key]["axis3"] for res in self.algorithms_results]

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
        Visualize the kinematics component from the pre-aggregated per-bodypart velocity.

        3D data lives on the single ["3d"] pseudo-camera slot (one metric per subject/bodypart);
        2D data has one metric per (subject, camera, bodypart).
        """
        for alg_idx, alg_data in enumerate(self.canvas_data[self.data_key]):
            if frame_idx >= alg_data.shape[2]:
                continue
            alg_name = self.algorithm_list[alg_idx]
            for subject_idx, subject in enumerate(self.subject_names):
                for bp_idx, bodypart in enumerate(self.bodypart_labels):
                    if self.is_3d:
                        # axis1 length 1 for 3D; axis4[0] is velocity, axis4[1] is confidence.
                        velocity = alg_data[subject_idx, 0, frame_idx, bp_idx, 0]
                        entity_path = self.logger.generate_component_entity_path(
                            self.component_name,
                            is_3d=True,
                            alg_name=alg_name,
                            subject_name=subject,
                            bodypart=bodypart,
                        )
                        self._log_data(entity_path, velocity)
                    else:
                        for camera_idx, camera in enumerate(self.camera_names):
                            velocity = alg_data[subject_idx, camera_idx, frame_idx, bp_idx, 0]
                            entity_path = self.logger.generate_component_entity_path(
                                self.component_name,
                                is_3d=False,
                                alg_name=alg_name,
                                subject_name=subject,
                                cam_name=camera,
                                bodypart=bodypart,
                            )
                            self._log_data(entity_path, velocity)


class BodyMeshComponent(Component):
    """
    Class for visualizing body mesh data.
    """

    def __init__(self, visualizer_config: Dict, io, logger, component_name: str, calib: Dict):
        super().__init__(visualizer_config, io, logger, component_name)

        self.calib = calib
        self.subject_names = self.algorithms_results[0]["data_description"].item()["vertices"]["axis0"]
        self.camera_names = self.algorithms_results[0]["data_description"].item()["vertices"]["axis1"]

        self.faces = self.algorithms_results[0]["faces"]  # (F, 3) int32
        self._missing_world_warned = set()

    def _get_algorithms_labels(self) -> List[List[str]]:
        """
        Get the labels for the algorithms.

        Returns:
            List[List[str]]: The labels for the algorithms.
        """
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, _alg in enumerate(self.algorithm_list):
            algorithm_labels.append(self.algorithms_results[i]["data_description"].item()["vertices"]["axis3"])
        return algorithm_labels

    def _log_data(self, entity_path: str, data_points: np.ndarray, alg_idx: int):
        if np.all(np.isnan(data_points)):
            return
        color = self._parse_alg_color(alg_idx)
        rgba = [color[0], color[1], color[2], self._parse_transparency()]
        vertex_colors = np.tile(rgba, (data_points.shape[0], 1))
        normals = self._compute_vertex_normals(data_points)
        rr.log(
            entity_path,
            rr.Mesh3D(
                vertex_positions=data_points,
                vertex_normals=normals,
                triangle_indices=self.faces,
                vertex_colors=vertex_colors,
            ),
        )

    def visualize(self, frame_idx: int) -> None:
        for canvas in self.canvas_list:
            if not canvas:
                continue

            cam_name = canvas
            camera_index = self.camera_names.index(cam_name)

            for alg_idx, alg_name in enumerate(self.algorithm_list):
                if "vertices_world" not in self.algorithms_results[alg_idx].files:
                    if alg_name not in self._missing_world_warned:
                        self._missing_world_warned.add(alg_name)
                        print(
                            f"[BodyMeshComponent] '{alg_name}' results do not contain 'vertices_world', "
                            f"skipping 3D mesh visualization for this algorithm."
                        )
                    continue

                alg_data_world = self.algorithms_results[alg_idx]["vertices_world"]

                if frame_idx >= alg_data_world.shape[2]:
                    continue

                for subject_idx, subject in enumerate(self.subject_names):
                    vertices_world = alg_data_world[subject_idx, camera_index, frame_idx]  # (N_v, 3)
                    # get mesh head centroid — top ~5% of vertices by Z (or Y in rerun coords)
                    vertices_finite = vertices_world[np.isfinite(vertices_world).all(axis=1)]

                    # TODO: subjects vertices of sam3d located at 0,0,0.
                    #  To separate subjects location during visualization we are using here
                    #  a kind of fix unit based on mesh's size, so their distance does not
                    # represent the real locations in 3D world.
                    # Later we will fix it while post-processing body_mesh results
                    mesh_height = vertices_finite[:, 0].max() - vertices_finite[:, 0].min()
                    offset = np.array([subject_idx * mesh_height * 2, 0.0, 0.0])
                    # shift mesh based on offset
                    vertices_corrected = vertices_world + offset

                    vertices_rr = self._to_rerun_coords(vertices_corrected)
                    entity_path_3d = self.logger.generate_component_entity_path(
                        self.component_name,
                        is_3d=True,
                        alg_name=alg_name,
                        subject_name=subject,
                        cam_name=cam_name,
                    )
                    self._log_data(entity_path_3d, vertices_rr, alg_idx)

    def _to_rerun_coords(self, vertices: np.ndarray) -> np.ndarray:
        """Convert from Z-up (world) to Rerun's Y-up convention.

        World:  X=right, Y=forward, Z=up
        Rerun:  X=right, Y=up,      Z=backward
        """
        v = vertices.copy()
        return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)

    def _compute_vertex_normals(self, vertices: np.ndarray) -> np.ndarray:
        """Compute per-vertex normals for proper lighting/shading."""
        v0 = vertices[self.faces[:, 0]]
        v1 = vertices[self.faces[:, 1]]
        v2 = vertices[self.faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.where(norms == 0, 1, norms)

        vertex_normals = np.zeros_like(vertices)
        for j in range(3):
            np.add.at(vertex_normals, self.faces[:, j], face_normals)

        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = vertex_normals / np.where(norms == 0, 1, norms)
        return vertex_normals

    def _parse_transparency(self) -> int:
        """
        Parse the alpha value from the visulizer config.

        Returns:
            int: The alpha value.
        """
        return self.visualizer_config["media"][self.component_name]["appearance"]["alpha"]
