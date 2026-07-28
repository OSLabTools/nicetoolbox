import logging
import warnings
from typing import List

import numpy as np

from nicetoolbox_core.data.array_schema import (
    BOOLEAN_NAN,
    FLOAT,
    VECTOR_2D,
    VECTOR_2D_CONF,
    VECTOR_2D_CONF_PER_LABEL,
    VECTOR_3D,
    VECTOR_3D_CONF,
    VECTOR_3D_CONF_PER_LABEL,
)

from ....utils import linear_algebra as alg
from ....utils import visual_utils as vis_ut
from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ..base_feature import BaseFeature
from ..gaze_interaction import utils as gaze_interaction_utils


class BaseGazeDistance(BaseFeature):
    """
    Computes the gaze_interaction component for a 2-person context.

    For each subject, finds the smallest distance between their gaze direction vector and
    the other subject's head (usually the nose keypoint), then derives whether the gaze is directed
    at the head (look_at) and whether it is mutual. Subclasses defines the working dimension
    (2D/3D) and declare their static inputs (body pose + fused gaze) and outputs.
    """

    components = ["gaze_interaction"]  # TODO: delete me

    # fields
    dim: str  # working dimension ("2d"/"3d"), set by subclasses; drives the output keys.
    used_keypoints: List[str]
    keypoint_index: List[int]
    _viz_gaze: np.ndarray

    def _initialize_detector(self) -> None:
        """Resolve the head-anchor keypoint indices from the upstream pose's keypoint mapping.

        The head point is the mean of the configured `used_keypoints` body joints (e.g.
        ["nose"], or ["left_eye", "right_eye"] for the eyes midpoint). They are indexed
        directly into the pose array's keypoints axis (no wholebody/face offset).
        """
        upstream_config = self.loaded_inputs[f"pose_{self.dim}"].upstream_config
        keypoint_mapping = getattr(self.predictions_mapping.human_pose, upstream_config.keypoint_mapping)
        keypoints_index = keypoint_mapping.keypoints_index.body

        self.used_keypoints = self.detector_config.used_keypoints
        for keypoint in self.used_keypoints:
            if keypoint not in keypoints_index:
                logging.error(f"Given used_keypoint could not be found in predictions_mapping: {keypoint}")
        self.keypoint_index = [keypoints_index[keypoint] for keypoint in self.used_keypoints]

    def compute(self) -> DetectorOutput:
        """
        Compute the gaze_interaction component.

        Returns a DetectorOutput carrying distance_gaze_<dim>, gaze_look_at_<dim> and
        gaze_mutual_<dim> under the gaze_interaction component; validation and saving are
        handled by BaseFeature.run().
        """
        pose = self.loaded_inputs[f"pose_{self.dim}"]
        gaze = self.loaded_inputs[f"gaze_{self.dim}"]
        pose_data, pose_axes = pose.data, pose.axes
        # Gaze arrays carry a trailing confidence column on the labels axis; drop it for the
        # line-point distance math (which expects pure coordinates on the last axis).
        gaze_data = gaze.data[..., :-1]
        gaze_shape = gaze.axes.replace(labels=gaze.axes.labels[:-1])

        # TODO: redundant?
        # Gaze and pose must describe the same subjects.
        if gaze_shape.subjects != pose_axes.subjects:
            raise ValueError(f"Gaze and pose subjects differ: {gaze_shape.subjects} vs {pose_axes.subjects}.")
        subjects = gaze_shape.subjects
        subjects_count = len(subjects)

        # Gaze interaction is pairwise: it needs at least two subjects to look at each other.
        if subjects_count < 2:
            raise ValueError(f"Gaze interaction requires at least 2 subjects, got {subjects_count}: {subjects}")

        # TODO: move to some common system for tensor alligment
        # pose and gaze can have different cameras set, so we recut them to match
        pose_data, gaze_data, shared_cameras = self._align_cameras(pose_data, pose_axes, gaze_data, gaze_shape)
        gaze_shape = gaze_shape.replace(cameras=shared_cameras)
        self._viz_gaze = gaze_data  # TODO: smell, need better way to pass aligned gaze

        # TODO: we should allow user to define different keypoints, i.e. is the subject looks onto other subkject leg?
        # Keypoint 2d/3d (usually head): mean over the configured keypoints (drop the confidence column)
        keypoint_cords = pose_axes.data[:-1]  # xyz/xy
        keypoint_axes = pose_axes.replace(labels=keypoint_cords, data=[])  # (S, C, F, cords)
        keypoint = pose_data[:, :, :, self.keypoint_index, :-1].mean(axis=-2)

        # prepare pairwise distance output (S, C, F, S)
        # diagonal same subject-subject pairs are marked by nan
        axes = gaze_shape.replace(labels=list(subjects), data=[])
        distances = np.full(axes.shape, np.nan, dtype=float)
        for i in range(subjects_count):
            for j in range(subjects_count):
                if i == j:  # skip diagonal pairs
                    continue
                d = alg.distance_line_point(keypoint[i], gaze_data[i], keypoint[j])
                distances[i, ..., j] = d.squeeze(-1)

        # save all the nans (missing face/gaze detections or diag pairs)
        missing = np.isnan(distances)

        # look_at[i, ..., j]: subject i's gaze is close enough to subject j's target keypoint.
        look_at_bool = distances <= self.detector_config.threshold_look_at
        # booleans are encoded as floats to support nan for missing data
        look_at = look_at_bool.astype(float)
        look_at[missing] = np.nan

        # mutual[i, ..., j]: i and j look at each other (symmetric). Computed as booleans first.
        mutual_bool = look_at_bool & np.swapaxes(look_at_bool, 0, -1)
        mutual = mutual_bool.astype(float)
        mutual[missing] = np.nan

        out = DetectorOutput()
        out.add("gaze_interaction", f"distance_gaze_{self.dim}", data=distances, axes=axes)
        out.add("gaze_interaction", f"gaze_look_at_{self.dim}", data=look_at, axes=axes)
        out.add("gaze_interaction", f"gaze_mutual_{self.dim}", data=mutual, axes=axes)
        out.add("gaze_interaction", f"head_position_{self.dim}", data=keypoint, axes=keypoint_axes)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out

    def visualization(self, out: DetectorOutput) -> None:
        """
        Visualize the gaze interaction: distance line graph and a per-camera frame video.

        The line graph passes the saved subject x subject matrices to the plotting util, which
        collapses each to a per-subject value. The frame video draws each subject's gaze arrow on
        the source frames (green when looking at someone, red otherwise) plus the look-at target
        circle; 2D arrays are already image-plane, 3D ones are reprojected per camera.
        """
        logging.info(
            f"Visualizing the feature detector output {self.components}. "
            "This may take longer due to the evolving linegraph video creation."
        )

        # draw a plot graph
        distance_array = out.get("gaze_interaction", f"distance_gaze_{self.dim}")
        look_at = out.get("gaze_interaction", f"gaze_look_at_{self.dim}").data
        mutual = out.get("gaze_interaction", f"gaze_mutual_{self.dim}").data
        gaze_interaction_utils.visualize_gaze_interaction(
            distance_array.data, look_at, mutual, self.viz_folder, self.subjects_descr, distance_array.axes.cameras
        )

        head_array = out.get("gaze_interaction", f"head_position_{self.dim}")
        head = head_array.data  # (S, C, F, dim)
        gaze = self._viz_gaze  # camera-aligned gaze stashed by compute(), (S, C, F, dim)
        cameras = head_array.axes.cameras

        # "Looks at anyone" per subject: 1 if look_at is true toward any other subject this frame.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            looks_at_anyone = np.nanmax(look_at, axis=-1)  # (S, C, F), nan when all-nan

        threshold = self.detector_config.threshold_look_at
        if self.dim == "2d":
            # 2D threshold is already pixels: a constant target-circle radius per head.
            radii = np.full(looks_at_anyone.shape, threshold, dtype=float)
        else:
            # 3D arrows: reuse gaze_fusion's per-camera reprojection instead of redoing the
            # world->pixel math here. The only thing left to compute is the pixel radius of the
            # metric threshold sphere (the sphere shrinks with depth, so it varies per view).
            gaze_2d_input = self.loaded_inputs["gaze_2d"]
            cameras = gaze_2d_input.axes.cameras
            gaze = gaze_2d_input.data[..., :2]  # (S, C, F, 2)
            head = self.loaded_inputs["gaze_origin_2d"].data[..., :2]  # (S, C, F, 2)
            radii = self._project_threshold_radii(head_array.data, cameras, threshold)
            looks_at_anyone = np.repeat(looks_at_anyone, len(cameras), axis=1)  # ["3d"] -> real cams

        gaze_interaction_utils.visualize_gaze_arrows(
            heads=head,
            gaze=gaze,
            looks_at_anyone=looks_at_anyone,
            radii=radii,
            dataloader_config=self.data.get_input_recipes(),
            camera_names=cameras,
            subjects_descr=self.subjects_descr,
            output_folder=self.viz_folder,
            fps=self.data.fps,
        )
        logging.info(f"Visualization of feature detector {self.components} completed.")

    @staticmethod
    def _select_cameras(data: np.ndarray, current: List[str], wanted: List[str]) -> np.ndarray:
        """Slice `data`'s camera axis (axis1) down to `wanted`, in `wanted`'s order."""
        idx = [current.index(cam) for cam in wanted]
        return np.take(data, idx, axis=1)

    def _align_cameras(self, pose_data, pose_axes, gaze_data, gaze_axes):
        """Restrict pose and gaze to the cameras present in BOTH, in canonical (sorted) order.

        A detector may legitimately cover a subset of cameras (e.g. pose on 2, gaze on 4); we
        compute on the overlap and drop the rest. Returns (pose_data, gaze_data, shared_cameras).
        """
        shared_cameras = [cam for cam in sorted(pose_axes.cameras) if cam in set(gaze_axes.cameras)]
        if not shared_cameras:
            raise ValueError(f"Gaze and pose share no cameras: pose {pose_axes.cameras}, gaze {gaze_axes.cameras}.")
        pose_data = self._select_cameras(pose_data, pose_axes.cameras, shared_cameras)
        gaze_data = self._select_cameras(gaze_data, gaze_axes.cameras, shared_cameras)
        return pose_data, gaze_data, shared_cameras

    def _project_threshold_radii(self, head_world: np.ndarray, target_cameras: List[str], threshold: float):
        """Per-view, per-frame pixel radius of the metric look-at sphere around each head.

        The head positions and per-camera gaze arrows are already provided by gaze_fusion; the
        only viz quantity gaze_distance must still compute is how big the metric threshold sphere
        appears in each camera. We offset the head along the camera's world x-axis by `threshold`
        (which stays perpendicular to the view ray) and project both points; their pixel distance
        is the silhouette radius, which shrinks with depth. NaN heads stay NaN.

        Args:
            head_world (ndarray): world head points on the ["3d"] slot, (subjects, 1, frames, 3).
            target_cameras (list): cameras to project into (must be calibration keys).
            threshold (float): look-at sphere radius in world units.

        Returns:
            radii (ndarray): (subjects, len(target_cameras), frames) pixel radii, NaN where head is NaN.
        """
        calibration = self.data.calibration
        n_sub, _, n_frames, _ = head_world.shape
        n_cam = len(target_cameras)
        radii = np.full((n_sub, n_cam, n_frames), np.nan)

        for cam_idx, cam_name in enumerate(target_cameras):
            if not calibration or cam_name not in calibration:
                logging.warning(f"Calibration missing for camera '{cam_name}'; skipping its threshold radii.")
                continue
            calib = calibration[cam_name]
            projection_matrix = np.asarray(calib["projection_matrix"], dtype=float)  # 3x4 world->pixel
            _, _, cam_R, _ = vis_ut.get_cam_para_studio(calibration, cam_name)
            cam_x_axis = np.asarray(cam_R, dtype=float)[0]

            for sub in range(n_sub):
                head = head_world[sub, 0]  # (F, 3), single ["3d"] slot
                valid = ~np.isnan(head).any(axis=1)
                if not np.any(valid):
                    continue
                head_valid = head[valid]
                offset = head_valid + threshold * cam_x_axis
                ones = np.ones((valid.sum(), 1))
                head_px = self._project_points(projection_matrix, np.concatenate([head_valid, ones], axis=1))
                offset_px = self._project_points(projection_matrix, np.concatenate([offset, ones], axis=1))
                radii[sub, cam_idx, valid] = np.linalg.norm(offset_px - head_px, axis=1)

        return radii

    @staticmethod
    def _project_points(projection_matrix: np.ndarray, points_homogeneous: np.ndarray) -> np.ndarray:
        """Project homogeneous world points (n, 4) to pixels (n, 2) via a 3x4 projection matrix."""
        projected = (projection_matrix @ points_homogeneous.T).T  # (n, 3)
        return projected[:, :2] / projected[:, 2:3]


# TODO: 2d gaze calculation is very naive, should we deprecate it?
class GazeDistance2D(BaseGazeDistance):
    """Gaze interaction computed on 2D body pose (nose) and filtered 2D gaze."""

    algorithm_type = "gaze_distance_2d"
    dim = "2d"

    inputs = [
        NpzDetectorInput("body_joints", "pose_2d", schema=VECTOR_2D_CONF_PER_LABEL),
        NpzDetectorInput("gaze_multiview", "gaze_2d", schema=VECTOR_2D_CONF),
    ]
    outputs = [
        NpzDetectorOutput("gaze_interaction", "distance_gaze_2d", schema=FLOAT),
        NpzDetectorOutput("gaze_interaction", "gaze_look_at_2d", schema=BOOLEAN_NAN),
        NpzDetectorOutput("gaze_interaction", "gaze_mutual_2d", schema=BOOLEAN_NAN),
        NpzDetectorOutput("gaze_interaction", "head_position_2d", schema=VECTOR_2D),
    ]


class GazeDistance3D(BaseGazeDistance):
    """Gaze interaction computed on 3D body pose (nose) and fused 3D gaze."""

    algorithm_type = "gaze_distance_3d"
    dim = "3d"

    inputs = [
        NpzDetectorInput("body_joints", "pose_3d", schema=VECTOR_3D_CONF_PER_LABEL),
        NpzDetectorInput("gaze_multiview", "gaze_3d", schema=VECTOR_3D_CONF),
        NpzDetectorInput("gaze_multiview", "gaze_2d", schema=VECTOR_2D_CONF),
        NpzDetectorInput("gaze_multiview", "gaze_origin_2d", schema=VECTOR_2D_CONF),
    ]
    outputs = [
        NpzDetectorOutput("gaze_interaction", "distance_gaze_3d", schema=FLOAT),
        NpzDetectorOutput("gaze_interaction", "gaze_look_at_3d", schema=BOOLEAN_NAN),
        NpzDetectorOutput("gaze_interaction", "gaze_mutual_3d", schema=BOOLEAN_NAN),
        NpzDetectorOutput("gaze_interaction", "head_position_3d", schema=VECTOR_3D),
    ]
