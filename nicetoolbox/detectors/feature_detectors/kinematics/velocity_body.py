"""
Velocity Body feature detector class for kinematics of the body.
"""

import logging
import warnings

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF_PER_LABEL, ArraySchema

from ....utils import check_and_exception as check
from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ..base_feature import BaseFeature
from . import utils as kinematics_utils

# Velocity output axis4: scalar motion magnitude per frame plus the propagated confidence.
VELOCITY_CONFIDENCE = ArraySchema(data_columns=("velocity", "confidence_score"))


class VelocityBody(BaseFeature):
    """
    Abstract base for the body-velocity kinematics feature detector.

    Computes per-keypoint displacement and velocity between adjacent frames from a
    body_joints pose input. Concrete subclasses define the working dimension (2D or 3D)
    and declare their static inputs/outputs accordingly.
    """

    components = ["kinematics"]  # TODO: delete me

    # working dimension ("2d"/"3d"), set by subclasses
    dim: str

    def _initialize_detector(self) -> None:
        # Upstream pose config carries the keypoint mapping and cameras.
        upstream_config = self.loaded_inputs[f"pose_{self.dim}"].upstream_config
        keypoints_mapping_name = upstream_config.keypoint_mapping  # e.g. coco_wholebody

        # Predictions mapping from runtime_config (already loaded and validated)
        self.keypoints_mapping = getattr(self.predictions_mapping.human_pose, keypoints_mapping_name)
        self.bodyparts_list = list(self.keypoints_mapping.bodypart_index.model_dump().keys())

        self.fps = self.data.fps

    def compute(self) -> DetectorOutput:
        """
        Computes the kinematics component (displacement + velocity per keypoint).
        """
        pose = self.loaded_inputs[f"pose_{self.dim}"]
        pose, pose_axes = pose.data, pose.axes

        # separate pos vector (2d/3d) and confidence (last dim)
        keypoints = pose[..., :-1]
        conf_score = pose[..., -1:]

        # differences[t] = keypoints[t] - keypoints[t-1]; first frame keeps NaN.
        differences = np.full_like(keypoints, np.nan)
        differences[:, :, 1:] = keypoints[:, :, 1:] - keypoints[:, :, :-1]

        # Confidence propagated as the minimum of the two consecutive frames.
        min_confidence = np.full_like(conf_score, np.nan)
        min_confidence[:, :, 1:] = np.minimum(conf_score[:, :, 1:], conf_score[:, :, :-1])

        # Euclidean distance for each keypoint between adjacent frames, scaled to per-second.
        motion_magnitude = np.linalg.norm(differences, axis=-1, keepdims=True)
        motion_velocity = motion_magnitude * self.fps

        # Displacement keeps the coordinate columns + confidence; velocity is a scalar + confidence.
        displacement_axes = pose_axes  # x, y, z, conf
        displacement = np.concatenate([differences, min_confidence], axis=-1)

        # Velocity is one scalar + confidence score
        velocity_axes = pose_axes.replace(data=["velocity", "confidence_score"])
        velocity = np.concatenate([motion_velocity, min_confidence], axis=-1)

        # Mean velocity per body part (head, upper_body, ...), for downstream consumers
        # and visualization. Bodypart labels come from the upstream keypoint mapping.
        bodypart_axes = velocity_axes.replace(labels=list(self.bodyparts_list))
        bodypart_motion = self._calculate_bodypart_motion(velocity)

        out = DetectorOutput()
        out.add("kinematics", f"displacement_vector_body_{self.dim}", data=displacement, axes=displacement_axes)
        out.add("kinematics", f"velocity_body_{self.dim}", data=velocity, axes=velocity_axes)
        out.add("kinematics", f"velocity_bodypart_{self.dim}", data=bodypart_motion, axes=bodypart_axes)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out

    def visualization(self, out: DetectorOutput):
        """
        Creates visualizations for the computed kinematics component.

        Sums movement per body part and plots the mean motion magnitude by body part
        across frames, per camera.
        """
        logging.info(f"Visualizing the feature detector output {self.components}.")

        bodypart_motion = out.get("kinematics", f"velocity_bodypart_{self.dim}")

        # Camera names travel with the array (axis1); the upstream config may list fewer.
        # Pass only the velocity column (axis4[0]); the util expects (S, C, F, n_bodyparts).
        kinematics_utils.visualize_mean_of_motion_magnitude_by_bodypart(
            bodypart_motion.data[..., 0],
            self.bodyparts_list,
            self.viz_folder,
            self.subjects_descr,
            bodypart_motion.axes.cameras,
        )

        logging.info(f"Visualization of feature detector {self.components} completed.")

    def _calculate_bodypart_motion(self, motion_velocity):
        """
        Calculates the mean movement per body part with an aggregated confidence.

        Averages the velocity across the joints of each body part; confidence is aggregated
        as the min over the same joints (matching the min-propagation used across frames),
        so a low-quality joint drags its bodypart's confidence down.

        Parameters:
            motion_velocity (numpy.ndarray): A 5D numpy array with shape
                (#persons, #cameras, #frames, #joints/keypoints, [velocity, confidence])
                representing the motion velocity.

        Returns:
            bodypart_motion (numpy.ndarray): A 5D numpy array with shape
                (#persons, #cameras, #frames, #bodyparts, 2) where axis4 = [velocity, confidence].
        """
        bodypart_motion = []
        bodypart_index = self.keypoints_mapping.bodypart_index

        # Split axis4 into velocity (col 0) and confidence (col 1); aggregate each separately
        # then re-concat on the last axis. Keep the joint-column axis for concatenate to stack.
        velocity = motion_velocity[..., 0:1]
        confidence = motion_velocity[..., 1:2]

        # All-NaN joint slices are expected (frame 0 has no previous frame to diff, and a
        # subject not seen by a camera is NaN across every frame); nanmean/nanmin correctly
        # return NaN for them, so silence the cosmetic "Mean of empty slice" RuntimeWarning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for bodypart_name in self.bodyparts_list:
                joint_indices = getattr(bodypart_index, bodypart_name)
                mean_vel = np.nanmean(velocity[:, :, :, joint_indices, :], axis=-2)  # (S, C, F, 1)
                min_conf = np.nanmin(confidence[:, :, :, joint_indices, :], axis=-2)  # (S, C, F, 1)
                # Stack velocity + confidence on a new last axis, then a bodypart axis in front.
                bodypart_motion.append(np.concatenate([mean_vel, min_conf], axis=-1)[..., np.newaxis, :])

        # Concatenate on the bodypart axis -> (S, C, F, n_bodyparts, 2)
        bodypart_motion = np.concatenate(bodypart_motion, axis=-2)

        # check for any [0,0,0] prediction
        check.check_zeros(bodypart_motion[:, :, 1:])

        return bodypart_motion


class VelocityBody2D(VelocityBody):
    """Body velocity computed on 2D pose keypoints (displacement in pixels)."""

    algorithm_type = "velocity_body_2d"
    dim = "2d"

    inputs = [NpzDetectorInput("body_joints", "pose_2d", schema=VECTOR_2D_CONF_PER_LABEL)]
    outputs = [
        NpzDetectorOutput("kinematics", "displacement_vector_body_2d", schema=VECTOR_2D_CONF_PER_LABEL),
        NpzDetectorOutput("kinematics", "velocity_body_2d", schema=VELOCITY_CONFIDENCE),
        NpzDetectorOutput("kinematics", "velocity_bodypart_2d", schema=VELOCITY_CONFIDENCE),
    ]


class VelocityBody3D(VelocityBody):
    """Body velocity computed on 3D pose keypoints (displacement in real-world units)."""

    algorithm_type = "velocity_body_3d"
    dim = "3d"

    inputs = [NpzDetectorInput("body_joints", "pose_3d", schema=VECTOR_3D_CONF_PER_LABEL)]
    outputs = [
        NpzDetectorOutput("kinematics", "displacement_vector_body_3d", schema=VECTOR_3D_CONF_PER_LABEL),
        NpzDetectorOutput("kinematics", "velocity_body_3d", schema=VELOCITY_CONFIDENCE),
        NpzDetectorOutput("kinematics", "velocity_bodypart_3d", schema=VELOCITY_CONFIDENCE),
    ]
