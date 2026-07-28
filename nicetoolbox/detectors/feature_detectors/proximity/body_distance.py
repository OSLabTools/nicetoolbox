"""
Body Distance feature detector class for the proximity component.
"""

import logging
from typing import List

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF_PER_LABEL, ArraySchema

from ....configs.schemas.detectors_instances_configs import BodyDistanceConfig
from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ..base_feature import BaseFeature
from . import utils as pro_utils

# Proximity output: axis3 = ["distance", "confidence_score"] (scalar + aggregated confidence).
# Confidence is the min across the keypoints averaged into the distance and across subjects.
DISTANCE = ArraySchema(labels_columns=("distance", "confidence_score"))


class BodyDistance(BaseFeature):
    """
    Abstract base for the body-distance proximity feature detector.

    Computes the Euclidean distance between selected keypoints of two individuals
    in the scene, per frame. Concrete subclasses fix the working dimension (2D or
    3D) and declare their static inputs/outputs/components accordingly.
    """

    # internal fields
    components = ["proximity"]  # TODO: delete me
    detector_config: BodyDistanceConfig
    used_keypoints: List[str]
    keypoint_index: List[int]

    # working dimension ("2d"/"3d"), set by subclasses; drives the input handle name
    # and the output npz key (body_distance_{dim}).
    dim: str

    def _initialize_detector(self) -> None:
        """Setup the BodyDistance feature detector.

        Extracts the keypoint indices to measure between, from the upstream pose
        detector's keypoint mapping.
        """
        # Upstream pose config carries the keypoint mapping
        upstream_config = self.loaded_inputs[f"pose_{self.dim}"].upstream_config
        keypoint_mapping_name = upstream_config.keypoint_mapping  # e.g., "coco_wholebody"
        # Get predictions_mapping from runtime_config for proximity index
        self.keypoint_mapping = getattr(self.predictions_mapping.human_pose, keypoint_mapping_name)

        # Get indexes of keypoints from detectors config (i.e. [nose])
        self.used_keypoints = self.detector_config.used_keypoints
        keypoints_index = self.keypoint_mapping.keypoints_index.body
        for keypoint in self.used_keypoints:
            if keypoint not in keypoints_index:
                logging.error(f"Given used_keypoint could not be found in predictions_mapping: {keypoint}")

        self.keypoint_index = [keypoints_index[keypoint] for keypoint in self.used_keypoints]

    def compute(self):
        """
        Computes the proximity component.

        This method calculates the Euclidean distance between the keypoints of personL
        and personR. If the length of the keypoint index list is greater than 1, the
        midpoint of the keypoints will be used in the proximity measure.

        Returns:
            DetectorOutput: carrying the proximity scores under this detector's npz_key
            for the "proximity" component. Validation and saving are handled by
            BaseFeature.run().
        """
        pose = self.loaded_inputs[f"pose_{self.dim}"]
        pose_data, pose_axes = pose.data, pose.axes

        # extract exactly 2 subjects
        # TODO: extend it to support arbitrary amount of subjects
        if len(pose_axes.subjects) != 2:
            raise ValueError(
                f"Proximity requires exactly 2 subjects, got {len(pose_axes.subjects)}: {pose_axes.subjects}."
            )
        personL, personR = pose_data

        # Split coord vs conf on axis4 (last column). Keypoints axis is 2 (per subject slice).
        coordsL = personL[:, :, self.keypoint_index, :-1]
        coordsR = personR[:, :, self.keypoint_index, :-1]
        confL = personL[:, :, self.keypoint_index, -1]
        confR = personR[:, :, self.keypoint_index, -1]

        # Average coordinates over the selected keypoints, per frame.
        average_coords_L = np.mean(coordsL, axis=2, keepdims=True)
        average_coords_R = np.mean(coordsR, axis=2, keepdims=True)

        # Euclidean distance between the two people's average coordinates, per frame. Shape (C, F, 1).
        proximity_score = np.linalg.norm(average_coords_L - average_coords_R, axis=-1)

        # Min confidence across the averaged keypoints for each subject, then across the pair.
        # Matches velocity_body's min-propagation: a low-quality joint drags the pair's confidence down.
        min_conf_L = np.nanmin(confL, axis=2, keepdims=True)  # (C, F, 1)
        min_conf_R = np.nanmin(confR, axis=2, keepdims=True)  # (C, F, 1)
        pair_conf = np.minimum(min_conf_L, min_conf_R)  # (C, F, 1)

        # Pack [distance, confidence] on axis3 (no axis4 for this scalar metric).
        packed = np.concatenate([proximity_score, pair_conf], axis=-1)  # (C, F, 2)
        # The score is symmetric; store it under both subject slots (axis0 = subjects = 2).
        distance = np.stack((packed, packed), axis=0)  # (S, C, F, 2)

        distance_axes = pose_axes.replace(labels=["distance", "confidence_score"], data=[])
        return DetectorOutput().add("proximity", f"body_distance_{self.dim}", data=distance, axes=distance_axes)

    def visualization(self, out: DetectorOutput):
        distance = out.get("proximity", f"body_distance_{self.dim}")
        pro_utils.visualize_proximity_score(distance, self.viz_folder, self.used_keypoints)


class BodyDistance2D(BodyDistance):
    """Body distance computed on 2D pose keypoints (distance in pixels)."""

    algorithm_type = "body_distance_2d"
    dim = "2d"

    inputs = [NpzDetectorInput("body_joints", "pose_2d", schema=VECTOR_2D_CONF_PER_LABEL)]
    outputs = [NpzDetectorOutput("proximity", "body_distance_2d", schema=DISTANCE)]


class BodyDistance3D(BodyDistance):
    """Body distance computed on 3D pose keypoints (distance in real-world units)."""

    algorithm_type = "body_distance_3d"
    dim = "3d"

    inputs = [NpzDetectorInput("body_joints", "pose_3d", schema=VECTOR_3D_CONF_PER_LABEL)]
    outputs = [NpzDetectorOutput("proximity", "body_distance_3d", schema=DISTANCE)]
