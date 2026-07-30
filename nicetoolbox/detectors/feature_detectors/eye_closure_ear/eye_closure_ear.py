"""
Eye Closure (EAR method) feature detector class.
"""

import logging

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL, VECTOR_2D_PER_LABEL, AnyOf, ArraySchema
from nicetoolbox_core.data.loaded_array import select_array
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ..base_feature import BaseFeature
from . import utils as ear_utils

# One EAR score per eye, per subject/camera/frame (axis3 = eye side, no data axis).
EYE_CLOSURE_SCORE = ArraySchema(labels_columns=("left_eye", "right_eye"))


class EyeClosureEar(BaseFeature):
    """
    Computes the eye closure score using the Eye Aspect Ratio (EAR) method from upstream
    2D face landmarks.

    The landmark source is wired in config and may come from either family:
      - face_landmarks (e.g. hrnetw48) -> human_pose keypoint mapping
      - head_orientation (e.g. spiga)  -> head_orientation keypoint mapping
    Both ship (subjects, cameras, frames, landmarks, [x, y]); only the eye-index
    resolution differs, driven by whether the upstream config declares a keypoint_mapping.

    Output is (subjects, cameras, frames, [left_eye, right_eye]) EAR scores.
    """

    components = ["eye_closure_score"]  # TODO: delete me
    algorithm_type = "eye_closure_ear"

    # TODO: SPIGA doesn't return confidence, force it in the future
    landmarks_2d_schema = AnyOf(VECTOR_2D_CONF_PER_LABEL, VECTOR_2D_PER_LABEL)
    # TODO: SPIGA isn't face_landmarks, decompose it to multiple components. Now it is fine, nothing validate component.
    inputs = [
        NpzDetectorInput("face_landmarks", "landmarks_2d", schema=landmarks_2d_schema),
    ]
    outputs = [
        NpzDetectorOutput("eye_closure_score", "score", schema=EYE_CLOSURE_SCORE),
        NpzDetectorOutput("eye_closure_score", "eye_landmarks_2d", schema=VECTOR_2D_PER_LABEL),
    ]

    def _initialize_detector(self) -> None:
        """Resolve the left/right eye landmark indices from the upstream keypoint mapping."""
        self.camera_names = self.detector_config.camera_names
        upstream_config = self.loaded_inputs["landmarks_2d"].upstream_config
        keypoint_mapping_name = getattr(upstream_config, "keypoint_mapping", None)

        # TODO: unify keypoint mapping lookup here
        if keypoint_mapping_name:
            # HumanPose family (e.g. coco_wholebody): face landmarks are a slice of the
            # whole-body keypoint set, so global IDs must be remapped to slice-local ones.
            self.keypoint_mapping = getattr(self.predictions_mapping.human_pose, keypoint_mapping_name)
            face_indices = self.keypoint_mapping.keypoints_index.face
            self.left_eye_indices = ear_utils.resolve_eye_indices(face_indices, "left_eye")
            self.right_eye_indices = ear_utils.resolve_eye_indices(face_indices, "right_eye")
        else:
            # HeadOrientation family (spiga): face landmarks are already their own array.
            self.keypoint_mapping = self.predictions_mapping.head_orientation.spiga
            face_indices = self.keypoint_mapping.keypoints_index.face
            self.left_eye_indices = face_indices["left_eye"]
            self.right_eye_indices = face_indices["right_eye"]

        self.eye_layout = self.keypoint_mapping.eye_layout
        self.eyes_landmark_indexes = list(self.left_eye_indices) + list(self.right_eye_indices)

        # prepare left and right labels
        left_eye_labels = [f"left_eye_{i}" for i in range(len(self.left_eye_indices))]
        right_eye_labels = [f"right_eye_{i}" for i in range(len(self.right_eye_indices))]
        self.eye_landmark_labels = left_eye_labels + right_eye_labels

    def compute(self) -> DetectorOutput:
        """Compute the per-eye EAR score for every subject/camera/frame.

        Returns a DetectorOutput carrying the score under the eye_closure_score component;
        validation and saving are handled by BaseFeature.run().
        """
        # Restrict to the configured cameras; raises if the upstream lacks any of them.
        landmarks_raw = self.loaded_inputs["landmarks_2d"].array
        landmarks = select_array(landmarks_raw, cameras=self.camera_names)

        # get landmarks coordinates
        axes = landmarks.axes
        coords = landmarks.data[..., :2]  # drop confidence

        # select all eye landmarks
        eye_landmarks = coords[:, :, :, self.eyes_landmark_indexes, :]
        eye_landmarks_axes = VECTOR_2D_PER_LABEL.make_axes(
            axes.subjects, axes.cameras, axes.frames, labels=self.eye_landmark_labels
        )
        # select each eye individually
        left_eye = coords[:, :, :, self.left_eye_indices, :]
        right_eye = coords[:, :, :, self.right_eye_indices, :]

        # calculate EAR metrics for them
        left_eye_ear = ear_utils.calculate_ear(left_eye, self.eye_layout)
        right_eye_ear = ear_utils.calculate_ear(right_eye, self.eye_layout)

        # save it into final tensor
        score_axes = EYE_CLOSURE_SCORE.make_axes(axes.subjects, axes.cameras, axes.frames)
        eye_closure_score = np.stack([left_eye_ear, right_eye_ear], axis=-1)

        # write output
        out = DetectorOutput()
        out.add("eye_closure_score", "score", data=eye_closure_score, axes=score_axes)
        out.add("eye_closure_score", "eye_landmarks_2d", data=eye_landmarks, axes=eye_landmarks_axes)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out

    def visualization(self, out: DetectorOutput) -> None:
        """Draw the eye bounding boxes and EAR scores on the source frames, then stitch a video."""
        logging.info(f"Visualizing the feature detector output {self.components}.")

        score = out.get("eye_closure_score", "score")
        # Same camera restriction as compute(), so landmarks and scores share a camera axis.
        landmarks = select_array(self.loaded_inputs["landmarks_2d"].array, cameras=self.camera_names)
        cameras = score.axes.cameras

        dataloader = ImagePathsByFrameIndexLoader(config=self.data.get_input_recipes(), expected_cameras=cameras)

        ear_utils.visualize_eye_closure_scores(
            viz_folder=self.viz_folder,
            dataloader=dataloader,
            camera_names=cameras,
            subjects_descr=self.subjects_descr,
            cam_sees_subjects=self.data.cam_sees_subjects,
            landmarks=landmarks.data,
            landmarks_axes=landmarks.axes,
            eye_closure_score=score.data,
            left_eye_indices=self.left_eye_indices,
            right_eye_indices=self.right_eye_indices,
            fps=self.data.fps,
            video_start_frame_index=self.data.video_start_frame_index,
        )

        logging.info(f"Visualization of feature detector {self.components} completed.")
