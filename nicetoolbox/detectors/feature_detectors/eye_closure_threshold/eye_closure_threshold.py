"""
Eye Closure State (Threshold method) feature detector class.
"""

import logging

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_PER_LABEL, BooleanSchema
from nicetoolbox_core.data.loaded_array import select_array
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ..base_feature import BaseFeature
from ..eye_closure_ear.eye_closure_ear import EYE_CLOSURE_SCORE
from . import utils as threshold_utils

# Per-eye closed/open state, per subject/camera/frame (axis3 = eye side, no data axis).
# Boolean-valued but float-backed, so a missing upstream score stays NaN rather than
# collapsing into "open".
EYE_CLOSED_STATE = BooleanSchema(labels_columns=("left_eye", "right_eye", "both_eyes"))


class EyeClosureThreshold(BaseFeature):
    """
    Decides whether the eyes are closed by thresholding an upstream eye_closure_score.

    Emits a binary state per eye: 1 = closed, 0 = open, NaN where the upstream score is
    missing. With min_duration/max_duration set, only contiguous closed runs whose length
    falls in that window are kept — which turns the same detector into a blink detector.

    The eye landmarks are taken as a second input, used only to draw eye contours in the
    visualization. They come from the same upstream component as the score, already sliced to
    the eye points and labelled `left_eye_*` / `right_eye_*`, so no keypoint-mapping lookup is
    needed here. Optional, because not every score producer emits them.
    """

    components = ["eye_closed_state"]  # TODO: delete me
    algorithm_type = "eye_closure_threshold"

    inputs = [
        NpzDetectorInput("eye_closure_score", "score", schema=EYE_CLOSURE_SCORE),
        NpzDetectorInput("eye_closure_score", "eye_landmarks_2d", schema=VECTOR_2D_PER_LABEL, optional=True),
    ]
    outputs = [
        NpzDetectorOutput("eye_closed_state", "per_camera_state", schema=EYE_CLOSED_STATE),
        NpzDetectorOutput("eye_closed_state", "global_state", schema=EYE_CLOSED_STATE),
    ]

    def _initialize_detector(self) -> None:
        self.threshold = self.detector_config.threshold
        self.min_duration = self.detector_config.min_duration
        self.max_duration = self.detector_config.max_duration
        self.camera_names = self.detector_config.camera_names

    def compute(self) -> DetectorOutput:
        """Threshold the upstream score into a per-eye closed/open state.

        Returns a DetectorOutput carrying the state under the eye_closed_state component;
        validation and saving are handled by BaseFeature.run().
        """
        # Restrict to the configured cameras; raises if the upstream lacks any of them.
        score = select_array(self.loaded_inputs["score"].array, cameras=self.camera_names)
        axes = score.axes

        # 1.0 where the eye is closed (score below threshold), 0.0 where open. Float, so that
        # a missing upstream score stays NaN rather than collapsing into "open" — comparing
        # against NaN yields False, which would otherwise be indistinguishable from an open eye.
        missing = np.isnan(score.data)
        eye_closed = np.where(score.data < self.threshold, 1.0, 0.0)

        # Keep only closed runs whose duration falls inside [min_duration, max_duration].
        # Run detection needs a clean 0/1 signal, so filter with the gaps treated as open and
        # restore the NaNs afterwards. A closed run interrupted by missing frames is therefore
        # measured as several shorter runs, not one long one.
        if self.min_duration > 0.0 or self.max_duration is not None:
            eye_closed = threshold_utils.filter_duration(
                eye_closed,
                self.data.fps,
                self.min_duration,
                self.max_duration,
                axis=2,
            )

        eye_closed[missing] = np.nan

        # Compute both_eyes state: 1.0 if both are 1.0, 0.0 if either is 0.0, else NaN
        left = eye_closed[..., 0]
        right = eye_closed[..., 1]
        both = np.full_like(left, np.nan)
        both[(left == 1.0) & (right == 1.0)] = 1.0
        both[(left == 0.0) | (right == 0.0)] = 0.0

        eye_closed = np.concatenate([eye_closed, both[..., np.newaxis]], axis=-1)

        # Compute global left, right, and both-eyes closed state:
        # 1.0 if closed for all cameras where subject is visible
        # 0.0 if at least one visible camera is open
        # NaN if not visible/missing in all cameras
        any_left_open = np.any(left == 0.0, axis=1)  # (S, F)
        any_left_closed = np.any(left == 1.0, axis=1)  # (S, F)
        global_left = np.where(any_left_open, 0.0, np.where(any_left_closed, 1.0, np.nan))

        any_right_open = np.any(right == 0.0, axis=1)  # (S, F)
        any_right_closed = np.any(right == 1.0, axis=1)  # (S, F)
        global_right = np.where(any_right_open, 0.0, np.where(any_right_closed, 1.0, np.nan))

        global_both = np.full_like(global_left, np.nan)
        global_both[(global_left == 1.0) & (global_right == 1.0)] = 1.0
        global_both[(global_left == 0.0) | (global_right == 0.0)] = 0.0

        global_data = np.stack([global_left, global_right, global_both], axis=-1)[:, np.newaxis, :, :]

        state_axes = EYE_CLOSED_STATE.make_axes(axes.subjects, axes.cameras, axes.frames)
        global_axes = EYE_CLOSED_STATE.make_axes(axes.subjects, ["3d"], axes.frames)

        out = DetectorOutput()
        out.add("eye_closed_state", "per_camera_state", data=eye_closed, axes=state_axes)
        out.add("eye_closed_state", "global_state", data=global_data, axes=global_axes)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out

    def visualization(self, out: DetectorOutput) -> None:
        """Plot the per-frame closed state and, when landmarks are wired, draw eye overlays."""
        logging.info(f"Visualizing the feature detector output {self.components}.")

        for array_key in ("per_camera_state", "global_state"):
            state = out.get("eye_closed_state", array_key)
            threshold_utils.plot_eye_closed_states(
                viz_folder=self.viz_folder,
                eye_closed=state.data,
                subjects_descr=self.subjects_descr,
                camera_names=state.axes.cameras,
                cam_sees_subjects=self.data.cam_sees_subjects,
                threshold=self.threshold,
            )

        # The frame overlay needs the eye points the score was computed from.
        landmarks_input = self.loaded_inputs.get("eye_landmarks_2d")
        if landmarks_input is None:
            logging.info("No eye_landmarks_2d input wired; skipping the eye overlay video.")
            return

        state = out.get("eye_closed_state", "per_camera_state")
        cameras = state.axes.cameras

        # Same camera restriction as compute(), so every array shares one camera axis.
        landmarks = select_array(landmarks_input.array, cameras=self.camera_names)
        score = select_array(self.loaded_inputs["score"].array, cameras=self.camera_names)

        # The array is already sliced to the eye points; split it by label prefix.
        labels = landmarks.axes.labels
        left_eye_indices = [i for i, name in enumerate(labels) if name.startswith("left_eye")]
        right_eye_indices = [i for i, name in enumerate(labels) if name.startswith("right_eye")]

        dataloader = ImagePathsByFrameIndexLoader(config=self.data.get_input_recipes(), expected_cameras=cameras)

        threshold_utils.visualize_eye_closed_videos(
            viz_folder=self.viz_folder,
            dataloader=dataloader,
            camera_names=cameras,
            subjects_descr=self.subjects_descr,
            cam_sees_subjects=self.data.cam_sees_subjects,
            landmarks=landmarks.data,
            landmarks_axes=landmarks.axes,
            eye_closure_score=score.data,
            eye_closed=state.data,
            left_eye_indices=left_eye_indices,
            right_eye_indices=right_eye_indices,
            fps=self.data.fps,
            video_start_frame_index=self.data.video_start_frame_index,
        )

        logging.info(f"Visualization of feature detector {self.components} completed.")
