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
EYE_CLOSED_STATE = BooleanSchema(labels_columns=("left_eye", "right_eye"))


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
        NpzDetectorOutput("eye_closed_state", "state", schema=EYE_CLOSED_STATE),
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
            n_subjects, n_cameras, _, n_eyes = eye_closed.shape
            for subject_idx in range(n_subjects):
                for camera_idx in range(n_cameras):
                    for eye_idx in range(n_eyes):
                        eye_closed[subject_idx, camera_idx, :, eye_idx] = threshold_utils.filter_duration(
                            eye_closed[subject_idx, camera_idx, :, eye_idx],
                            self.data.fps,
                            self.min_duration,
                            self.max_duration,
                        )

        eye_closed[missing] = np.nan

        state_axes = EYE_CLOSED_STATE.make_axes(axes.subjects, axes.cameras, axes.frames)

        out = DetectorOutput()
        out.add("eye_closed_state", "state", data=eye_closed, axes=state_axes)

        logging.info(f"Computation of feature detector for {self.components} completed.")
        return out

    def visualization(self, out: DetectorOutput) -> None:
        """Plot the per-frame closed state and, when landmarks are wired, draw eye overlays."""
        logging.info(f"Visualizing the feature detector output {self.components}.")

        state = out.get("eye_closed_state", "state")
        cameras = state.axes.cameras

        threshold_utils.plot_eye_closed_states(
            viz_folder=self.viz_folder,
            eye_closed=state.data,
            subjects_descr=self.subjects_descr,
            camera_names=cameras,
            cam_sees_subjects=self.data.cam_sees_subjects,
            threshold=self.threshold,
        )

        # The frame overlay needs the eye points the score was computed from.
        landmarks_input = self.loaded_inputs.get("eye_landmarks_2d")
        if landmarks_input is None:
            logging.info("No eye_landmarks_2d input wired; skipping the eye overlay video.")
            return

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
