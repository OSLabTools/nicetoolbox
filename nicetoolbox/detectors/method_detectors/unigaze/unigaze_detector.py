"""
UniGaze method detector class.
"""

import logging
import os

import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF, VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF
from nicetoolbox_core.data.loaded_array import NpzArray
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ....configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ....utils import visual_utils as vis_ut
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ...utils.draw_2d import RenderContext, draw_direction_vector
from ...utils.filters import SGFilter
from ..base_method import BaseMethod

LANDMARK_NAMES = [f"landmark_{i}" for i in range(68)]


class UniGaze(BaseMethod):
    """
    The UniGaze class is a method detector that computes the gaze_individual
    component using the UniGaze model.
    """

    components = ["gaze_individual"]
    algorithm_type = "unigaze"

    # Outputs
    def resolve_outputs(self) -> list:
        """Add the filtered world/2D gaze outputs only when temporal filtering is enabled."""
        outputs = [
            NpzDetectorOutput("gaze_individual", "gaze_3d_camera_space", schema=VECTOR_3D_CONF),
            NpzDetectorOutput("gaze_individual", "gaze_3d", schema=VECTOR_3D_CONF),
            NpzDetectorOutput("gaze_individual", "gaze_2d", schema=VECTOR_2D_CONF),
            NpzDetectorOutput("gaze_individual", "landmarks_2d", schema=VECTOR_2D_CONF_PER_LABEL),
            NpzDetectorOutput("gaze_individual", "gaze_origin_2d", schema=VECTOR_2D_CONF),
        ]
        if self.detector_config.filtered:
            outputs.append(NpzDetectorOutput("gaze_individual", "gaze_3d_filtered", schema=VECTOR_3D_CONF))
            outputs.append(NpzDetectorOutput("gaze_individual", "gaze_2d_filtered", schema=VECTOR_2D_CONF))
        return outputs

    def _initialize_detector(self) -> MethodDetectorRuntime:
        """
        Initialize the UniGaze method detector.
        """
        self.subjects = self.data.subjects_descr
        self.cameras = self.detector_config.camera_names
        self.video_start = self.data.video_start_frame_index
        self.calibration = self.data.calibration
        self.cam_sees_subjects = self.data.cam_sees_subjects
        self.results_folder = self.result_folders[self.components[0]]

        self.filtered = self.detector_config.filtered
        if self.filtered:
            self.filter_window_length = self.detector_config.window_length
            self.filter_polyorder = self.detector_config.polyorder

        # (2) Initialise data loader
        self.dataloader = ImagePathsByFrameIndexLoader(
            config=self.data.get_input_recipes(), expected_cameras=self.cameras
        )
        self.frame_names = self.data.frame_labels

        # Used by visualization() to walk the source frames via the shared 2d renderer.
        self.render_context = RenderContext(
            dataloader=self.dataloader,
            cam_sees_subjects=self.cam_sees_subjects,
            fps=self.data.fps,
            video_start=self.video_start,
        )

        return super()._initialize_detector()

    def post_inference(self) -> DetectorOutput:
        """
        Post-processing after inference completes. Loads raw results, applies SG filtering,
        projects to 2D camera views, and builds a DetectorOutput object.
        """
        raw_path = os.path.join(self.out_folders[self.components[0]], "unigaze_inference_raw.npz")
        try:
            raw = np.load(raw_path, allow_pickle=True)
        except FileNotFoundError:
            logging.error(f"UniGaze: raw inference file {raw_path} not found, skipping post-inference.")
            return None

        # Assign raw pack detections to subjects. The gaze origin is the 3D face center that
        # UniGaze normalizes around, projected to 2D during inference, so it matches the point
        # the gaze vector is actually anchored to.
        gaze_cam, landmarks_2d, gaze_origin_2d = self._assign_subjects(raw)

        # World-lift the raw camera gaze
        gaze_3d = self._world_lift(gaze_cam)

        # Project world-lifted gaze to 2D camera views
        gaze_2d = self._project_to_2d(gaze_3d)

        # Apply filtering if enabled
        gaze_3d_filtered = gaze_2d_filtered = None
        # TODO: very sensitive to NaN, use more robust filtering
        if self.filtered:
            # Safeguard window length for short sequences/tests
            win_len = self.filter_window_length

            logging.info(
                f"APPLYING Savitzky-Golay filtering (window={win_len}, "
                f"polyorder={self.filter_polyorder}) to UniGaze 3D Gaze Individual data..."
            )
            smoothed = SGFilter(win_len, self.filter_polyorder).apply(gaze_cam.data, is_3d=True)
            # convert to world coordinates and then project to 2D camera views
            gaze_3d_filtered = self._world_lift(NpzArray(smoothed, gaze_cam.axes))
            gaze_2d_filtered = self._project_to_2d(gaze_3d_filtered)

        out = DetectorOutput()
        out.add_array("gaze_individual", "landmarks_2d", landmarks_2d)
        out.add_array("gaze_individual", "gaze_origin_2d", gaze_origin_2d)
        out.add_array("gaze_individual", "gaze_3d_camera_space", gaze_cam)
        out.add_array("gaze_individual", "gaze_3d", gaze_3d)
        out.add_array("gaze_individual", "gaze_2d", gaze_2d)
        if self.filtered:
            out.add_array("gaze_individual", "gaze_3d_filtered", gaze_3d_filtered)
            out.add_array("gaze_individual", "gaze_2d_filtered", gaze_2d_filtered)

        return out

    def _assign_subjects(self, raw):
        """Assign the raw pack's ragged per-camera detections to dense subject-indexed arrays.

          - Too many faces for a camera: keep the N ones that are leftmost, or simply sort
            left-to-right and keep the first N (N = subjects this camera sees).
          - Too few faces: cannot tell which subject is missing, so mark all of this camera's
            subjects missing (NaN) for that frame.
          - Otherwise: assign faces left-to-right onto the subject slots this camera sees.

        Returns:
            gaze_cam (NpzArray): camera-local gaze, (subjects, cameras, frames, x/y/z/conf).
            landmarks_2d (NpzArray): face landmarks, (subjects, cameras, frames, 68, x/y/conf).
            gaze_origin_2d (NpzArray): gaze origin, (subjects, cameras, frames, x/y/conf).
        """
        per_frame_outputs = raw["per_frame_outputs"]
        n_subjects = len(self.subjects)
        n_cams = len(self.cameras)
        n_frames = len(per_frame_outputs)

        gaze_cam = np.full((n_subjects, n_cams, n_frames, 4), np.nan, dtype=float)
        landmarks_2d = np.full((n_subjects, n_cams, n_frames, 68, 3), np.nan, dtype=float)
        gaze_origin_2d = np.full((n_subjects, n_cams, n_frames, 3), np.nan, dtype=float)

        for frame_idx, frame_bundle in enumerate(per_frame_outputs):
            for cam_idx, cam_name in enumerate(self.cameras):
                detections = frame_bundle.get(cam_name, []) if frame_bundle else []
                subjects_by_cam = self.cam_sees_subjects[cam_name]
                n_seen = len(subjects_by_cam)

                if len(detections) < n_seen:
                    if detections:
                        logging.debug(
                            f"UniGaze: camera '{cam_name}' frame {frame_idx} detected "
                            f"{len(detections)} face(s) < {n_seen} expected; marking all missing."
                        )
                    continue

                # Sort detections left-to-right by x-coordinate of the landmarks
                detections_sorted = sorted(detections, key=lambda det: det["landmarks_2d"][:, 0].min())

                if len(detections_sorted) > n_seen:
                    detections_sorted = detections_sorted[:n_seen]

                # Assign left-to-right onto the subject slots this camera sees.
                for subject_index, det in zip(subjects_by_cam, detections_sorted):
                    gaze_cam[subject_index, cam_idx, frame_idx, :3] = det["gaze_cam"]
                    gaze_cam[subject_index, cam_idx, frame_idx, 3] = 1.0
                    landmarks_2d[subject_index, cam_idx, frame_idx, :, :2] = det["landmarks_2d"]
                    landmarks_2d[subject_index, cam_idx, frame_idx, :, 2] = 1.0
                    gaze_origin_2d[subject_index, cam_idx, frame_idx, :2] = det["gaze_origin_2d"]
                    gaze_origin_2d[subject_index, cam_idx, frame_idx, 2] = 1.0

        gaze_axes = VECTOR_3D_CONF.make_axes(self.subjects, self.cameras, self.frame_names)
        landmarks_axes = VECTOR_2D_CONF_PER_LABEL.make_axes(
            self.subjects, self.cameras, self.frame_names, LANDMARK_NAMES
        )
        origin_axes = VECTOR_2D_CONF.make_axes(self.subjects, self.cameras, self.frame_names)
        return (
            NpzArray(gaze_cam, gaze_axes),
            NpzArray(landmarks_2d, landmarks_axes),
            NpzArray(gaze_origin_2d, origin_axes),
        )

    def _world_lift(self, gaze_cam: NpzArray) -> NpzArray:
        """Rotate each camera's local gaze direction into the shared world frame."""
        data = gaze_cam.data.copy()
        for cam_idx, cam_name in enumerate(self.cameras):
            cam_rotation = np.asarray(self.calibration[cam_name]["rotation_matrix"], dtype=float)

            vectors = data[:, cam_idx, :, :3]  # (subjects, frames, 3)
            world = vectors @ cam_rotation  # apply inv(R) = R.T (since R is orthogonal)
            norms = np.linalg.norm(world, axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                world = world / norms
            data[:, cam_idx, :, :3] = world

        return NpzArray(data, gaze_cam.axes)

    def _project_to_2d(self, gaze_world: NpzArray) -> NpzArray:
        """Reproject a world-space gaze direction back into each camera as a 2D pixel arrow."""
        n_subjects, n_cams, n_frames, _ = gaze_world.data.shape
        projected = np.full((n_subjects, n_cams, n_frames, 3), np.nan, dtype=float)

        for cam_idx, cam_name in enumerate(self.cameras):
            _, _, cam_rotation, _ = vis_ut.get_cam_para_studio(self.calibration, cam_name)

            for subject_idx in range(n_subjects):
                vectors = gaze_world.data[subject_idx, cam_idx, :, :3]  # (frames, 3)
                dx, dy = vis_ut.reproject_gaze_to_camera_view_vectorized(cam_rotation, vectors)
                projected[subject_idx, cam_idx, :, 0] = -dx
                projected[subject_idx, cam_idx, :, 1] = -dy
                projected[subject_idx, cam_idx, :, 2] = gaze_world.data[subject_idx, cam_idx, :, 3]  # confidence

        axes = VECTOR_2D_CONF.make_axes(self.subjects, self.cameras, self.frame_names)
        return NpzArray(projected, axes)

    def visualization(self, out: DetectorOutput) -> None:
        """
        Draw each subject's 2D gaze arrow on the source frames and stitch a video per camera.
        """
        gaze_key = "gaze_2d_filtered" if self.filtered else "gaze_2d"
        gaze_2d = out.get("gaze_individual", gaze_key)
        gaze_origin_2d = out.get("gaze_individual", "gaze_origin_2d")
        draw_direction_vector(gaze_2d, gaze_origin_2d, self.render_context, self.viz_folder)
