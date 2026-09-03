import logging
import os
import warnings

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

# Raw native pack written by eth_xgaze_inference.py (kept in sync manually; importing the
# inference module here would pull its eth_xgaze-venv-only deps into the main toolbox env).
RAW_INFERENCE_NPZ_NAME = "eth_xgaze_inference_raw.npz"
LANDMARK_NAMES = ["right_eye_0", "right_eye_1", "left_eye_0", "left_eye_1", "mouth_0", "mouth_1"]


class EthXgaze(BaseMethod):
    """
    The ETH XGaze class is a method detector that computes the gaze_individual
    component.

    The method detector computes the gaze of individuals in the scene using multiple
    cameras.It provides the necessary preparations and post-inference visualizations to
    integrate the ETH XGaze algorithm into our pipeline.
    """

    components = ["gaze_individual"]
    algorithm_type = "eth_xgaze"

    # Outputs
    def resolve_outputs(self) -> list:
        """Add the filtered world/2D gaze outputs only when temporal filtering is enabled."""
        outputs = [
            NpzDetectorOutput("gaze_individual", "gaze_3d_camera_space", schema=VECTOR_3D_CONF),
            NpzDetectorOutput("gaze_individual", "gaze_3d", schema=VECTOR_3D_CONF),
            NpzDetectorOutput("gaze_individual", "gaze_2d", schema=VECTOR_2D_CONF),
            NpzDetectorOutput("gaze_individual", "gaze_origin_2d", schema=VECTOR_2D_CONF),
            # TODO: move this to facial landmarks component (see insight_face or spiga)
            NpzDetectorOutput("gaze_individual", "landmarks_2d", schema=VECTOR_2D_CONF_PER_LABEL),
        ]
        if self.detector_config.filtered:
            outputs.append(NpzDetectorOutput("gaze_individual", "gaze_3d_filtered", schema=VECTOR_3D_CONF))
            outputs.append(NpzDetectorOutput("gaze_individual", "gaze_2d_filtered", schema=VECTOR_2D_CONF))
        return outputs

    def _initialize_detector(self) -> MethodDetectorRuntime:
        """
        Initialize the XGaze method detector.
        """
        # (1) Convenience reference
        self.subjects = self.data.subjects_descr
        self.frame_names = self.data.frame_labels
        self.cameras = self.detector_config.camera_names
        self.video_start = self.data.video_start_frame_index
        self.cam_sees_subjects = self.data.cam_sees_subjects
        self.results_folder = self.result_folders[self.components[0]]
        self.filtered = self.detector_config.filtered
        if self.filtered:
            self.filter_window_length = self.detector_config.window_length
            self.filter_polyorder = self.detector_config.polyorder

        self.dataloader = ImagePathsByFrameIndexLoader(
            config=self.data.get_input_recipes(), expected_cameras=self.cameras
        )

        # Used by visualization() to walk the source frames via the shared 2d renderer.
        self.render_context = RenderContext(
            dataloader=self.dataloader,
            cam_sees_subjects=self.cam_sees_subjects,
            fps=self.data.fps,
            video_start=self.video_start,
        )

        # Sanity checks
        # ETH-XGaze needs per-camera calibration: intrinsics for the head-pose/normalization at inference
        self.calibration = self.data.calibration
        if not self.calibration:  # TODO: fake it?
            raise ValueError(f"UniGaze '{self.algorithm_instance}' requires camera calibration, but none is available.")

        # SVG window cannot exceed the number of frames to smooth.
        if self.filtered and self.filter_window_length > len(self.frame_names):
            raise ValueError(
                f"ETH-XGaze '{self.algorithm_instance}': filter window_length "
                f"({self.filter_window_length}) exceeds the {len(self.frame_names)} frames available; "
                f"reduce window_length or disable filtering."
            )

        return super()._initialize_detector()

    def post_inference(self) -> DetectorOutput:
        """
        Structure the raw ETH-XGaze pack into the toolbox's dense component arrays.

        Returns the produced DetectorOutput; BaseMethod.run() validates + saves it (and passes
        it to visualization()).
        """
        raw_path = os.path.join(self.out_folders[self.components[0]], RAW_INFERENCE_NPZ_NAME)
        raw = np.load(raw_path, allow_pickle=True)

        # dense camera-local gaze + landmarks (typed with axes), faces assigned to slots.
        gaze_cam, landmarks_2d = self._assign_subjects(raw)

        # TODO: get to the eth xgaze guts, extract real camera space 3d head pos
        # Face origin per (subject, camera, frame): mean of the 6 face landmarks.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gaze_origin_2d_data = np.nanmean(landmarks_2d.data, axis=3)  # (S, C, F, 3) = (x, y, conf)
        gaze_origin_2d = NpzArray(
            gaze_origin_2d_data, VECTOR_2D_CONF.make_axes(self.subjects, self.cameras, self.frame_names)
        )

        # world-lift the raw camera gaze, and reproject it back to each camera as 2D.
        gaze_3d = self._world_lift(gaze_cam)
        gaze_2d = self._project_to_2d(gaze_3d)

        # filtered variants (only when temporal filtering is enabled). Smooth the camera
        # gaze first (coherent single-view tracks), then lift and reproject.
        gaze_3d_filtered = gaze_2d_filtered = None
        if self.filtered:
            smoothed = SGFilter(self.filter_window_length, self.filter_polyorder).apply(gaze_cam.data, is_3d=True)
            gaze_3d_filtered = self._world_lift(NpzArray(smoothed, gaze_cam.axes))
            gaze_2d_filtered = self._project_to_2d(gaze_3d_filtered)

        # Collect all produced arrays (filtered pair only present when filtering is enabled).
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

    def visualization(self, out: DetectorOutput) -> None:
        """
        Draw each subject's 2D gaze arrow on the source frames and stitch a video per camera.

        The arrow starts at the subject's mean face-landmark position (gaze_origin_2d) and points
        along the reprojected gaze (gaze_2d, or gaze_2d_filtered when filtering is on), which is a
        unit heading scaled to a fixed on-screen length by the shared renderer.
        """
        gaze_key = "gaze_2d_filtered" if self.filtered else "gaze_2d"
        gaze_2d = out.get("gaze_individual", gaze_key)
        gaze_origin_2d = out.get("gaze_individual", "gaze_origin_2d")
        draw_direction_vector(gaze_2d, gaze_origin_2d, self.render_context, self.viz_folder)

    def _world_lift(self, gaze_cam: NpzArray) -> NpzArray:
        """Rotate each camera's local gaze direction into the shared world frame.

        gaze is a direction, so only the camera rotation applies (no translation): for each camera
        gaze_world = inv(R) @ gaze_cam, renormalized to a unit vector. The confidence channel
        (axis3 index 3) is carried through unchanged.
        """
        data = gaze_cam.data.copy()
        for cam_idx, cam_name in enumerate(self.cameras):
            cam_rotation = np.asarray(self.calibration[cam_name]["rotation_matrix"], dtype=float)
            r_inv = np.linalg.inv(cam_rotation)

            vectors = data[:, cam_idx, :, :3]  # (subjects, frames, 3)
            world = vectors @ r_inv.T  # apply inv(R) to each xyz vector
            norms = np.linalg.norm(world, axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                world = world / norms
            data[:, cam_idx, :, :3] = world

        return NpzArray(data, gaze_cam.axes)

    def _project_to_2d(self, gaze_world: NpzArray) -> NpzArray:
        """Reproject a world-space gaze direction back into each camera as a 2D unit direction.

        Projecting into camera C re-applies that camera's rotation, undoing the world-lift, so the
        result equals projecting the original camera-local gaze — a (dx, dy) image-plane direction
        of unit length, resolution-independent. Consumers scale it by a pixel length when drawing.
        The confidence channel (axis3 index 3) is carried through unchanged onto the 2D output.
        """
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

    def _assign_subjects(self, raw):
        """Assign the raw pack's ragged per-camera detections to dense subject-indexed arrays.

          - Too many faces for a camera: keep the N highest mean-confidence faces (N = subjects
            this camera sees), then restore their left-to-right order.
          - Too few faces: cannot tell which subject is missing, so mark all of this camera's
            subjects missing (NaN) for that frame.
          - Otherwise: assign faces left-to-right onto the subject slots this camera sees.

        Returns:
            gaze_cam (NpzArray): camera-local gaze, (subjects, cameras, frames, x/y/z/conf).
            landmarks_2d (NpzArray): face landmarks, (subjects, cameras, frames, 6, x/y/conf).
        """
        per_frame_outputs = raw["per_frame_outputs"]
        n_subjects = len(self.subjects)
        n_cams = len(self.cameras)
        n_frames = len(per_frame_outputs)

        # gaze_cam axis3 = (x, y, z, confidence). ETH-XGaze has no native gaze confidence, so we
        # use the mean of the 6 face-landmark detection scores as a proxy quality signal.
        gaze_cam = np.full((n_subjects, n_cams, n_frames, 4), np.nan, dtype=float)
        landmarks_2d = np.full((n_subjects, n_cams, n_frames, 6, 3), np.nan, dtype=float)

        for frame_idx, frame_bundle in enumerate(per_frame_outputs):
            for cam_idx, cam_name in enumerate(self.cameras):
                detections = frame_bundle.get(cam_name, []) if frame_bundle else []
                subjects_by_cam = self.cam_sees_subjects[cam_name]
                n_seen = len(subjects_by_cam)

                if len(detections) < n_seen:
                    # Missing a face: cannot know which subject, so leave all NaN for this frame.
                    if detections:
                        logging.debug(
                            f"ETH-XGaze: camera '{cam_name}' frame {frame_idx} detected "
                            f"{len(detections)} face(s) < {n_seen} expected; marking all missing."
                        )
                    continue

                if len(detections) > n_seen:
                    # Too many faces: keep the N highest mean-confidence, restore left-to-right order.
                    mean_scores = [float(np.nanmean(det["landmark_scores"])) for det in detections]
                    keep = sorted(np.argsort(mean_scores)[-n_seen:])
                    detections = [detections[i] for i in keep]

                # Assign left-to-right onto the subject slots this camera sees.
                for subject_index, det in zip(subjects_by_cam, detections):
                    gaze_cam[subject_index, cam_idx, frame_idx, :3] = det["gaze_cam"]
                    gaze_cam[subject_index, cam_idx, frame_idx, 3] = float(np.nanmean(det["landmark_scores"]))
                    landmarks_2d[subject_index, cam_idx, frame_idx, :, :2] = det["landmarks_2d"]
                    landmarks_2d[subject_index, cam_idx, frame_idx, :, 2] = det["landmark_scores"]

        gaze_axes = VECTOR_3D_CONF.make_axes(self.subjects, self.cameras, self.frame_names)
        landmarks_axes = VECTOR_2D_CONF_PER_LABEL.make_axes(
            self.subjects, self.cameras, self.frame_names, LANDMARK_NAMES
        )
        return NpzArray(gaze_cam, gaze_axes), NpzArray(landmarks_2d, landmarks_axes)
