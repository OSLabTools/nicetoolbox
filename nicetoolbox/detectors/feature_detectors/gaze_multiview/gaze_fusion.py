"""
Fuses per-camera world gaze vectors into a single world-space direction.
"""

import os

import cv2
import numpy as np

from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF, VECTOR_3D_CONF
from nicetoolbox_core.data.loaded_array import NpzArray
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ....utils import video as vd
from ....utils import visual_utils as vis_ut
from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ...method_detectors.filters import SGFilter
from ..base_feature import BaseFeature


class GazeFusion(BaseFeature):
    """
    Fuse per-camera world gaze into a single world-space unit vector per (subject, frame),
    optionally smoothed with a Savitzky-Golay filter, and reproject the fused direction back
    to each camera as a 2D arrow for visualization.

    Inputs come already lifted to world space by the upstream method detector (e.g. eth_xgaze).
    Confidence (mean landmark score) is combined across views with the same weights used to
    fuse the vectors, and carried through onto both 3D and 2D outputs.
    """

    components = ["gaze_multiview"]
    algorithm_type = "gaze_fusion"

    inputs = [
        NpzDetectorInput("gaze_individual", "gaze_per_camera_3d", schema=VECTOR_3D_CONF),  # gaze vec per view
        NpzDetectorInput("gaze_individual", "gaze_origin_2d", schema=VECTOR_2D_CONF),  # visualization
        NpzDetectorInput("gaze_individual", "gaze_per_camera_2d", schema=VECTOR_2D_CONF),  # visualization
    ]

    def resolve_outputs(self):
        outputs = [
            NpzDetectorOutput("gaze_multiview", "gaze_3d", schema=VECTOR_3D_CONF),
            NpzDetectorOutput("gaze_multiview", "gaze_2d", schema=VECTOR_2D_CONF),
            NpzDetectorOutput("gaze_multiview", "gaze_origin_2d", schema=VECTOR_2D_CONF),
        ]
        if self.detector_config.filtered:
            outputs.append(NpzDetectorOutput("gaze_multiview", "gaze_3d_filtered", schema=VECTOR_3D_CONF))
            outputs.append(NpzDetectorOutput("gaze_multiview", "gaze_2d_filtered", schema=VECTOR_2D_CONF))
        return outputs

    def _initialize_detector(self) -> None:
        self.filtered = self.detector_config.filtered
        self.window = self.detector_config.window_length
        self.poly = self.detector_config.polyorder
        self.fusion_method = self.detector_config.fusion_method
        self.subject_view_map = self.detector_config.subject_view_map
        self.calibration = self.data.calibration

    def compute(self) -> DetectorOutput:
        """Fuse per-camera world gaze into a single unit vector per (subject, frame), reproject
        to each camera as 2D, and (optionally) produce a temporally-smoothed variant.

        Confidence is fused with the same weights that fused the vectors and carried through onto
        every output (3D fused, 2D reprojection, and filtered pair). Returns a DetectorOutput;
        BaseFeature.run() validates + saves it.
        """
        gaze = self.loaded_inputs["gaze_per_camera_3d"]  # (S, C, F, 4) = (x, y, z, conf)
        gaze_origin = self.loaded_inputs["gaze_origin_2d"]  # (S, C, F, 3) = (x, y, conf)

        subjects = gaze.axes.subjects
        cameras = gaze.axes.cameras
        frames = gaze.axes.frames

        # separate confidence from the data
        vectors = gaze.data[..., :3]  # (S, C, F, 3)
        conf = gaze.data[..., 3]  # (S, C, F)

        # Fuse per-camera gaze into a single world unit vector + fused confidence.
        fused_xyz, fused_conf = self._fuse_vectors(vectors, conf, subjects, cameras)  # (S, F, 3), (S, F)

        # Assemble the fused world gaze under the ["3d"] pseudo-camera slot with confidence.
        gaze_3d = self._pack_gaze_3d(fused_xyz, fused_conf, subjects, frames)
        # Reproject fused direction back to each real camera; conf broadcast across cameras.
        gaze_2d = self._pack_gaze_2d(fused_xyz, fused_conf, subjects, cameras, frames)

        out = DetectorOutput()
        out.add_array("gaze_multiview", "gaze_3d", gaze_3d)
        out.add_array("gaze_multiview", "gaze_2d", gaze_2d)
        out.add_array("gaze_multiview", "gaze_origin_2d", gaze_origin.array)

        if self.filtered:
            # Smooth the fused world vector (a coherent single-view track), not the per-camera gaze.
            smoothed = SGFilter(self.window, self.poly).apply(fused_xyz[:, np.newaxis, :, :], is_3d=True)
            filtered_xyz = smoothed[:, 0, :, :]
            # Renormalize after smoothing so the direction stays a unit vector.
            norms = np.linalg.norm(filtered_xyz, axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                filtered_xyz = filtered_xyz / norms

            gaze_3d_filtered = self._pack_gaze_3d(filtered_xyz, fused_conf, subjects, frames)
            gaze_2d_filtered = self._pack_gaze_2d(filtered_xyz, fused_conf, subjects, cameras, frames)
            out.add_array("gaze_multiview", "gaze_3d_filtered", gaze_3d_filtered)
            out.add_array("gaze_multiview", "gaze_2d_filtered", gaze_2d_filtered)

        return out

    def _fuse_vectors(
        self, vectors: np.ndarray, conf: np.ndarray, subjects: list[str], cameras: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse per-camera unit gaze vectors across the cameras axis.

        weighted_average uses conf as weights; NaN vectors are dropped from the sum. Confidence
        is combined with the same weights (self-weighted mean = sum(w*w)/sum(w)) so a low-quality
        view diluting the vector also dilutes the reported confidence.

        select_view adopts a specific camera's per-view estimate for each subject via subject_view_map

        Returns:
            fused_xyz: (S, F, 3) unit vectors (NaN where every camera is missing).
            fused_conf: (S, F) fused confidence (NaN where every camera is missing).
        """
        if self.fusion_method == "select_view":
            try:
                cam_indices = np.array([cameras.index(self.subject_view_map[s]) for s in subjects])
            except KeyError as e:
                raise ValueError(f"subject_view_map missing entry for subject {e}") from e
            except ValueError as e:
                raise ValueError(f"subject_view_map camera not in cameras {cameras}: {e}") from e
            subj_indices = np.arange(len(subjects))
            fused = vectors[subj_indices, cam_indices]  # (S, F, 3) — already unit
            fused_conf = conf[subj_indices, cam_indices]  # (S, F)
            return fused, fused_conf

        # Mask out camera slots where the vector or its confidence is NaN.
        valid = ~np.isnan(vectors).any(axis=-1) & ~np.isnan(conf)  # (S, C, F)

        if self.fusion_method == "weighted_average":
            weights = np.where(valid, np.nan_to_num(conf, nan=0.0), 0.0)  # (S, C, F)
            vec_safe = np.nan_to_num(vectors, nan=0.0)

            w = weights[..., np.newaxis]  # (S, C, F, 1)
            weighted_sum = np.sum(vec_safe * w, axis=1)  # (S, F, 3)
            total_weight = np.sum(weights, axis=1)  # (S, F)

            with np.errstate(divide="ignore", invalid="ignore"):
                avg_vector = weighted_sum / total_weight[..., np.newaxis]
                fused_conf = np.sum(weights * weights, axis=1) / total_weight  # (S, F)
            # Where no camera contributed, mark NaN.
            no_support = total_weight == 0
            avg_vector[no_support] = np.nan
            fused_conf[no_support] = np.nan
        else:
            avg_vector = np.nanmean(vectors, axis=1)  # (S, F, 3)
            fused_conf = np.nanmean(np.where(valid, conf, np.nan), axis=1)  # (S, F)

        # Re-normalize to a unit direction.
        norms = np.linalg.norm(avg_vector, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            fused = avg_vector / norms

        return fused, fused_conf

    def _pack_gaze_3d(
        self, fused_xyz: np.ndarray, fused_conf: np.ndarray, subjects: list[str], frames: list[str]
    ) -> NpzArray:
        """Pack (S, F, 3) fused direction + (S, F) conf into a (S, 1, F, 4) NpzArray on the ['3d'] slot."""
        n_subjects, n_frames = fused_conf.shape
        data = np.full((n_subjects, 1, n_frames, 4), np.nan, dtype=float)
        data[:, 0, :, :3] = fused_xyz
        data[:, 0, :, 3] = fused_conf
        axes = VECTOR_3D_CONF.make_axes(subjects, ["3d"], frames)
        return NpzArray(data, axes)

    def _pack_gaze_2d(
        self,
        fused_xyz: np.ndarray,
        fused_conf: np.ndarray,
        subjects: list[str],
        cameras: list[str],
        frames: list[str],
    ) -> NpzArray:
        """Reproject fused world gaze into each camera as a 2D arrow; broadcast fused conf per camera."""
        projected = self._project_to_cameras(fused_xyz, cameras)  # (S, C, F, 2)
        conf_broadcast = np.broadcast_to(fused_conf[:, np.newaxis, :, np.newaxis], projected.shape[:-1] + (1,))
        data = np.concatenate([projected, conf_broadcast], axis=-1)  # (S, C, F, 3)
        axes = VECTOR_2D_CONF.make_axes(subjects, cameras, frames)
        return NpzArray(data, axes)

    def _project_to_cameras(self, world_gaze: np.ndarray, cameras: list[str]) -> np.ndarray:
        """Project a world direction (S, F, 3) into each camera's image plane as (dx, dy) arrows.

        NaN gaze frames propagate to NaN pixel arrows through the trig chain, so no explicit
        masking is needed. Calibration is guaranteed present by the upstream eth_xgaze fail-fast.
        """
        n_subj, n_frames, _ = world_gaze.shape
        projected = np.full((n_subj, len(cameras), n_frames, 2), np.nan)

        for cam_idx, cam_name in enumerate(cameras):
            _, _, cam_R, _ = vis_ut.get_cam_para_studio(self.calibration, cam_name)
            image_width = self.calibration[cam_name]["image_size"][0]

            for sub_id in range(n_subj):
                dx, dy = vis_ut.reproject_gaze_to_camera_view_vectorized(cam_R, world_gaze[sub_id], image_width)
                projected[sub_id, cam_idx, :, 0] = -dx
                projected[sub_id, cam_idx, :, 1] = -dy

        return projected

    def visualization(self, out: DetectorOutput) -> None:
        """Draw the fused 2D gaze arrow (red) plus every per-view arrow (yellow, thin, half-length)
        on each source frame, then stitch a video per camera.

        Reads gaze_2d_filtered when temporal filtering is enabled, else gaze_2d. Origins come
        straight from the pass-through gaze_origin_2d. The per-view arrows all share the same
        camera-X origin — divergence between yellow arrows shows which views pull the fusion.
        """
        gaze_key = "gaze_2d_filtered" if self.filtered else "gaze_2d"
        gaze_2d = out.get("gaze_multiview", gaze_key).data[..., :2]  # (S, C, F, 2)
        origins = out.get("gaze_multiview", "gaze_origin_2d").data[..., :2]  # (S, C, F, 2)
        per_view = self.loaded_inputs["gaze_per_camera_2d"].data[..., :2]  # (S, V, F, 2)

        cameras = out.get("gaze_multiview", gaze_key).axes.cameras
        n_subjects, n_views = per_view.shape[0], per_view.shape[1]

        dataloader = ImagePathsByFrameIndexLoader(self.data.get_input_recipes(), expected_cameras=cameras)

        for frame_idx, (real_idx, files) in enumerate(dataloader):
            for cam_name, path in files.items():
                if cam_name not in cameras:
                    continue
                cam_idx = cameras.index(cam_name)

                img = cv2.imread(str(path))
                if img is None:
                    continue

                for sub_id in range(n_subjects):
                    origin = origins[sub_id, cam_idx, frame_idx]
                    if np.isnan(origin).any():
                        continue
                    origin_i = np.round(origin).astype(np.int32)

                    # Per-view arrows: thin yellow, half length, drawn first so the fused arrow
                    # renders on top.
                    for view_idx in range(n_views):
                        view_vec = per_view[sub_id, view_idx, frame_idx]
                        if np.isnan(view_vec).any():
                            continue
                        view_end = np.round(origin + 0.5 * view_vec).astype(np.int32)
                        cv2.arrowedLine(
                            img,
                            origin_i,
                            view_end,
                            color=(0, 255, 255),  # yellow (BGR)
                            thickness=1,
                            line_type=cv2.LINE_AA,
                            tipLength=0.2,
                        )

                    # Fused arrow: red, full length.
                    vec = gaze_2d[sub_id, cam_idx, frame_idx]
                    if np.isnan(vec).any():
                        continue
                    end_point = np.round(origin + vec).astype(np.int32)
                    cv2.arrowedLine(
                        img,
                        origin_i,
                        end_point,
                        color=(0, 0, 255),
                        thickness=2,
                        line_type=cv2.LINE_AA,
                        tipLength=0.2,
                    )

                out_dir = os.path.join(self.viz_folder, cam_name)
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, f"{real_idx:09d}.jpg"), img)

        for cam in cameras:
            vd.frames_to_video(
                os.path.join(self.viz_folder, cam),
                os.path.join(self.viz_folder, f"{cam}.mp4"),
                fps=self.data.fps,
                start_frame=int(dataloader.start),
            )
