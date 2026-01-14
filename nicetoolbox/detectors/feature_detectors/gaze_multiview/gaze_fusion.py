"""
MultiviewFusion feature detector.
Fuses raw per-camera gaze vectors into a single world-space vector.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from nicetoolbox_core.dataloader import ImagePathsByFrameIndexLoader

from ....utils import video as vd
from ....utils import visual_utils as vis_ut
from ...data import Data
from ...in_out import IO
from ...method_detectors.filters import SGFilter
from ..base_feature import BaseFeature


class GazeFusion(BaseFeature):
    """
    Computes a single 3D gaze vector from multiple camera views and/or multiple
    algorithms.

    Given the fused 3D gaze vectors, the user can optionally select to apply
    a Savitzky-Golay filter for temporal smoothing. Finally, the fused 3D gaze
    vectors are projected back to each camera view to obtain 2D gaze points
    for visualization (If enabled).

    Expected Input:
        - Method detectors outputting '3d' array.
        - Shape: (Subjects, Cameras, Frames, 3)
        - Optional: 'confidence_scores' (Subjects, Cameras, Frames) for weighted fusion.
    """

    components = ["gaze_multiview"]
    algorithm = "gaze_fusion"

    def __init__(self, config: Dict, io: IO, data: Data) -> None:
        """
        Initialize the MultiviewFusion detector.

        Args:
            config (Dict): Configuration dictionary.
            io (Any): IO handler.
            data (Any): Data handler.
        """
        super().__init__(config, io, data, requires_out_folder=True)

        self.config = config
        self.calibration = data.calibration

        # Load Metadata from inputs immediately
        self.camera_names = []
        self.subjects = []
        self.frame_indices = []

        # Data Cache
        self.raw_inputs: List[Dict[str, Any]] = []

        # Parse inputs
        self._load_inputs()

        # Config Parameters
        self.filtered = config.get("filtered", True)
        self.window = config.get("window_length", 11)
        self.poly = config.get("polyorder", 2)
        self.fusion_method = config.get("fusion_method", "average")
        self.ensemble_enabled = config.get("ensemble_fusion", False)

        # Init DataLoader config for visualization if needed
        self.dataloader = None
        if config.get("visualize", False) and self.camera_names:
            self.dataloader_config = data.get_input_recipe().copy()
            # TODO: input_recipe is required, we can assume it's passed in config from
            # TODO: main or we can add the recipe inside the base feature detector init.

    def _load_inputs(self) -> None:
        """
        Loads data from all input files defined in configuration.
        Populates self.raw_inputs and metadata.
        """
        if not self.input_files:
            logging.error("No input files found for MultiviewFusion.")
            return

        for file_path in self.input_files:
            try:
                data = np.load(file_path, allow_pickle=True)
                if "3d" not in data.files:
                    logging.warning(f"File {file_path} missing '3d'. Skipping.")
                    continue

                # Extract Metadata from the first valid file
                if not self.camera_names:
                    desc = data["data_description"].item()
                    self.camera_names = desc["3d"]["axis1"]
                    self.subjects = desc["3d"]["axis0"]
                    self.frame_indices = desc["3d"]["axis2"]

                # Extract Algorithm Name from path, e.g.:
                # /gaze_individual/multiview_eth_xgaze.npz -> multiview_eth_xgaze
                algo_name = Path(file_path).stem

                # Extract Confidence from Landmarks
                # Landmarks shape: (S, C, F, 6, 3) -> [u, v, conf]
                conf = None
                landmarks = data.get("landmarks_2d")

                if landmarks is not None and landmarks.shape[-1] == 3:
                    # Take the 3rd element (index 2) from the last axis
                    raw_conf = landmarks[..., 2]  # Result: (S, C, F, 6)

                    # Reduce the 6 landmarks to 1 score per face (mean)
                    conf = np.nanmean(raw_conf, axis=-1)  # Result: (S, C, F)

                input_entry = {
                    "name": algo_name,
                    "vectors": data["3d"],
                    "conf": conf,  # for weighted fusion
                    "landmarks": landmarks[..., :2],  # for viz arrow with face origin
                }
                self.raw_inputs.append(input_entry)

            except Exception as e:
                logging.error(f"Failed to load input {file_path}: {e}")

    def compute(self) -> List[Dict[str, Any] | None]:
        """
        Run the fusion pipeline.

        Returns:
            A list of dictionaries containing fused results.
        """
        if not self.raw_inputs:
            return [None]

        all_results = []

        # 1. Individual Fusion (Per Algorithm)
        for inp in self.raw_inputs:
            base_algo_name = inp["name"]
            vectors = inp["vectors"]
            conf_scores = inp["conf"]

            logging.info(f"Fusing results for {base_algo_name}...")

            # Fuse -> Filter
            fused_3d, fused_3d_filtered = self._process_single_input(vectors, conf_scores)

            # Save and collect
            filename = f"{self.algorithm}_{base_algo_name}"
            out_dict = self._save_fused_result(fused_3d, fused_3d_filtered, filename)
            out_dict["_meta_title"] = f"Algorithm: {base_algo_name}"
            all_results.append(out_dict)

        # 2. Ensemble Fusion (Across Base Algorithms - Optional)
        if self.ensemble_enabled and len(self.raw_inputs) > 1:
            logging.info("Running Ensemble Fusion...")

            all_vectors = [inp["vectors"] for inp in self.raw_inputs]
            # Stack along Camera axis (axis 1) to treat as "more views"
            # (S, C, F, 3) -> (S, Total_C, F, 3)
            combined_vectors = np.concatenate(all_vectors, axis=1)

            all_conf_scores = None
            if self.fusion_method == "weighted_average":
                conf_scores = []
                for inp in self.raw_inputs:
                    if inp["conf"] is not None:
                        conf_scores.append(inp["conf"])
                    else:
                        # Fallback to ones if missing
                        conf_scores.append(np.ones(inp["vectors"].shape[:3]))
                all_conf_scores = np.concatenate(conf_scores, axis=1)

            # Fuse combined data
            ensemble_fused, ensemble_fused_filtered = self._process_single_input(combined_vectors, all_conf_scores)

            # Save
            out_dict = self._save_fused_result(ensemble_fused, ensemble_fused_filtered, f"{self.algorithm}_ensemble")
            out_dict["_meta_title"] = "Ensemble Fusion"
            all_results.append(out_dict)

        return all_results

    def _process_single_input(
        self, vectors: np.ndarray, confidence: Optional[np.ndarray]
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Runs the core feature detector logic: Fuse -> Filter.

        Args:
            vectors: (S, C, F, 3)
            confidence: (S, C, F) or None

        Returns:
            fused_3d: (S, F, 3)
        """
        # 1. Fuse
        fused = self._fuse_vectors(vectors, confidence)
        fused_filtered = None

        # 2. Filter
        if self.filtered:
            # Reshape for SGFilter: (S, F, 3) -> (S, 1, F, 3) -> Add fake camera axis
            # SGFilter usually expects (Subjects, Cameras, Frames, Dims)
            fused_4d = fused[:, np.newaxis, :, :]

            filter_obj = SGFilter(self.window, self.poly)
            filtered_4d = filter_obj.apply(fused_4d, is_3d=True)

            # Remove fake camera axis - Back to (S, F, 3)
            fused_filtered = filtered_4d[:, 0, :, :]

        return fused, fused_filtered

    def _fuse_vectors(self, vectors: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fuses vectors using the configured method.

        Args:
            vectors (np.ndarray): Shape (S, C_total, F, 3).
            weights (Optional[np.ndarray]): Shape (S, C_total, F).

        Returns:
            np.ndarray: Shape (S, F, 3).
        """
        if self.fusion_method == "weighted_average" and weights is not None:
            # Weighted Mean
            # Expand weights to (S, C, F, 1) for broadcasting
            # (We have one score per face detected after averaging landmarks scores)
            w_expanded = weights[:, :, :, np.newaxis]

            # Replace NaN weights with 0 (to ignore them in sum)
            w_safe = np.nan_to_num(w_expanded, nan=0.0)

            # Mask NaNs (Treat as 0 weight)
            valid_mask = ~np.isnan(vectors).any(axis=-1, keepdims=True)
            w_final = w_safe * valid_mask

            # Replace NaN vectors with 0
            vec_safe = np.nan_to_num(vectors, nan=0.0)

            # Now compute the nominator and denominator of the weighted average
            # across the camera axis (axis=1) to fuse the vectors from different views
            weighted_sum = np.sum(vec_safe * w_final, axis=1)
            total_weight = np.sum(w_final, axis=1)

            # Final division plus avoid div by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                avg_vector = weighted_sum / total_weight

        else:
            # Simple Mean (default)
            avg_vector = np.nanmean(vectors, axis=1)

        # Re-normalize to Unit Vectors
        norms = np.linalg.norm(avg_vector, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            fused = avg_vector / norms

        return fused

    def _save_fused_result(
        self, fused_3d: np.ndarray, fused_3d_filtered: np.ndarray | None, filename: str
    ) -> Dict[str, Any]:
        """
        Projects, packages, and saves a fused and optionally fused filtered results.
        """
        out_dict = {
            "gaze_fused": fused_3d[:, np.newaxis, :, :].copy(),  # Add 3d camera axis
            "data_description": {},
        }

        # 1. Project Raw Fused
        proj_raw = self._project_to_cameras(fused_3d)
        out_dict["gaze_2d"] = proj_raw

        # Metadata for Raw
        out_dict["data_description"]["gaze_fused"] = {
            "axis0": self.subjects,
            "axis1": ["3d"],
            "axis2": self.frame_indices,
            "axis3": ["coordinate_x", "coordinate_y", "coordinate_z"],
        }
        out_dict["data_description"]["gaze_2d"] = {
            "axis0": self.subjects,
            "axis1": self.camera_names,
            "axis2": self.frame_indices,
            "axis3": ["coordinate_u", "coordinate_v"],
        }

        # 2. Handle Filtered
        if fused_3d_filtered is not None:  # Add 3d camera axis (Common NICE format)
            out_dict["gaze_fused_filtered"] = fused_3d_filtered[:, np.newaxis, :, :].copy()

            # Project Filtered
            proj_filtered = self._project_to_cameras(fused_3d_filtered)
            out_dict["gaze_2d_filtered"] = proj_filtered

            # Metadata for Filtered (Copy structure)
            out_dict["data_description"]["gaze_fused_filtered"] = out_dict["data_description"]["gaze_fused"]
            out_dict["data_description"]["gaze_2d_filtered"] = out_dict["data_description"]["gaze_2d"]

        # Save
        save_path = os.path.join(self.result_folders["gaze_multiview"], f"{filename}.npz")
        np.savez_compressed(save_path, **out_dict)

        return out_dict

    def _project_to_cameras(self, world_gaze: np.ndarray) -> np.ndarray:
        """
        Projects world vectors back to camera planes.

        Args:
            world_gaze (np.ndarray): Fused vectors (S, F, 3).

        Returns:
            np.ndarray: Projected 2D points (S, C, F, 2).
        """
        n_subj, n_frames, _ = world_gaze.shape
        n_cams = len(self.camera_names)

        projected = np.full((n_subj, n_cams, n_frames, 2), np.nan)

        for cam_idx, cam_name in enumerate(self.camera_names):
            if not self.calibration or cam_name not in self.calibration:
                logging.warning(
                    f"Calibration missing or camera '{cam_name}' not found;" " skipping projection for this camera."
                )
                continue

            calib = self.calibration[cam_name]

            image_width = calib["image_size"][0]
            _, _, cam_R, _ = vis_ut.get_cam_para_studio(self.calibration, cam_name)

            for sub_id in range(n_subj):
                vectors = world_gaze[sub_id]  # (F, 3)

                # Vectorized Projection per subject
                valid_mask = ~np.isnan(vectors).any(axis=1)
                if not np.any(valid_mask):
                    continue

                valid_vectors = vectors[valid_mask]

                dx, dy = vis_ut.reproject_gaze_to_camera_view_vectorized(cam_R, valid_vectors, image_width)
                projected[sub_id, cam_idx, valid_mask, 0] = -dx
                projected[sub_id, cam_idx, valid_mask, 1] = -dy

        return projected

    def post_compute(self):
        """No post-compute needed."""
        pass

    def visualization(self, results_list: List[Dict[str, Any]]) -> None:
        """
        Generates visualization images with fused gaze overlays.
        """
        logging.info("Visualizing Fused Gaze...")

        # We need landmarks for the origin point.
        # We use the first input's landmarks as the reference.
        # TODO: Different algorithms may have different landmark sets.
        first_input = self.raw_inputs[0]
        landmarks_2d = first_input.get("landmarks")  # (S, C, F, 6, 3)

        if landmarks_2d is None:
            logging.warning("No landmarks found in input. Cannot visualize gaze origin.")
            return

        landmarks_2d = landmarks_2d[..., :2]  # Use only (u, v)
        face_centers = np.nanmean(landmarks_2d, axis=-2)  # (S, C, F, 2)

        for result in results_list:
            title = result.get("_meta_title", "Gaze Fusion Result")
            logging.info(f"Visualizing: {title}")

            dataloader = ImagePathsByFrameIndexLoader(self.dataloader_config, expected_cameras=self.camera_names)

            # Prefer to use filtered gaze if available
            if "gaze_2d_filtered" in result:
                gaze_2d = result["gaze_2d_filtered"]  # (S, C, F, 2)
                logging.info("Visualizing FILTERED fused gaze.")
            else:
                gaze_2d = result["gaze_2d"]  # (S, C, F, 2)
                logging.info("Visualizing RAW fused gaze.")

            for frame_idx, (real_idx, files) in enumerate(dataloader):
                for cam_name, path in files.items():
                    if cam_name not in self.camera_names:
                        continue
                    cam_idx = self.camera_names.index(cam_name)

                    img = cv2.imread(str(path))
                    if img is None:
                        continue

                    for sub_id in range(gaze_2d.shape[0]):
                        # Check Landmarks
                        lms = landmarks_2d[sub_id, cam_idx, frame_idx]
                        # lms shape is (6, 3) -> [u, v, score]
                        if np.isnan(lms).all():
                            continue

                        face_center = face_centers[sub_id, cam_idx, frame_idx]

                        vec = gaze_2d[sub_id, cam_idx, frame_idx]
                        if np.isnan(vec).any():
                            continue

                        # Draw Arrow (vec is dx, dy)
                        end_point = np.round(face_center + vec).astype(np.int32)
                        face_center = np.round(face_center).astype(np.int32)

                        cv2.arrowedLine(
                            img,
                            face_center,
                            end_point,
                            color=(0, 0, 255),  # Red
                            thickness=2,
                            line_type=cv2.LINE_AA,
                            tipLength=0.2,
                        )

                    # TODO: Different viz folders per result! (Also for MP4)
                    out_dir = os.path.join(self.viz_folder, cam_name)
                    os.makedirs(out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(out_dir, f"{real_idx:09d}.jpg"), img)

            # Generate MP4
            for cam in self.camera_names:
                vd.frames_to_video(
                    os.path.join(self.viz_folder, cam),
                    os.path.join(self.viz_folder, f"{cam}.mp4"),
                    fps=self.config.get("fps", 30),
                    start_frame=int(dataloader.start),
                )
