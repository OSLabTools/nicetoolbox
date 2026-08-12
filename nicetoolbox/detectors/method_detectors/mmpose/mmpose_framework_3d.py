"""
MMPose 3D frameworks and specific algorithm definitions.
"""

import logging
import os
from typing import Any, Optional

import numpy as np

from ....configs.schemas.detectors_instances_configs import MotionbertAlgorithmConfig
from ....utils import video as vd
from ... import config_handler as confh
from ...utils import pose_utils
from ...utils.filters import SGFilter
from ...utils.keypoints_post_processing import extract_key_per_value
from .mmpose_framework_base import BaseMMPose


def _pose_det_dataset_from_keypoint_mapping(mapping_name: str) -> str:
    """MMPose dataset_name for convert_keypoint_definition from upstream keypoint_mapping."""
    n = (mapping_name or "").lower()
    if n in ("coco_wholebody", "coco", "aic"):
        return "coco"
    if n == "mpii":
        return "mpii"
    if n == "human36m":
        return "human36m"
    raise ValueError(
        f"MotionBERT: unsupported upstream keypoint_mapping {mapping_name!r} for convert_keypoint_definition."
    )


def _upstream_pose_model_ref(upstream: Any) -> str:
    """Alias string or config path for the upstream 2D pose model (Pose3DInferencer / metadata)."""
    pose_cfg = getattr(upstream, "pose_config", None)
    if pose_cfg:
        return str(pose_cfg)
    assets = getattr(upstream, "required_assets", None) or {}
    pcf = assets.get("pose_config_file")
    if pcf:
        return str(pcf)
    raise ValueError(
        "MotionBERT: upstream body_joints algorithm has no pose_config or required_assets.pose_config_file."
    )


class MMPose3D(BaseMMPose):
    """
    MMPose3D handles native 3D pose lifters (e.g. MotionBERT). Post-inference applies
    optional temporal filtering and confidence masking on stored 2D/3D arrays; 3D stays
    in the lifter's root-relative frame (no PnP or camera alignment).
    """

    def post_inference(self):
        """
        Post-inference processing for 3D pose estimation.
        """
        for _component, result_folder in self.result_folders.items():
            prediction_file = os.path.join(result_folder, f"{self.algorithm_instance}.npz")
            prediction = np.load(prediction_file, allow_pickle=True)
            data_description = prediction["data_description"].item()

            results_2d = prediction["2d"]
            results_bbox_2d = prediction["bbox_2d"]
            results_3d = prediction.get("3d", None)

            if results_3d is None:
                logging.error(f"No 3D data found in {prediction_file}. Inference may have failed.")
                continue

            iou_array = pose_utils.create_iou_all_pairs(results_bbox_2d)

            # Filtering is applied to both 2D and 3D data.
            if self.filtered:
                logging.info("APPLYING filtering to 2D and 3D data...")
                filter = SGFilter(self.filter_window_length, self.filter_polyorder)

                results_2d_filtered = filter.apply(results_2d.copy())
                results_2d_interpolated = results_2d_filtered.copy()

                results_3d_filtered = filter.apply(results_3d.copy())
                results_3d_final = results_3d_filtered.copy()
            else:
                results_2d_interpolated = results_2d.copy()
                results_3d_final = results_3d.copy()

            keypoint_conf_threshold = self.min_detection_confidence

            # A mask is created for low confidence 2D points, and interpolation is applied.
            low_conf_mask_2d = results_2d_interpolated[:, :, :, :, 2] < keypoint_conf_threshold
            results_2d_interpolated[low_conf_mask_2d, 0:2] = np.nan
            results_2d_interpolated = pose_utils.interpolate_data(results_2d_interpolated, is_3d=False)

            # A mask is created for low confidence 3D points, and interpolation is applied.
            low_conf_mask_3d = results_3d_final[:, :, :, :, 3] < keypoint_conf_threshold
            results_3d_final[low_conf_mask_3d, 0:3] = np.nan
            results_3d_final = pose_utils.interpolate_data(results_3d_final, is_3d=True)

            logging.info("Packaging 2D/3D arrays (root-relative 3D; no PnP / camera alignment).")

            # Data descriptions are updated.
            data_description["2d_interpolated"] = data_description["2d"]
            if self.filtered:
                data_description["2d_filtered"] = data_description["2d"]
                data_description["3d_filtered"] = data_description["3d"]

            data_description["bbox_overlap"] = dict(
                axis0=data_description["2d"]["axis0"],
                axis1=data_description["2d"]["axis1"],
                axis2=data_description["2d"]["axis2"],
                axis3=[f"with_{subj}" for subj in self.subjects_descr],
            )

            # Results are packaged and saved into the .npz file.
            results_dict = {
                "2d": results_2d,
                "2d_interpolated": results_2d_interpolated,
                "bbox_2d": results_bbox_2d,
                "bbox_overlap": iou_array,
                "3d": results_3d_final,
                "data_description": data_description,
            }

            if self.filtered:
                results_dict["2d_filtered"] = results_2d_filtered
                results_dict["3d_filtered"] = results_3d_filtered

            np.savez_compressed(prediction_file, **results_dict)

    def _get_2d_array_for_visualization(self, prediction):
        """Merge temporally interpolated 2D with raw detections where interpolation left gaps."""
        interp = prediction["2d_interpolated"]
        raw = prediction["2d"] if "2d" in prediction.files else None
        return pose_utils.merge_2d_for_pose_overlay(interp, projected=None, raw=raw)


# =============================================================================
# 3D Algorithm Implementations
# =============================================================================


class MotionBERT(MMPose3D):
    """
    Handler for MotionBERT models trained on Human3.6M (17 Keypoints).
    """

    algorithm_type = "motionbert"
    components = ["body_joints_local"]

    def compute_viz_folder(self, _visualize: bool) -> Optional[str]:
        """Return None so base __init__ does not mkdir the visualization folder.

        The subprocess creates the per-camera frame folders itself; the main
        process assembles the mp4s in visualization() after inference.
        """
        return None

    def visualization(self, _):
        for _, result_folder in self.result_folders.items():
            viz_dir = os.path.join(result_folder, self.algorithm_instance, "visualization")
            for camera_name in self.camera_names:
                cam_frames_dir = os.path.join(viz_dir, camera_name)
                out_mp4 = os.path.join(viz_dir, f"{camera_name}.mp4")
                rc = vd.frames_to_video(cam_frames_dir, out_mp4, int(self.fps), start_frame=int(self.video_start))
                if rc == 0:
                    logging.info("MotionBERT visualization video -> %s", out_mp4)
                else:
                    logging.warning("frames_to_video failed (code %s) for %s", rc, out_mp4)

    def _initialize_detector(self) -> MotionbertAlgorithmConfig.RuntimeConfig:
        runtime = super()._initialize_detector()
        cfg = self.detector_config
        src = next(e[1] for e in cfg.input_detector_names if len(e) == 2 and e[0] == "body_joints")
        upstream = self.subsequence_context.get_detector_config(src)
        npz_path = self.io.get_detector_output_folder("body_joints", src, "result") / f"{src}.npz"
        if not npz_path.is_file():
            logging.warning(
                "MotionBERT: source 2D NPZ not found at %s — run detector '%s' before motionbert.",
                npz_path,
                src,
            )

        merged_assets = dict(upstream.required_assets)
        merged_assets.update(dict(cfg.required_assets))

        if npz_path.is_file():
            bundle = np.load(npz_path, allow_pickle=True)
            try:
                k_src = int(bundle["2d"].shape[-2])
            finally:
                bundle.close()
            coco_indices = list(range(k_src))
        else:
            coco_indices = list(range(17))

        payload = runtime.model_dump()
        payload["pose_config"] = _upstream_pose_model_ref(upstream)
        payload["motionbert_2d_pose_det_dataset"] = _pose_det_dataset_from_keypoint_mapping(upstream.keypoint_mapping)
        payload["motionbert_2d_coco_body_indices"] = coco_indices
        payload["motionbert_2d_keypoints_npz"] = str(npz_path)
        payload["required_assets"] = merged_assets
        return MotionbertAlgorithmConfig.RuntimeConfig(**payload)

    def get_per_component_keypoint_mapping(self, keypoints_indices):
        indices = dict(body_joints_local=confh.flatten_list(list(keypoints_indices.body.values())))
        description = dict(body_joints_local=confh.flatten_list(extract_key_per_value(keypoints_indices.body)))
        return indices, description
