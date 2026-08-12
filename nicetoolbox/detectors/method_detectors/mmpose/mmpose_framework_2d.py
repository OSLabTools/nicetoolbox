"""
MMPose 2D framework — single consolidated detector class.

Previously this file held six subclasses (HRNetw48, VitPose, VitPoseHuge,
RTMPoseLAIC, RTMPoseWHolebody, RTMPoseMPII) that differed only in their
class-level `components` list and `get_per_component_keypoint_mapping()`.
They are now a single `MMPose2D` class whose components are declared in TOML
config and whose keypoint mapping logic dispatches on the requested components.
"""

import logging
import os

import numpy as np

from ....utils import check_and_exception as check
from ....utils import triangulation as tri
from ... import config_handler as confh
from ...utils import pose_utils
from ...utils.filters import SGFilter
from ...utils.keypoints_post_processing import extract_key_per_value
from .mmpose_framework_base import BaseMMPose


class MMPose2D(BaseMMPose):
    """
    The MMPose2D class is a method detector for 2D pose estimation using the MMPose
    framework.

    Its post_inference method computes 3D points via multi-view stereo triangulation.
    `components` is set per-instance from the TOML config (e.g. body-only models
    declare `components = ["body_joints"]`, wholebody models declare
    `["body_joints", "hand_joints", "face_landmarks"]`).
    """

    algorithm_type = "mmpose_2d"

    def get_per_component_keypoint_mapping(self, keypoints_indices):
        """
        Build per-component keypoint index/description maps based on the
        components this instance was configured to produce.

        - body_joints: when components include hand_joints or face_landmarks we
          treat the model as wholebody and union body + foot keypoints; otherwise
          we use the body block only.
        - hand_joints / face_landmarks: straightforward 1:1 mapping.
        """
        has_hand = "hand_joints" in self.components
        has_face = "face_landmarks" in self.components
        is_wholebody = has_hand or has_face

        indices = {}
        description = {}

        if "body_joints" in self.components:
            if is_wholebody:
                indices["body_joints"] = confh.flatten_list(
                    list(keypoints_indices.body.values()) + list(keypoints_indices.foot.values())
                )
                description["body_joints"] = confh.flatten_list(
                    extract_key_per_value(keypoints_indices.body) + extract_key_per_value(keypoints_indices.foot)
                )
            else:
                indices["body_joints"] = confh.flatten_list(list(keypoints_indices.body.values()))
                description["body_joints"] = confh.flatten_list(extract_key_per_value(keypoints_indices.body))

        if has_hand:
            indices["hand_joints"] = confh.flatten_list(list(keypoints_indices.hand.values()))
            description["hand_joints"] = confh.flatten_list(extract_key_per_value(keypoints_indices.hand))

        if has_face:
            indices["face_landmarks"] = confh.flatten_list(list(keypoints_indices.face.values()))
            description["face_landmarks"] = confh.flatten_list(extract_key_per_value(keypoints_indices.face))

        return indices, description

    def post_inference(self):
        """
        Post-inference processing for pose estimation components such as body_joints,
        hand_joints, and face_landmarks.

        See module docstring for the full description of steps and outputs.
        """
        for _component, result_folder in self.result_folders.items():
            prediction_file = os.path.join(result_folder, f"{self.algorithm_instance}.npz")
            prediction = np.load(prediction_file, allow_pickle=True)
            data_description = prediction["data_description"].item()
            results_2d = prediction["2d"]
            results_2d_bbox = prediction["bbox_2d"]

            iou_array = pose_utils.create_iou_all_pairs(results_2d_bbox)

            # Filter is applied.
            if self.filtered:
                logging.info("APPLYING filtering to 3d data...")
                results_2d_filtered = results_2d.copy()
                filter = SGFilter(self.filter_window_length, self.filter_polyorder)
                results_2d_filtered = filter.apply(results_2d_filtered)

            # If the filter is applied, 2d interpolation is performed on filtered data.
            if self.filtered:
                results_2d_interpolated = results_2d_filtered.copy()
            else:
                results_2d_interpolated = results_2d.copy()

            keypoint_conf_threshold = self.min_detection_confidence

            # A mask is created where the confidence score is below the threshold.
            low_confidence_mask = results_2d_interpolated[:, :, :, :, 2] < keypoint_conf_threshold

            # The mask is applied to set the first and second values of num_estimates to NaN
            # where the confidence is low.
            results_2d_interpolated[low_confidence_mask, 0:2] = np.nan
            results_2d_interpolated = pose_utils.interpolate_data(results_2d_interpolated, is_3d=False)

            data_description["2d_interpolated"] = data_description["2d"]
            data_description.update(
                {
                    "bbox_overlap": dict(
                        axis0=data_description["2d"]["axis0"],
                        axis1=data_description["2d"]["axis1"],
                        axis2=data_description["2d"]["axis2"],
                        axis3=[f"with_{subj}" for subj in self.subjects_descr],
                    )
                }
            )
            triangulation = self.detector_config.triangulation
            can_estimate_3d = triangulation.triangulate
            if can_estimate_3d and not self.calibration:
                logging.warning(
                    "WARNING - Calibration file is not valid. "
                    "Therefore, cannot compute 3d positions of the joints."
                    "Please see docs/wikis/wiki_calibration.md"
                )
                can_estimate_3d = False

            if not can_estimate_3d:
                if self.filtered:
                    data_description["2d_filtered"] = data_description["2d"]
                    # Results are saved.
                    results_dict = {
                        "2d": results_2d,
                        "2d_filtered": results_2d_filtered,
                        "2d_interpolated": results_2d_interpolated,
                        "bbox_2d": results_2d_bbox,
                        "bbox_overlap": iou_array,
                        "data_description": data_description,
                    }
                    np.savez_compressed(prediction_file, **results_dict)

                else:
                    # Results are saved.
                    results_dict = {
                        "2d": results_2d,
                        "2d_interpolated": results_2d_interpolated,
                        "bbox_2d": results_2d_bbox,
                        "bbox_overlap": iou_array,
                        "data_description": data_description,
                    }
                    np.savez_compressed(prediction_file, **results_dict)

            else:
                logging.info("COMPUTING 3d position of the joints...")

                cam1_name, cam2_name = triangulation.triangulation_cameras
                missing = [cam for cam in triangulation.triangulation_cameras if cam not in self.camera_names]
                if missing:
                    raise ValueError(
                        f"triangulation.triangulation_cameras {missing} are not among the resolved cameras "
                        f"{self.camera_names}."
                    )
                cam1_idx = self.camera_names.index(cam1_name)
                cam2_idx = self.camera_names.index(cam2_name)

                # Interpolated_2d results are used instead of original 2d.
                cam1_data, cam2_data = (
                    results_2d_interpolated[:, cam1_idx],
                    results_2d_interpolated[:, cam2_idx],
                )

                # Subject indices common in both camera views are found.
                subjects_cam1 = set(self.cam_sees_subjects[cam1_name])
                subjects_cam2 = set(self.cam_sees_subjects[cam2_name])
                common_subjects_idx = list(subjects_cam1 & subjects_cam2)

                person_data_list = []
                for subject_idx in common_subjects_idx:
                    person_cam1 = cam1_data[subject_idx]
                    person_cam2 = cam2_data[subject_idx]

                    # The x and y values are extracted.
                    xy_points_cam1 = person_cam1[:, :, :2].reshape(-1, 1, 2)
                    xy_points_cam2 = person_cam2[:, :, :2].reshape(-1, 1, 2)

                    # Confidence scores are extracted.
                    conf_cam1 = person_cam1[:, :, 2].reshape(-1, 1, 1)
                    conf_cam2 = person_cam2[:, :, 2].reshape(-1, 1, 1)

                    # Since interpolated data is used, some missing values might be present.
                    # A combined mask for NaN values in either camera's data is created.
                    nan_mask_cam1 = np.isnan(xy_points_cam1).any(axis=2)
                    nan_mask_cam2 = np.isnan(xy_points_cam2).any(axis=2)
                    combined_nan_mask = nan_mask_cam1 | nan_mask_cam2

                    # Rows with NaNs are filtered out for processing.
                    filtered_xy_points_cam1 = xy_points_cam1[~combined_nan_mask]
                    filtered_xy_points_cam2 = xy_points_cam2[~combined_nan_mask]

                    filtered_confidence_cam1 = conf_cam1[~combined_nan_mask]
                    filtered_confidence_cam2 = conf_cam2[~combined_nan_mask]

                    # Data is undistorted.
                    cam1_undistorted = np.squeeze(
                        tri.undistort_points_pinhole(
                            filtered_xy_points_cam1,
                            np.array(self.calibration[cam1_name]["intrinsic_matrix"]),
                            np.array(self.calibration[cam1_name]["distortions"]),
                        )
                    )
                    cam2_undistorted = np.squeeze(
                        tri.undistort_points_pinhole(
                            filtered_xy_points_cam2,
                            np.array(self.calibration[cam2_name]["intrinsic_matrix"]),
                            np.array(self.calibration[cam2_name]["distortions"]),
                        )
                    )

                    # Data is triangulated.
                    person_data_3d = tri.triangulate_stereo(
                        np.array(self.calibration[cam1_name]["projection_matrix"]),
                        np.array(self.calibration[cam2_name]["projection_matrix"]),
                        cam1_undistorted.T,
                        cam2_undistorted.T,
                    )

                    # Confidence score is added; cam1 and cam2 are combined, and the minimum confidence value is kept.
                    confidence_combined = np.minimum(filtered_confidence_cam1, filtered_confidence_cam2)
                    person_data_3d_with_conf = np.concatenate([person_data_3d.T, confidence_combined], axis=1)

                    # 3d array is reshaped.
                    # Output arrays filled with NaNs are created.
                    output_shape = (xy_points_cam1.shape[0], 4)
                    output_data_3d = np.full(output_shape, np.nan)

                    # The processed data is inserted back into the correct positions.
                    output_data_3d[~combined_nan_mask.reshape(-1)] = person_data_3d_with_conf
                    reshaped_3D_points = output_data_3d.reshape(person_cam1.shape[0], person_cam1.shape[1], 4)
                    person_data_list.append(reshaped_3D_points)

                # Any [0,0,0] prediction is checked.
                for person_data in person_data_list:
                    check.check_zeros(person_data)

                # Results are saved.
                descr_2d = data_description["2d"]
                common_subjects = [s for i, s in enumerate(self.subjects_descr) if i in common_subjects_idx]
                data_description.update(
                    {
                        "3d": dict(
                            axis0=common_subjects,
                            axis1=["3d"],
                            axis2=descr_2d["axis2"],
                            axis3=descr_2d["axis3"],
                            axis4=[
                                "coordinate_x",
                                "coordinate_y",
                                "coordinate_z",
                                "confidence_score",
                            ],
                        )
                    }
                )
                if self.filtered:
                    data_description["2d_filtered"] = data_description["2d"]
                    results_dict = {
                        "2d": results_2d,
                        "2d_filtered": results_2d_filtered,
                        "2d_interpolated": results_2d_interpolated,
                        "bbox_2d": results_2d_bbox,
                        "bbox_overlap": iou_array,
                        "3d": np.stack(person_data_list)[:, None],
                        "data_description": data_description,
                    }
                    np.savez_compressed(prediction_file, **results_dict)
                else:
                    results_dict = {
                        "2d": results_2d,
                        "2d_interpolated": results_2d_interpolated,
                        "bbox_2d": results_2d_bbox,
                        "bbox_overlap": iou_array,
                        "3d": np.stack(person_data_list)[:, None],
                        "data_description": data_description,
                    }
                    np.savez_compressed(prediction_file, **results_dict)
