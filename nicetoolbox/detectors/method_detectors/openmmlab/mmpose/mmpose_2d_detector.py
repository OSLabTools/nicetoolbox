import logging
import pickle
from typing import Any

import numpy as np

from nicetoolbox_core.data.array_schema import BBOX_2D_CONF, VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF_PER_LABEL
from nicetoolbox_core.data.loaded_array import NpzArray
from nicetoolbox_core.entrypoint import SubprocessResult
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from .....configs.schemas.detectors_instances_configs import MMPose2DAlgorithmConfig
from ....detector_inputs import NpzDetectorInput
from ....detector_outputs import DetectorOutput, NpzDetectorOutput
from ....utils.draw_2d import RenderContext, draw_skeleton
from ....utils.keypoints_post_processing import (
    extract_key_per_value,
    filter_keypoints,
    interpolate_keypoints,
    mask_low_confidence_keypoints,
    reproject_keypoints,
)
from ....utils.keypoints_triangualation import triangulate_keypoints
from ...base_method import BaseMethod


class MMPose2D(BaseMethod):
    algorithm_type = "mmpose_2d"

    # bounding boxes input
    inputs = [
        NpzDetectorInput("person_bounding_box", "person_bboxes", schema=BBOX_2D_CONF),
    ]

    # dynamic multi-components output (depends on what weight supports)
    def resolve_outputs(self):
        # go through each expected component, declare npz fields
        outs = []
        for component in self.detector_config.components:
            outs.append(NpzDetectorOutput(component, "2d", schema=VECTOR_2D_CONF_PER_LABEL))
            # filtering
            if self.detector_config.filter_keypoints.filtered:
                outs.append(NpzDetectorOutput(component, "2d_filtered", schema=VECTOR_2D_CONF_PER_LABEL))
            # low-confidence masking always runs
            outs.append(NpzDetectorOutput(component, "2d_masked", schema=VECTOR_2D_CONF_PER_LABEL))
            # interpolation
            if self.detector_config.interpolate_keypoints.interpolated:
                outs.append(NpzDetectorOutput(component, "2d_interpolated", schema=VECTOR_2D_CONF_PER_LABEL))
            # triangulation also requires calibration
            if self.detector_config.triangulate_keypoints.triangulate and self.data.calibration:
                outs.append(NpzDetectorOutput(component, "3d", schema=VECTOR_3D_CONF_PER_LABEL))
                outs.append(NpzDetectorOutput(component, "2d_reprojected_from_3d", schema=VECTOR_2D_CONF_PER_LABEL))
        return outs

    def _initialize_detector(self) -> MMPose2DAlgorithmConfig.RuntimeConfig:
        self.subjects = self.data.subjects_descr
        self.frame_names = self.data.frame_labels
        self.cameras = self.detector_config.camera_names
        self.video_start = self.data.video_start_frame_index

        # visualization context
        self.render_context = RenderContext(
            dataloader=ImagePathsByFrameIndexLoader(
                config=self.data.get_input_recipes(), expected_cameras=self.cameras
            ),
            fps=self.data.fps,
            video_start=self.video_start,
        )

        # need to send bboxes to native process
        bbox = self.loaded_inputs["person_bboxes"]
        base_runtime = super()._initialize_detector()
        return MMPose2DAlgorithmConfig.RuntimeConfig(
            **base_runtime.model_dump(),
            person_bbox_npz=str(bbox.file_path),
            person_bbox_npz_key=bbox.npz_key,
        )

    def post_inference(self, result_manifest: SubprocessResult) -> DetectorOutput:
        # load raw inference data
        with open(result_manifest.get_raw_output(), "rb") as raw_file:
            per_camera_outputs = pickle.load(raw_file)

        # get expected components keypoints indices
        comps = self._resolve_component_indices()

        # run for each component post-processing ritual
        out = DetectorOutput()
        for component, (indices, labels) in comps.items():
            pose_2d = self._parse_raw(per_camera_outputs, indices, labels)
            self._post_processs(out, component, pose_2d)
        return out

    def visualization(self, out: DetectorOutput) -> None:
        mapping = getattr(self.predictions_mapping.human_pose, self.detector_config.keypoint_mapping)
        kp_key = self.detector_config.visualize_keypoints_npz_key
        for component in self.detector_config.components:
            kp_2d = out.get(component, kp_key)
            connections = getattr(mapping.connections, component, [])
            draw_skeleton(kp_2d, connections, self.render_context, self.viz_folders[component])
            # TODO: draw 3d skeleton like motiobert?

    def _parse_raw(self, per_camera_outputs, indices, labels) -> NpzArray:
        n_subjects = len(self.subjects)
        n_cameras = len(self.cameras)
        n_frames = len(self.frame_names)
        n_keypoints = len(indices)

        pose_2d = np.full((n_subjects, n_cameras, n_frames, n_keypoints, 3), np.nan, dtype=float)
        for cam_idx, cam_name in enumerate(self.cameras):
            for frame_idx, persons in enumerate(per_camera_outputs[cam_name]):
                for person in persons:
                    # read raw keypoints
                    subj_idx = person["subject_idx"]
                    kpts = np.asarray(person["keypoints"])[indices]
                    # ! we prefer visibility rather than raw score (if available)
                    score = person["score"]
                    visibility = person.get("visibility")
                    conf = visibility if visibility else score
                    # save data to npz
                    pose_2d[subj_idx, cam_idx, frame_idx, :, :2] = kpts
                    pose_2d[subj_idx, cam_idx, frame_idx, :, 2] = np.asarray(conf)[indices]

        axes = VECTOR_2D_CONF_PER_LABEL.make_axes(self.subjects, self.cameras, self.frame_names, labels=labels)
        return NpzArray(pose_2d, axes)

    def _post_processs(self, out: DetectorOutput, component: str, pose_2d: NpzArray) -> None:
        # save raw original pose 2d and propagate post-processing passes
        out.add_array(component, "2d", pose_2d)

        # SGF filter (smooth xy along frame axis)
        if self.detector_config.filter_keypoints.filtered:
            logging.info(f"[{component}] SGF filtering 2d keypoints...")
            pose_2d_filtered = filter_keypoints(pose_2d, self.detector_config.filter_keypoints)
            out.add_array(component, "2d_filtered", pose_2d_filtered)
            pose_2d = pose_2d_filtered

        # low-conf mask
        logging.info(f"[{component}] Discarding low-confidence 2d keypoints...")
        min_conf = self.detector_config.min_detection_confidence
        pose_2d_masked = mask_low_confidence_keypoints(pose_2d, min_conf)
        out.add_array(component, "2d_masked", pose_2d_masked)
        pose_2d = pose_2d_masked

        # gap interpolation
        if self.detector_config.interpolate_keypoints.interpolated:
            logging.info(f"[{component}] Interpolating 2d keypoints...")
            pose_2d_interp = interpolate_keypoints(pose_2d, self.detector_config.interpolate_keypoints)
            out.add_array(component, "2d_interpolated", pose_2d_interp)
            pose_2d = pose_2d_interp

        # stereo triangulation (requires calibration)
        if self.detector_config.triangulate_keypoints.triangulate and self.data.calibration:
            logging.info(f"[{component}] Triangulating 2d keypoints to 3d...")
            pose_3d = triangulate_keypoints(
                pose_2d, self.detector_config.triangulate_keypoints, calibration=self.data.calibration
            )
            out.add_array(component, "3d", pose_3d)

            # project back to 2d for reprojection error
            pose_2d_reprojected = reproject_keypoints(
                pose_3d, calibration=self.data.calibration, camera_names=self.cameras
            )
            out.add_array(component, "2d_reprojected_from_3d", pose_2d_reprojected)

    @staticmethod
    def _flatten_list(input_list) -> list[Any]:
        if isinstance(input_list, str):
            return [input_list]
        if isinstance(input_list, int):
            return [input_list]
        if isinstance(input_list, list):
            output_list = []
            for item in input_list:
                output_list += MMPose2D._flatten_list(item)
            return output_list
        raise NotImplementedError

    def _resolve_component_indices(self) -> dict:
        # TODO: I'm transferring logic as is, but this need to be refactored
        components = self.detector_config.components
        is_wholebody = "hand_joints" in components or "face_landmarks" in components
        component_parts = {}
        if "body_joints" in components:
            component_parts["body_joints"] = ("body", "foot") if is_wholebody else ("body",)  # what?
        if "hand_joints" in components:
            component_parts["hand_joints"] = ("hand",)
        if "face_landmarks" in components:
            component_parts["face_landmarks"] = ("face",)

        mapping = getattr(self.predictions_mapping.human_pose, self.detector_config.keypoint_mapping)
        kpts_index = mapping.keypoints_index

        resolved = {}
        for component, part_names in component_parts.items():
            indices, labels = [], []
            for part_name in part_names:
                part_dict = getattr(kpts_index, part_name)  # {name: int} or {name: List[int]}
                indices.extend(self._flatten_list(list(part_dict.values())))
                labels.extend(extract_key_per_value(part_dict))
            resolved[component] = (indices, labels)
        return resolved
