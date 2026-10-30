import logging
import pickle

import numpy as np

from nicetoolbox_core.data.array_schema import BBOX_2D_CONF, VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF_PER_LABEL
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes
from nicetoolbox_core.entrypoint import SubprocessResult
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ....configs.schemas.detectors_instances_configs import Sam3dBodyAlgorithmConfig
from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput, NpzStaticDetectorOutput
from ...utils.draw_2d import RenderContext, draw_mesh_wireframe, draw_skeleton
from ...utils.keypoints_post_processing import (
    filter_keypoints,
    interpolate_keypoints,
    reproject_keypoints,
    reproject_keypoints_mat,
)
from ...utils.keypoints_triangualation import triangulate_keypoints
from ..base_method import BaseMethod

# TODO: we can use MHR LOD 6 level to move 18k verts to 595
# MHR mesh (LOD 1) used by SAM 3D Body: 18439 vertices, 36874 faces
MHR_NUM_VERTICES = 18439


class Sam3dBody(BaseMethod):
    algorithm_type = "sam_3d_body"
    components = ["body_joints", "hand_joints", "body_joints_local", "hand_joints_local"]

    # inputs
    inputs = [
        NpzDetectorInput("person_bounding_box", "person_bboxes", schema=BBOX_2D_CONF),
    ]

    # outputs
    def resolve_outputs(self):
        outputs = []
        for component in ("body_joints", "hand_joints"):
            # === raw sam3d 2d ===
            outputs.append(NpzDetectorOutput(component, "2d", schema=VECTOR_2D_CONF_PER_LABEL))
            # 2d filtering
            if self.detector_config.filter_keypoints.filtered:
                outputs.append(NpzDetectorOutput(component, "2d_filtered", schema=VECTOR_2D_CONF_PER_LABEL))
            # 2d interpolation
            if self.detector_config.interpolate_keypoints.interpolated:
                outputs.append(NpzDetectorOutput(component, "2d_interpolated", schema=VECTOR_2D_CONF_PER_LABEL))
            # sam3d 2d => triangulated 3d
            if self.detector_config.triangulate_keypoints.triangulate and self.data.calibration:
                outputs.append(NpzDetectorOutput(component, "3d", schema=VECTOR_3D_CONF_PER_LABEL))
                # triangulated 3d => back projected 2d
                outputs.append(NpzDetectorOutput(component, "2d_reprojected_from_3d", schema=VECTOR_2D_CONF_PER_LABEL))
            # === raw sam3d 3d (camera-space) ====
            local_comp = f"{component}_local"
            outputs.append(NpzDetectorOutput(local_comp, "3d", schema=VECTOR_3D_CONF_PER_LABEL))
            # 3d filtering
            if self.detector_config.filter_keypoints.filtered:
                outputs.append(NpzDetectorOutput(local_comp, "3d_filtered", schema=VECTOR_3D_CONF_PER_LABEL))
            # 3d interpolation
            if self.detector_config.interpolate_keypoints.interpolated:
                outputs.append(NpzDetectorOutput(local_comp, "3d_interpolated", schema=VECTOR_3D_CONF_PER_LABEL))
            # final 3d => back projected 2d
            outputs.append(NpzDetectorOutput(local_comp, "2d_reprojected_from_3d", schema=VECTOR_2D_CONF_PER_LABEL))
            # final 3d => world 3d per camera, with calibration extrinsics (no triangulation)
            if self.data.calibration:
                outputs.append(NpzDetectorOutput(local_comp, "3d_world", schema=VECTOR_3D_CONF_PER_LABEL))

        # full 3d mesh (optional)
        if self.detector_config.save_vertices:
            # vertices in camera coordinates (per camera)
            outputs.append(NpzDetectorOutput("body_mesh_local", "vertices"))
            # mesh triangles (vertex indices), shared by all subjects, cameras and frames
            outputs.append(NpzStaticDetectorOutput("body_mesh_local", "faces"))
        return outputs

    def _initialize_detector(self) -> Sam3dBodyAlgorithmConfig.RuntimeConfig:
        if self.detector_config.save_vertices:  # result folder for the optional mesh component
            self.components = [*self.components, "body_mesh_local"]
            # base init computed the folders before this component existed
            self.result_folders = self.compute_result_folders()
            self.viz_folders = self.compute_viz_folders(self.visualize)
        self.subjects = self.data.subjects_descr
        self.frame_names = self.data.frame_labels
        self.cameras = self.detector_config.camera_names
        self.mapping = getattr(self.predictions_mapping.human_pose, self.detector_config.keypoint_mapping)

        # visualization context
        self.render_context = RenderContext(
            dataloader=ImagePathsByFrameIndexLoader(
                config=self.data.get_input_recipes(), expected_cameras=self.cameras
            ),
            fps=self.data.fps,
            video_start=self.data.video_start_frame_index,
            save_frames=self.save_frames,
        )

        # need to send bboxes to native process
        bbox = self.loaded_inputs["person_bboxes"]
        base_runtime = super()._initialize_detector()
        return Sam3dBodyAlgorithmConfig.RuntimeConfig(
            **base_runtime.model_dump(),
            person_bbox_npz=str(bbox.file_path),
            person_bbox_npz_key=bbox.npz_key,
        )

    def post_inference(self, result_manifest: SubprocessResult) -> DetectorOutput:
        # load raw inference data
        with open(result_manifest.get_raw_output(), "rb") as raw_file:
            raw = pickle.load(raw_file)
        frames = raw["frames"]

        out = DetectorOutput()
        projection_matrices = self._resolve_projection_matrices()
        # parse and save 2d and 3d keypoints for body and hands
        for component, (indices, labels) in self._resolve_component_indices().items():
            kp_2d, kp_3d = self._parse_keypoints(frames, indices, labels)
            self._post_process_2d(out, component, kp_2d)
            self._post_process_3d(out, component, kp_3d, projection_matrices)

        # TODO: save raw mhr body parameter for future decoding?
        # TODO: no temporal smoothing for body mesh? need decoder?
        if self.detector_config.save_vertices:
            out.add_array("body_mesh_local", "vertices", self._parse_mesh(frames))
            out.add_static("body_mesh_local", "faces", raw["faces"])
        return out

    def visualization(self, out: DetectorOutput) -> None:
        # 2d skeleton overlays for body and hands, from the configured post-processing stage
        kp_key = self.detector_config.visualize_keypoints_npz_key
        for component in ("body_joints", "hand_joints"):
            # raw 2d keypoints
            kp_2d = out.get(component, kp_key)
            connections = getattr(self.mapping.connections, component)
            draw_skeleton(kp_2d, connections, self.render_context, self.viz_folders[component])
            # camera space 3d keypoints
            local_component = f"{component}_local"
            kp_2d_local = out.get(local_component, "2d_reprojected_from_3d")
            draw_skeleton(kp_2d_local, connections, self.render_context, self.viz_folders[local_component])

        # mesh wireframe overlay
        if self.detector_config.save_vertices:
            vertices = out.get("body_mesh_local", "vertices")
            faces = out.get_static("body_mesh_local", "faces")
            viz_folder = self.viz_folders["body_mesh_local"]
            projection_matrices = self._resolve_projection_matrices()
            draw_mesh_wireframe(vertices, faces, projection_matrices, self.render_context, viz_folder)

    def _post_process_2d(self, out: DetectorOutput, component: str, kp_2d: NpzArray) -> None:
        """Same chain as mmpose: raw 2d -> filter -> interpolate -> triangulate to world 3d -> reproject."""
        out.add_array(component, "2d", kp_2d)
        # no low-confidence masking: SAM 3D Body has no keypoint confidence (hardcoded to one)

        # SGF filter (smooth xy along frame axis)
        filter_config = self.detector_config.filter_keypoints
        if filter_config.filtered:
            logging.info(f"[{component}] SGF filtering 2d keypoints...")
            kp_2d = filter_keypoints(kp_2d, filter_config)
            out.add_array(component, "2d_filtered", kp_2d)

        # gap interpolation
        interpolation_config = self.detector_config.interpolate_keypoints
        if interpolation_config.interpolated:
            logging.info(f"[{component}] Interpolating 2d keypoints...")
            kp_2d = interpolate_keypoints(kp_2d, interpolation_config)
            out.add_array(component, "2d_interpolated", kp_2d)

        # stereo triangulation (requires calibration)
        triangulation_config = self.detector_config.triangulate_keypoints
        if triangulation_config.triangulate and self.data.calibration:
            logging.info(f"[{component}] Triangulating 2d keypoints to 3d...")
            kp_3d = triangulate_keypoints(kp_2d, triangulation_config, calibration=self.data.calibration)
            out.add_array(component, "3d", kp_3d)

            # project back to 2d for reprojection error
            kp_2d_reprojected = reproject_keypoints(kp_3d, calibration=self.data.calibration, camera_names=self.cameras)
            out.add_array(component, "2d_reprojected_from_3d", kp_2d_reprojected)

    def _post_process_3d(
        self, out: DetectorOutput, component: str, kp_3d: NpzArray, projection_matrices: dict[str, np.ndarray]
    ) -> None:
        """Camera space 3d: raw -> filter -> interpolate (same settings as 2d) -> project to 2d."""
        local_component = f"{component}_local"
        out.add_array(local_component, "3d", kp_3d)

        # SGF filter (smooth xyz along frame axis)
        filter_config = self.detector_config.filter_keypoints
        if filter_config.filtered:
            logging.info(f"[{local_component}] SGF filtering 3d keypoints...")
            kp_3d = filter_keypoints(kp_3d, filter_config, is_3d=True)
            out.add_array(local_component, "3d_filtered", kp_3d)

        # gap interpolation
        interpolation_config = self.detector_config.interpolate_keypoints
        if interpolation_config.interpolated:
            logging.info(f"[{local_component}] Interpolating 3d keypoints...")
            kp_3d = interpolate_keypoints(kp_3d, interpolation_config, is_3d=True)
            out.add_array(local_component, "3d_interpolated", kp_3d)

        # project the final camera space 3d back into each camera image
        kp_2d_reprojected = reproject_keypoints_mat(kp_3d, projection_matrices)
        out.add_array(local_component, "2d_reprojected_from_3d", kp_2d_reprojected)

        # move it to world space using extrinsics
        if self.data.calibration:
            logging.info(f"[{local_component}] Moving 3d keypoints to world coordinates...")
            kp_3d = self._camera_to_world(kp_3d)
            out.add_array(local_component, "3d_world", kp_3d)

    def _parse_keypoints(self, raw_frames: list, indices: list[int], labels: list[str]) -> tuple[NpzArray, NpzArray]:
        """2d (pixels) and 3d (camera coordinates, metres) keypoints from raw person results."""
        axes_2d = VECTOR_2D_CONF_PER_LABEL.make_axes(self.subjects, self.cameras, self.frame_names, labels=labels)
        axes_3d = VECTOR_3D_CONF_PER_LABEL.make_axes(self.subjects, self.cameras, self.frame_names, labels=labels)
        kp_2d = np.full(axes_2d.shape, np.nan, dtype=float)
        kp_3d = np.full(axes_3d.shape, np.nan, dtype=float)

        for frame_idx, persons_per_camera in enumerate(raw_frames):
            for cam_name, persons in persons_per_camera.items():
                cam_idx = self.cameras.index(cam_name)
                for person in persons:
                    s = person["subject_idx"]
                    kp_2d[s, cam_idx, frame_idx, :, :2] = person["keypoints_2d"][indices]
                    kp_3d[s, cam_idx, frame_idx, :, :3] = person["keypoints_3d"][indices]
                    # TODO: save cam_t to npz?
                    kp_3d[s, cam_idx, frame_idx, :, :3] += person["cam_t"]
                    # SAM 3D Body has no keypoint confidence, hardcode it to one
                    kp_2d[s, cam_idx, frame_idx, :, 2] = 1.0
                    kp_3d[s, cam_idx, frame_idx, :, 3] = 1.0

        return NpzArray(kp_2d, axes_2d), NpzArray(kp_3d, axes_3d)

    def _parse_mesh(self, raw_frames: list) -> NpzArray:
        """Mesh vertices in the camera coordinates (metres), per subject, camera and frame."""
        shape = (len(self.subjects), len(self.cameras), len(self.frame_names))
        # float32: vertices are ~18k x 3 per subject, camera and frame
        vertices = np.full((*shape, MHR_NUM_VERTICES, 3), np.nan, dtype=np.float32)

        for frame_idx, persons_per_camera in enumerate(raw_frames):
            for cam_name, persons in persons_per_camera.items():
                cam_idx = self.cameras.index(cam_name)
                for person in persons:
                    subj_idx = person["subject_idx"]
                    # body-centred vertices + camera translation of this body
                    vertices[subj_idx, cam_idx, frame_idx] = person["vertices"] + person["cam_t"]

        xyz = ["coordinate_x", "coordinate_y", "coordinate_z"]
        vertex_labels = [f"mesh_vertex_{i}" for i in range(MHR_NUM_VERTICES)]
        vertices_axes = NpzArrayAxes(self.subjects, list(self.cameras), self.frame_names, vertex_labels, xyz)

        # TODO: better way to mark vericles to be ignored for csv export?
        # topology is too big for csv export, mark it to ignore
        ignore_csv = {"csv_skip": True}
        return NpzArray(vertices, vertices_axes, ignore_csv)

    def _camera_to_world(self, kp_3d: NpzArray) -> NpzArray:
        """Camera space 3d to world 3d, per camera: X_w = R^T (X_c - t), for world->camera extrinsics R, t."""
        data = kp_3d.data.copy()
        for cam_idx, cam_name in enumerate(kp_3d.axes.cameras):
            cam_calib = self.data.calibration[cam_name]
            rotation = np.asarray(cam_calib["rotation_matrix"], dtype=float)
            translation = np.asarray(cam_calib["translation"], dtype=float).reshape(3)
            # row vectors: R^T (x - t) == (x - t) @ R
            data[:, cam_idx, ..., :3] = (data[:, cam_idx, ..., :3] - translation) @ rotation
        # confidence is kept as is
        return NpzArray(data, kp_3d.axes)

    def _resolve_projection_matrices(self) -> dict[str, np.ndarray]:
        """(3, 4) projection matrix per camera for camera space points: P = [K | 0] (identity extrinsics)."""
        calibration = self.data.calibration or {}
        projection_matrices = {}
        for cam_name in self.cameras:
            cam_calib = calibration.get(cam_name) or {}
            if "intrinsic_matrix" in cam_calib:
                intrinsic_matrix = np.asarray(cam_calib["intrinsic_matrix"], dtype=float).reshape(3, 3)
            else:
                # direct copy of how sam3d handle missing calibration
                width, height = self.data.video_resolutions[cam_name]
                focal = (height**2 + width**2) ** 0.5
                intrinsic_matrix = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]])
            projection_matrices[cam_name] = np.hstack([intrinsic_matrix, np.zeros((3, 1))])
        return projection_matrices

    # TODO: same as mmpose pred mapping problems, please refactor
    def _resolve_component_indices(self) -> dict[str, tuple[list[int], list[str]]]:
        component_parts = {
            "body_joints": ("body", "foot", "extra"),
            "hand_joints": ("hand",),
        }
        resolved = {}
        for component, part_names in component_parts.items():
            indices, labels = [], []
            for part_name in part_names:
                part_dict: dict[str, int] = getattr(self.mapping.keypoints_index, part_name)
                indices.extend(part_dict.values())
                labels.extend(part_dict.keys())
            resolved[component] = (indices, labels)
        return resolved
