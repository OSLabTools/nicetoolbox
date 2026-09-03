"""
SPIGA method detector class.
"""

import logging
import os
import pickle

import numpy as np

from nicetoolbox_core.data.array_schema import (
    BBOX_2D_CONF,
    VECTOR_2D_CONF,
    VECTOR_2D_CONF_PER_LABEL,
    VECTOR_2D_PER_LABEL,
    VECTOR_3D_CONF,
    VECTOR_3D_CONF_PER_LABEL,
    ArraySchema,
)
from nicetoolbox_core.data.loaded_array import NpzArray
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ....configs.schemas.detectors_instances_configs import SpigaConfig
from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ...utils.draw_2d import RenderContext, draw_direction_vector, draw_keypoints
from ...utils.keypoints_post_processing import (
    extract_key_per_value,
    filter_keypoints,
    interpolate_keypoints,
    reproject_keypoints,
)
from ...utils.keypoints_triangualation import triangulate_keypoints
from ..base_method import BaseMethod

RAW_INFERENCE_PICKLE_NAME = "spiga_inference_raw.pkl"

# Native spiga axes for head position rotation
# Rotation is degree euler angles (rotation order is yaw * pitch * roll)
SPIGA_RAW_HEAD_ROT = ArraySchema(labels_columns=("yaw", "pitch", "roll"))

# Landmark the head axes are drawn from: the WFLW nose tip, which is where the mean face model
# puts its origin ([0, 0, 0] in mean_face_3D_98.txt), so the rotation is defined about this point.
HEAD_ROTATION_ORIGIN_INDEX = 54


class Spiga(BaseMethod):
    components = ["head_orientation", "face_landmarks"]
    algorithm_type = "spiga"

    # spiga needs face bounding boxes
    inputs = [
        NpzDetectorInput("face_bounding_box", "face_bbox_2d", schema=BBOX_2D_CONF),
    ]

    # outputs two components - head_orientation and face_landmarks
    def resolve_outputs(self) -> list[NpzDetectorOutput]:
        outputs = [
            NpzDetectorOutput("head_orientation", "head_rotation_raw", schema=SPIGA_RAW_HEAD_ROT),
            NpzDetectorOutput("head_orientation", "head_direction_3d_camera_space", schema=VECTOR_3D_CONF),
            NpzDetectorOutput("head_orientation", "head_direction_2d", schema=VECTOR_2D_CONF),
            NpzDetectorOutput("head_orientation", "head_origin_2d", schema=VECTOR_2D_CONF),
            NpzDetectorOutput("face_landmarks", "2d", schema=VECTOR_2D_CONF_PER_LABEL),
        ]
        # world space head direction needs each camera's rotation matrix
        if self.data.calibration:
            outputs.append(NpzDetectorOutput("head_orientation", "head_direction_3d", schema=VECTOR_3D_CONF))
        # optional post processing + triangulation
        if self.detector_config.filter_keypoints.filtered:
            outputs.append(NpzDetectorOutput("face_landmarks", "2d_filtered", schema=VECTOR_2D_CONF_PER_LABEL))
        if self.detector_config.interpolate_keypoints.interpolated:
            outputs.append(NpzDetectorOutput("face_landmarks", "2d_interpolated", schema=VECTOR_2D_CONF_PER_LABEL))
        # triangulation also requires calibration
        if self.detector_config.triangulate_keypoints.triangulate and self.data.calibration:
            outputs.append(NpzDetectorOutput("face_landmarks", "3d", schema=VECTOR_3D_CONF_PER_LABEL))
            outputs.append(NpzDetectorOutput("face_landmarks", "2d_reprojected_from_3d", schema=VECTOR_2D_PER_LABEL))
            outputs.append(NpzDetectorOutput("head_orientation", "head_origin_3d", schema=VECTOR_3D_CONF))
        return outputs

    def _initialize_detector(self) -> SpigaConfig.RuntimeConfig:
        """
        Initializes the Spiga class with extra configuration settings.
        """
        # Store convenience references for this class
        self.subjects_descr = self.data.subjects_descr
        self.frame_names = self.data.frame_labels
        self.camera_names = self.detector_config.camera_names
        self.cam_sees_subjects = self.data.cam_sees_subjects
        self.video_start = self.data.video_start_frame_index

        # Used by visualization() to walk the source frames; inference has its own loader.
        self.render_context = RenderContext(
            dataloader=ImagePathsByFrameIndexLoader(
                config=self.data.get_input_recipes(), expected_cameras=self.camera_names
            ),
            cam_sees_subjects=self.cam_sees_subjects,
            fps=self.data.fps,
            video_start=self.video_start,
        )

        # Labels for the face_landmarks axis3, in the order SPIGA emits them.
        # TODO: support the 68-point iBUG convention (300wpublic, 300wprivate, merlrav weights).
        keypoint_mapping = getattr(self.predictions_mapping.human_pose, self.detector_config.keypoint_mapping)
        self.keypoints_indices = keypoint_mapping.keypoints_index
        # TODO: this mmpose extract_key_per_value smells
        self.face_landmarks_description = extract_key_per_value(self.keypoints_indices.face)
        # TODO: universal system to better pass npz inside method detectors venv?
        face_bbox = self.loaded_inputs["face_bbox_2d"]
        face_bbox_path = face_bbox.file_path
        face_bbox_key = face_bbox.npz_key

        # Check that this detectors cameras are subset of the upstream detector
        face_bbox_cameras = face_bbox.axes.cameras
        missing_cameras = [name for name in self.camera_names if name not in face_bbox_cameras]
        if missing_cameras:
            raise ValueError(
                f"Detector '{self.algorithm_instance}': camera_names {missing_cameras} are not provided by "
                f"input '{face_bbox.name}' ({face_bbox.algorithm}), which covers {face_bbox_cameras}. "
                "Align camera_names across both detectors in the config."
            )

        # Return extended runtime with Spiga-specific fields
        base_runtime = super()._initialize_detector()
        return SpigaConfig.RuntimeConfig(
            **base_runtime.model_dump(),
            face_bbox_npz=str(face_bbox_path),
            face_bbox_npz_key=face_bbox_key,
        )

    def post_inference(self) -> DetectorOutput:
        out = DetectorOutput()

        # Parse output from the detector
        output_folder = self.out_folders[self.components[0]]
        raw_path = os.path.join(output_folder, RAW_INFERENCE_PICKLE_NAME)
        head_rot, landmarks_2d = self._parse_raw(raw_path)

        # keypoints post processing
        out.add_array("face_landmarks", "2d", landmarks_2d)
        # do we need to do keypoints filtering?
        filter_config = self.detector_config.filter_keypoints
        if filter_config.filtered:
            logging.info("Applying filtering for 2d facial keypoints...")
            landmarks_2d = filter_keypoints(landmarks_2d, filter_config)
            out.add_array("face_landmarks", "2d_filtered", landmarks_2d)
        # do we need to fill short detection dropouts?
        interpolation_config = self.detector_config.interpolate_keypoints
        if interpolation_config.interpolated:
            logging.info("Interpolating 2d facial keypoints...")
            landmarks_2d = interpolate_keypoints(landmarks_2d, interpolation_config)
            out.add_array("face_landmarks", "2d_interpolated", landmarks_2d)
        # do we need to lift the keypoints to 3d?
        triangulation_config = self.detector_config.triangulate_keypoints
        if triangulation_config.triangulate and self.data.calibration:
            logging.info("Triangulating 2d facial keypoints to 3d...")
            landmarks_3d = triangulate_keypoints(
                landmarks_2d,
                triangulation_config,
                calibration=self.data.calibration,
                cam_sees_subjects=self.cam_sees_subjects,
            )
            out.add_array("face_landmarks", "3d", landmarks_3d)
            # also add back projection for debugging/evaluation
            landmarks_2d_backprojected = reproject_keypoints(
                landmarks_3d,
                calibration=self.data.calibration,
                camera_names=self.camera_names,
            )
            out.add_array("face_landmarks", "2d_reprojected_from_3d", landmarks_2d_backprojected)
            # the head rotation's origin in world space, so the direction becomes a real 3d ray
            out.add_array("head_orientation", "head_origin_3d", self._head_origin_3d(landmarks_3d))

        # where that direction starts: the nose tip, which is the point the rotation is about.
        # TODO: it's not "real" head origin, but close enoug. Check the inference script for full info.
        # landmarks_2d is whatever the post-processing chain produced last.
        out.add_array("head_orientation", "head_origin_2d", self._head_origin_2d(landmarks_2d))

        # head rotation post-processing
        # TODO: we save only direction for now, head rotationn stays in raw unprocessed way
        # for detecting something like head noding, we need to properly convert to toolbox world space
        out.add_array("head_orientation", "head_rotation_raw", head_rot)
        # take a +X vector and apply rotation of the head (where subject looks at)
        # first in the camera space
        head_dir_cam = self._head_direction_camera_space(head_rot)
        out.add_array("head_orientation", "head_direction_3d_camera_space", head_dir_cam)
        # flatten the camera space vector onto each image plane, for 2d overlays
        out.add_array("head_orientation", "head_direction_2d", self._project_to_2d(head_dir_cam))
        # next transfer it to the world space, if we know how the cameras are oriented
        if self.data.calibration:
            head_dir_world = self._world_lift(head_dir_cam)
            out.add_array("head_orientation", "head_direction_3d", head_dir_world)

        return out

    def visualization(self, out: DetectorOutput) -> None:
        # visualize 2d keypoints
        kp_2d_key = self.detector_config.visualize_keypoints_npz_key
        kp_2d = out.get("face_landmarks", kp_2d_key)
        draw_keypoints(kp_2d, self.render_context, self.viz_folders["face_landmarks"])

        # visualize head direction
        head_dir_2d = out.get("head_orientation", "head_direction_2d")
        head_origin_2d = out.get("head_orientation", "head_origin_2d")
        draw_direction_vector(head_dir_2d, head_origin_2d, self.render_context, self.viz_folders["head_orientation"])

    def _head_direction_camera_space(self, head_rotation: NpzArray) -> NpzArray:
        # Turn SPIGA's euler angles into the face-forward unit vector, in the camera's frame.
        # The mean face model has +X pointing out of the face, so forward is R @ [1, 0, 0]
        angles = head_rotation.data
        directions = np.full(angles.shape[:3] + (4,), np.nan, dtype=float)
        directions[..., :3] = self._euler_to_rotation_matrix(angles) @ np.array([1.0, 0.0, 0.0])
        directions[..., 3] = 1.0  # ! SPIGA doesn't have conf, hardcode to 1
        axes = VECTOR_3D_CONF.make_axes(self.subjects_descr, self.camera_names, self.frame_names)
        return NpzArray(directions, axes)

    @staticmethod
    def _euler_to_rotation_matrix(head_rotation):
        # Build SPIGA's rotation matrix from its raw euler angles.
        # Ported from spiga.demo.visualize.layouts.plot_headpose. The +90/-90 offsets and the sign
        # flips are SPIGA's own coordinate change, so the stored angles cannot be fed to a generic
        # euler routine without them. Rotation order is Ry @ Rp @ Rr (intrinsic Y-Z-X).
        # Broadcasts over leading axes, so a whole (subjects, cameras, frames, 3) array of angles
        # converts in one call.
        euler = np.stack(
            [-(head_rotation[..., 0] - 90), -head_rotation[..., 1], -(head_rotation[..., 2] + 90)],
            axis=-1,
        )
        rad = euler * (np.pi / 180.0)
        cy, sy = np.cos(rad[..., 0]), np.sin(rad[..., 0])
        cp, sp = np.cos(rad[..., 1]), np.sin(rad[..., 1])
        cr, sr = np.cos(rad[..., 2]), np.sin(rad[..., 2])
        zero, one = np.zeros_like(cy), np.ones_like(cy)
        # Rows stacked on axis -2, so each 3x3 sits in the trailing two axes.
        rotation_yaw = np.stack(
            [np.stack([cy, zero, sy], -1), np.stack([zero, one, zero], -1), np.stack([-sy, zero, cy], -1)], -2
        )
        rotation_pitch = np.stack(
            [np.stack([cp, -sp, zero], -1), np.stack([sp, cp, zero], -1), np.stack([zero, zero, one], -1)], -2
        )
        rotation_roll = np.stack(
            [np.stack([one, zero, zero], -1), np.stack([zero, cr, -sr], -1), np.stack([zero, sr, cr], -1)], -2
        )
        return rotation_yaw @ rotation_pitch @ rotation_roll

    def _head_origin_3d(self, keypoints_3d: NpzArray) -> NpzArray:
        # Same nose tip as _head_origin_2d, but triangulated: a real world-space point in the
        # calibration frame, so pairing it with head_direction_3d gives a metric ray. Keeps the
        # triangulated array's pseudo-camera axis ("3d"), since the point is view-independent.
        origin = keypoints_3d.data[:, :, :, HEAD_ROTATION_ORIGIN_INDEX]
        axes = VECTOR_3D_CONF.make_axes(self.subjects_descr, keypoints_3d.axes.cameras, self.frame_names)
        return NpzArray(origin, axes)

    def _head_origin_2d(self, keypoints_2d: NpzArray) -> NpzArray:
        # The nose tip landmark, pulled out of the dense face landmarks. This is the point the
        # head rotation is defined about, so it anchors the direction vectors.
        origin = keypoints_2d.data[:, :, :, HEAD_ROTATION_ORIGIN_INDEX]
        axes = VECTOR_2D_CONF.make_axes(self.subjects_descr, self.camera_names, self.frame_names)
        return NpzArray(origin, axes)

    def _project_to_2d(self, direction_cam: NpzArray) -> NpzArray:
        # Flatten the camera space direction onto the image plane. The vector is already in the
        # OpenCV camera frame, so x and y are image-plane directions in the right sense and z is
        # simply dropped - an orthographic projection. The result is a unitless (dx, dy) heading,
        # not a pixel length, so a renderer picks its own arrow scale.
        data = direction_cam.data
        projected = np.stack([data[..., 0], data[..., 1], data[..., 3]], axis=-1)
        axes = VECTOR_2D_CONF.make_axes(self.subjects_descr, self.camera_names, self.frame_names)
        return NpzArray(projected, axes)

    def _world_lift(self, direction_cam: NpzArray) -> NpzArray:
        # Rotate each camera's local head direction into the shared world frame. A direction has
        # no position, so only the camera rotation applies: direction_world = inv(R) @ direction.
        # The confidence channel (axis3 index 3) is carried through unchanged.
        data = direction_cam.data.copy()
        for cam_idx, cam_name in enumerate(self.camera_names):
            cam_rotation = np.asarray(self.data.calibration[cam_name]["rotation_matrix"], dtype=float)
            r_inv = np.linalg.inv(cam_rotation)

            vectors = data[:, cam_idx, :, :3]  # (subjects, frames, 3)
            world = vectors @ r_inv.T  # apply inv(R) to each xyz vector
            norms = np.linalg.norm(world, axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                world = world / norms
            data[:, cam_idx, :, :3] = world

        return NpzArray(data, direction_cam.axes)

    def _parse_raw(self, raw_path) -> tuple[NpzArray, NpzArray]:
        # read the raw pack written by the inference subprocess
        with open(raw_path, "rb") as raw_file:
            per_frame_outputs = pickle.load(raw_file)

        # validate number of frames
        n_frames = len(self.frame_names)
        if len(per_frame_outputs) != n_frames:
            raise ValueError(
                f"Raw pack holds {len(per_frame_outputs)} frames but the subsequence has {n_frames}. "
                "The pack was produced for different data - re-run inference (skip_inference)."
            )

        # NaN marks "no measurement": SPIGA itself never returns NaN angles or coordinates.
        head_pose_axes = SPIGA_RAW_HEAD_ROT.make_axes(self.subjects_descr, self.camera_names, self.frame_names)
        head_rotation = np.full(head_pose_axes.shape, np.nan, dtype=float)

        landmarks_axes = VECTOR_2D_CONF_PER_LABEL.make_axes(
            self.subjects_descr, self.camera_names, self.frame_names, labels=self.face_landmarks_description
        )
        landmarks_2d = np.full(landmarks_axes.shape, np.nan, dtype=float)

        for frame_idx, detections_per_camera in enumerate(per_frame_outputs):
            for camera_name, detections in detections_per_camera.items():
                cam_idx = self.camera_names.index(camera_name)
                for detection in detections:
                    subj_idx = self.subjects_descr.index(detection["subject"])
                    head_rotation[subj_idx, cam_idx, frame_idx] = detection["head_rotation"]
                    landmarks_2d[subj_idx, cam_idx, frame_idx, :, :2] = detection["landmarks_2d"]
                    landmarks_2d[subj_idx, cam_idx, frame_idx, :, 2] = 1.0  # ! SPIGA doesn't have conf, hardcode to 1

        return NpzArray(head_rotation, head_pose_axes), NpzArray(landmarks_2d, landmarks_axes)
