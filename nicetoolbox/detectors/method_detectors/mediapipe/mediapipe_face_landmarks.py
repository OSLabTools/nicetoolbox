"""
MediaPipe Face Landmarker method detector class.
"""

import logging
import os
import pickle

import numpy as np

from nicetoolbox_core.data.array_schema import BBOX_2D_CONF, FLOAT, VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF_PER_LABEL
from nicetoolbox_core.data.loaded_array import NpzArray
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ....configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ...utils.draw_2d import RenderContext, draw_keypoints
from ...utils.keypoints_post_processing import filter_keypoints, interpolate_keypoints
from ..base_method import BaseMethod

RAW_INFERENCE_PICKLE_NAME = "mediapipe_face_landmarks_inference_raw.pkl"


class MediapipeFaceLandmarks(BaseMethod):
    """
    MediapipeFaceLandmarks is a method detector for facial landmarks and blendshapes using MediaPipe.
    """

    components = ["face_blendshapes", "face_landmarks_local"]
    algorithm_type = "mediapipe_face_landmarks"

    def resolve_outputs(self) -> list[NpzDetectorOutput]:
        outputs = [
            NpzDetectorOutput("face_blendshapes", "blendshapes", schema=FLOAT),
            NpzDetectorOutput("face_landmarks_local", "2d", schema=VECTOR_2D_CONF_PER_LABEL),
            NpzDetectorOutput("face_landmarks_local", "3d", schema=VECTOR_3D_CONF_PER_LABEL),
        ]

        # optional post processing for 2D facial landmarks
        if self.detector_config.filter_keypoints.filtered:
            outputs.append(NpzDetectorOutput("face_landmarks_local", "2d_filtered", schema=VECTOR_2D_CONF_PER_LABEL))
        if self.detector_config.interpolate_keypoints.interpolated:
            outputs.append(
                NpzDetectorOutput("face_landmarks_local", "2d_interpolated", schema=VECTOR_2D_CONF_PER_LABEL)
            )
        return outputs

    def _initialize_detector(self) -> MethodDetectorRuntime:
        self.subjects = self.data.subjects_descr
        self.frame_names = self.data.frame_labels
        self.cameras = self.detector_config.camera_names
        self.cam_sees_subjects = self.data.cam_sees_subjects
        self.video_start = self.data.video_start_frame_index

        # Used by visualization() to walk the source frames
        self.render_context = RenderContext(
            dataloader=ImagePathsByFrameIndexLoader(
                config=self.data.get_input_recipes(), expected_cameras=self.cameras
            ),
            cam_sees_subjects=self.cam_sees_subjects,
            fps=self.data.fps,
            video_start=self.video_start,
        )

        keypoint_mapping = getattr(self.predictions_mapping.human_pose, self.detector_config.keypoint_mapping)
        self.face_landmarks_description = [f"landmark_{idx}" for idx in range(478)]
        self.blendshape_names = keypoint_mapping.blendshapes

        return super()._initialize_detector()

    def post_inference(self) -> DetectorOutput:
        out = DetectorOutput()
        raw_path = os.path.join(self.algorithm_out_folder, RAW_INFERENCE_PICKLE_NAME)
        with open(raw_path, "rb") as raw_file:
            per_frame_outputs = pickle.load(raw_file)

        bbox_2d, kp_2d, blendshapes, kp_3d_local = self._assign_subjects(per_frame_outputs)
        self.internal_bbox_2d = bbox_2d

        out.add_array("face_blendshapes", "blendshapes", blendshapes)
        out.add_array("face_landmarks_local", "2d", kp_2d)
        out.add_array("face_landmarks_local", "3d", kp_3d_local)

        # do we need to do keypoints filtering?
        filter_config = self.detector_config.filter_keypoints
        if filter_config.filtered:
            logging.info("Applying filtering for 2d facial keypoints...")
            kp_2d = filter_keypoints(kp_2d, filter_config)
            out.add_array("face_landmarks_local", "2d_filtered", kp_2d)

        # do we need to fill short detection dropouts?
        interpolation_config = self.detector_config.interpolate_keypoints
        if interpolation_config.interpolated:
            logging.info("Interpolating 2d facial keypoints...")
            kp_2d = interpolate_keypoints(kp_2d, interpolation_config)
            out.add_array("face_landmarks_local", "2d_interpolated", kp_2d)

        return out

    def visualization(self, out: DetectorOutput) -> None:
        # visualize 2d keypoints, from whichever post-processing stage the config selects
        kp_2d_key = self.detector_config.visualize_keypoints_npz_key
        kp_2d = out.get("face_landmarks_local", kp_2d_key)
        draw_keypoints(kp_2d, self.render_context, self.viz_folders["face_landmarks_local"], radius=1)

    def _assign_subjects(self, per_frame_outputs) -> tuple[NpzArray, NpzArray, NpzArray, NpzArray]:
        n_subjects = len(self.subjects)
        n_cams = len(self.cameras)
        n_frames = len(self.frame_names)
        n_keypoints = len(self.face_landmarks_description)

        if len(per_frame_outputs) != n_frames:
            raise ValueError(
                f"Raw pack holds {len(per_frame_outputs)} frames but the subsequence has {n_frames}. "
                "The pack was produced for different data - re-run inference (skip_inference)."
            )

        # NaN marks "no measurement": the detector itself never returns NaN coordinates.
        bbox_2d_axes = BBOX_2D_CONF.make_axes(subjects=self.subjects, cameras=self.cameras, frames=self.frame_names)
        bbox_2d = np.full((n_subjects, n_cams, n_frames, 5), np.nan, dtype=float)  # x0, y0, x1, y1, score

        keypoints_2d_axes = VECTOR_2D_CONF_PER_LABEL.make_axes(
            self.subjects, self.cameras, self.frame_names, labels=self.face_landmarks_description
        )
        keypoints_2d = np.full((n_subjects, n_cams, n_frames, n_keypoints, 3), np.nan, dtype=float)

        keypoints_3d_axes = VECTOR_3D_CONF_PER_LABEL.make_axes(
            self.subjects, self.cameras, self.frame_names, labels=self.face_landmarks_description
        )
        keypoints_3d = np.full((n_subjects, n_cams, n_frames, n_keypoints, 4), np.nan, dtype=float)

        blendshape_names = self.blendshape_names

        blendshapes_axes = FLOAT.make_axes(
            subjects=self.subjects,
            cameras=self.cameras,
            frames=self.frame_names,
            labels=blendshape_names,
            data=["coefficient"],
        )
        blendshapes_data = np.full((n_subjects, n_cams, n_frames, len(blendshape_names), 1), np.nan, dtype=float)

        for frame_idx, frame_bundle in enumerate(per_frame_outputs):
            for cam_idx, cam_name in enumerate(self.cameras):
                detections = frame_bundle.get(cam_name, [])
                subjects_by_cam = self.cam_sees_subjects[cam_name]
                n_seen = len(subjects_by_cam)

                for det in detections:
                    kps = np.array(det.get("keypoints", det.get("keypoints_3d", [])))
                    det["_kps"] = kps
                    if "bbox" not in det:
                        det["bbox"] = [
                            float(kps[:, 0].min()),
                            float(kps[:, 1].min()),
                            float(kps[:, 0].max()),
                            float(kps[:, 1].max()),
                        ]

                if len(detections) < n_seen:
                    # Missing a face: cannot know which subject, so leave all NaN for this frame.
                    if detections:
                        logging.debug(
                            f"MediapipeFaceLandmarks: camera '{cam_name}' frame {frame_idx} detected "
                            f"{len(detections)} face(s) < {n_seen} expected; marking all missing."
                        )
                    continue

                if len(detections) > n_seen:
                    # Too many faces: keep the largest ones by bounding box area, because we don't have confidence score
                    areas = [
                        (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1]) for det in detections
                    ]
                    keep = np.argsort(areas)[-n_seen:]
                    detections = [detections[i] for i in keep]

                # Sort left-to-right by bbox x0, then assign onto the subject slots this camera sees.
                detections = sorted(detections, key=lambda det: det["bbox"][0])

                for subject_index, det in zip(subjects_by_cam, detections):
                    score = det.get("det_score", 1.0)
                    bbox_2d[subject_index, cam_idx, frame_idx, :4] = det["bbox"]
                    bbox_2d[subject_index, cam_idx, frame_idx, 4] = score

                    kps = det["_kps"]
                    keypoints_2d[subject_index, cam_idx, frame_idx, :, :2] = kps[:, :2]
                    keypoints_2d[subject_index, cam_idx, frame_idx, :, 2] = score

                    keypoints_3d[subject_index, cam_idx, frame_idx, :, :3] = kps
                    keypoints_3d[subject_index, cam_idx, frame_idx, :, 3] = score

                    coeffs = [det["blendshapes"].get(name, np.nan) for name in blendshape_names]
                    blendshapes_data[subject_index, cam_idx, frame_idx, :, 0] = coeffs

                    # TODO: we skip face transform matrix, read it from native format
                    # get head direction component from there?

        return (
            NpzArray(bbox_2d, bbox_2d_axes),
            NpzArray(keypoints_2d, keypoints_2d_axes),
            NpzArray(blendshapes_data, blendshapes_axes),
            NpzArray(keypoints_3d, keypoints_3d_axes),
        )
