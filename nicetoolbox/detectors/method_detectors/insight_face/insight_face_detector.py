"""
InsightFace method detector class.
"""

import logging
import os
import pickle

import numpy as np

from nicetoolbox_core.data.array_schema import (
    BBOX_2D_CONF,
    VECTOR_2D_CONF_PER_LABEL,
    VECTOR_2D_PER_LABEL,
    VECTOR_3D_CONF_PER_LABEL,
)
from nicetoolbox_core.data.loaded_array import NpzArray
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ....configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ...utils.draw_2d import RenderContext, draw_bounding_boxes, draw_keypoints
from ...utils.keypoints_post_processing import filter_keypoints, interpolate_keypoints, reproject_keypoints
from ...utils.keypoints_triangualation import triangulate_keypoints
from ..base_method import BaseMethod

# SCRFD emits the box plus these 5 keypoints per face, in this order
FACE_5PT_LABELS = ["eye_right", "eye_left", "nose", "mouth_right", "mouth_left"]
RAW_INFERENCE_PICKLE_NAME = "insight_face_inference_raw.pkl"


class InsightFace(BaseMethod):
    """
    InsightFace is a method detector for face detection.
    """

    components = ["face_bounding_box", "face_landmarks"]
    algorithm_type = "insight_face"

    # detectors outputs
    def resolve_outputs(self) -> list[NpzDetectorOutput]:
        outputs = [
            NpzDetectorOutput("face_bounding_box", "bbox_2d", schema=BBOX_2D_CONF),
            NpzDetectorOutput("face_landmarks", "2d", schema=VECTOR_2D_CONF_PER_LABEL),
        ]
        # optional post processing + triangulation for facial landmarks
        if self.detector_config.filter_keypoints.filtered:
            outputs.append(NpzDetectorOutput("face_landmarks", "2d_filtered", schema=VECTOR_2D_CONF_PER_LABEL))
        if self.detector_config.interpolate_keypoints.interpolated:
            outputs.append(NpzDetectorOutput("face_landmarks", "2d_interpolated", schema=VECTOR_2D_CONF_PER_LABEL))
        if self.detector_config.triangulate_keypoints.triangulate:
            outputs.append(NpzDetectorOutput("face_landmarks", "3d", schema=VECTOR_3D_CONF_PER_LABEL))
            outputs.append(NpzDetectorOutput("face_landmarks", "2d_reprojected_from_3d", schema=VECTOR_2D_PER_LABEL))
        return outputs

    def _initialize_detector(self) -> MethodDetectorRuntime:
        """
        Initialize the InsightFace detector with extra configuration settings.
        """
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

        return super()._initialize_detector()

    def post_inference(self) -> DetectorOutput:
        out = DetectorOutput()
        # read raw prediction from native side
        raw_path = os.path.join(self.out_folders[self.components[0]], RAW_INFERENCE_PICKLE_NAME)
        with open(raw_path, "rb") as raw_file:
            per_frame_outputs = pickle.load(raw_file)

        # TODO: make this snippet reusable across different body joints detectors
        # extract raw data and assign subjects identities
        bbox_2d, kp_2d = self._assign_subjects(per_frame_outputs)
        out.add_array("face_bounding_box", "bbox_2d", bbox_2d)
        out.add_array("face_landmarks", "2d", kp_2d)
        # do we need to do keypoints filtering?
        filter_config = self.detector_config.filter_keypoints
        if filter_config.filtered:
            logging.info("Applying filtering for 2d facial keypoints...")
            kp_2d = filter_keypoints(kp_2d, filter_config)
            out.add_array("face_landmarks", "2d_filtered", kp_2d)
        # do we need to fill short detection dropouts?
        interpolation_config = self.detector_config.interpolate_keypoints
        if interpolation_config.interpolated:
            logging.info("Interpolating 2d facial keypoints...")
            kp_2d = interpolate_keypoints(kp_2d, interpolation_config)
            out.add_array("face_landmarks", "2d_interpolated", kp_2d)
        # do we need to lift the keypoints to 3d?
        # TODO: on example dataset 3d quality is very poor - we use only top and center camera
        # using view_ceter and view_top allow us to see both subjects, but reprojection error is too high
        # ideally, we need more robust algorithm that can utilize view_center and face cameras
        triangulation_config = self.detector_config.triangulate_keypoints
        if triangulation_config.triangulate:
            logging.info("Triangulating 2d facial keypoints to 3d...")
            kp_3d = triangulate_keypoints(
                kp_2d,
                triangulation_config,
                calibration=self.data.calibration,
                cam_sees_subjects=self.cam_sees_subjects,
            )
            out.add_array("face_landmarks", "3d", kp_3d)
            # also add back projection for debugging/evaluation
            kp_2d_backprojected = reproject_keypoints(
                kp_3d,
                calibration=self.data.calibration,
                camera_names=self.cameras,
            )
            out.add_array("face_landmarks", "2d_reprojected_from_3d", kp_2d_backprojected)

        # TODO: do we need bounding box filtering/interpolation?
        return out

    def visualization(self, out: DetectorOutput) -> None:
        # visualize bounding boxes
        bbox_2d = out.get("face_bounding_box", "bbox_2d")
        draw_bounding_boxes(bbox_2d, self.render_context, self.viz_folders["face_bounding_box"])

        # visualize 2d keypoints, from whichever post-processing stage the config selects
        kp_2d_key = self.detector_config.visualize_keypoints_npz_key
        kp_2d = out.get("face_landmarks", kp_2d_key)
        draw_keypoints(kp_2d, self.render_context, self.viz_folders["face_landmarks"])

    def _assign_subjects(self, per_frame_outputs):
        """
        Assign the raw pack's ragged per-camera detections to dense subject-indexed arrays.

        InsightFace has no tracker and detection is stateless per frame, so faces carry no
        identity. The same slot policy as the other face detectors is used:

          - Too many faces for a camera: keep the N highest-confidence faces (N = subjects this
            camera sees), then restore their left-to-right order.
          - Too few faces: cannot tell which subject is missing, so mark all of this camera's
            subjects missing (NaN) for that frame.
          - Otherwise: assign faces left-to-right onto the subject slots this camera sees.

        Args:
            per_frame_outputs: Per-frame {camera_name: [detection dicts]} from the raw pack.

        Returns:
            bbox_2d (NpzArray): (subjects, cameras, frames, x0/y0/x1/y1/conf).
            keypoints_2d (NpzArray): (subjects, cameras, frames, 5, x/y/conf).
        """
        n_subjects = len(self.subjects)
        n_cams = len(self.cameras)
        n_frames = len(self.frame_names)
        n_keypoints = len(FACE_5PT_LABELS)

        if len(per_frame_outputs) != n_frames:
            raise ValueError(
                f"Raw pack holds {len(per_frame_outputs)} frames but the subsequence has {n_frames}. "
                "The pack was produced for different data - re-run inference (skip_inference)."
            )

        # NaN marks "no measurement": the detector itself never returns NaN coordinates.
        bbox_2d_axes = BBOX_2D_CONF.make_axes(self.subjects, self.cameras, self.frame_names)
        bbox_2d = np.full((n_subjects, n_cams, n_frames, 5), np.nan, dtype=float)

        keypoints_2d_axes = VECTOR_2D_CONF_PER_LABEL.make_axes(
            self.subjects, self.cameras, self.frame_names, labels=FACE_5PT_LABELS
        )
        keypoints_2d = np.full((n_subjects, n_cams, n_frames, n_keypoints, 3), np.nan, dtype=float)

        for frame_idx, frame_bundle in enumerate(per_frame_outputs):
            for cam_idx, cam_name in enumerate(self.cameras):
                detections = frame_bundle.get(cam_name, []) if frame_bundle else []
                subjects_by_cam = self.cam_sees_subjects[cam_name]
                n_seen = len(subjects_by_cam)

                if len(detections) < n_seen:
                    # Missing a face: cannot know which subject, so leave all NaN for this frame.
                    if detections:
                        logging.debug(
                            f"InsightFace: camera '{cam_name}' frame {frame_idx} detected "
                            f"{len(detections)} face(s) < {n_seen} expected; marking all missing."
                        )
                    continue

                if len(detections) > n_seen:
                    # Too many faces: keep the N highest-confidence ones.
                    scores = [det["det_score"] for det in detections]
                    keep = np.argsort(scores)[-n_seen:]
                    detections = [detections[i] for i in keep]

                # Sort left-to-right by bbox x0, then assign onto the subject slots this camera sees.
                detections = sorted(detections, key=lambda det: det["bbox"][0])

                for subject_index, det in zip(subjects_by_cam, detections):
                    score = det["det_score"]
                    bbox_2d[subject_index, cam_idx, frame_idx, :4] = det["bbox"]
                    bbox_2d[subject_index, cam_idx, frame_idx, 4] = score

                    keypoints_2d[subject_index, cam_idx, frame_idx, :, :2] = det["keypoints_2d"]
                    # SCRFD scores the face, not each keypoint; broadcast the detection score so
                    # the per-label confidence column carries the only confidence that exists.
                    keypoints_2d[subject_index, cam_idx, frame_idx, :, 2] = score

        return NpzArray(bbox_2d, bbox_2d_axes), NpzArray(keypoints_2d, keypoints_2d_axes)
