import pickle

import numpy as np

from nicetoolbox_core.data.array_schema import BBOX_2D_CONF, FLOAT
from nicetoolbox_core.data.loaded_array import NpzArray, select_array
from nicetoolbox_core.entrypoint import SubprocessResult
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from .....configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ....detector_inputs import NpzDetectorInput
from ....detector_outputs import DetectorOutput, NpzDetectorOutput
from ....subjects import assign_detections
from ....utils.draw_2d import RenderContext, draw_bounding_boxes
from ....utils.pose_utils import create_iou_all_pairs
from ...base_method import BaseMethod


# TODO: add support for temporal tracking (built-in into mmdet, i.e. ByteTrack)
class MMDet(BaseMethod):
    algorithm_type = "mmdet"

    # TODO: new components for other object classes (i.e. faces, chairs, tables)?
    # we can drive them through configs and mapping of coco - components
    components = ["person_bounding_box"]

    # input and outputs
    inputs = [
        NpzDetectorInput("subject_regions", "subjects"),
    ]
    outputs = [
        NpzDetectorOutput("person_bounding_box", "bbox_2d", schema=BBOX_2D_CONF),
        NpzDetectorOutput("person_bounding_box", "bbox_overlap", schema=FLOAT),
    ]

    def _initialize_detector(self) -> MethodDetectorRuntime:
        self.subjects = self.data.subjects_descr
        self.frame_names = self.data.frame_labels
        self.cameras = self.detector_config.camera_names
        self.video_start = self.data.video_start_frame_index

        # for visualization
        self.render_context = RenderContext(
            dataloader=ImagePathsByFrameIndexLoader(
                config=self.data.get_input_recipes(), expected_cameras=self.cameras
            ),
            fps=self.data.fps,
            video_start=self.video_start,
        )

        return super()._initialize_detector()

    def post_inference(self, result_manifest: SubprocessResult) -> DetectorOutput:
        with open(result_manifest.get_raw_output(), "rb") as raw_file:
            per_camera_outputs = pickle.load(raw_file)

        # assign subjects
        bbox_2d = self._assign_subjects(per_camera_outputs)
        # low confidence discard
        bbox_low_conf = bbox_2d.data[..., 4] < self.detector_config.det_thresh
        bbox_2d.data[bbox_low_conf] = np.nan
        # TODO: filtering and interpolation of bounding boxes?

        # bbox overlap + iou discard
        bbox_overlap = self._compute_bbox_overlap(bbox_2d)
        self._apply_iou_discard(bbox_2d, bbox_overlap)

        out = DetectorOutput()
        out.add_array("person_bounding_box", "bbox_2d", bbox_2d)
        out.add_array("person_bounding_box", "bbox_overlap", bbox_overlap)
        return out

    def visualization(self, out: DetectorOutput) -> None:
        bbox_2d = out.get("person_bounding_box", "bbox_2d")
        draw_bounding_boxes(bbox_2d, self.render_context, self.viz_folders["person_bounding_box"])

    def _assign_subjects(self, per_camera_outputs) -> NpzArray:
        n_subjects = len(self.subjects)
        n_cams = len(self.cameras)
        n_frames = len(self.frame_names)

        bbox_2d_axes = BBOX_2D_CONF.make_axes(self.subjects, self.cameras, self.frame_names)
        bbox_2d = np.full((n_subjects, n_cams, n_frames, 5), np.nan, dtype=float)

        assignment_config = self.subsequence_context.video_subject_assignment
        subject_regions = select_array(self.loaded_inputs["subjects"].array, cameras=self.cameras)
        for cam_idx, cam_name in enumerate(self.cameras):
            for frame_idx, detections in enumerate(per_camera_outputs[cam_name]):
                if not detections:
                    continue

                boxes = np.array([det["bbox"] for det in detections], dtype=float).reshape(-1, 4)
                scores = np.array([det["score"] for det in detections], dtype=float)
                frame_regions = subject_regions.data[:, cam_idx, frame_idx, :4]

                assigned = assign_detections(boxes, scores, frame_regions, assignment_config)

                for det_idx, subj_idx in enumerate(assigned):
                    if subj_idx < 0:
                        continue
                    bbox_2d[subj_idx, cam_idx, frame_idx, :4] = detections[det_idx]["bbox"]
                    bbox_2d[subj_idx, cam_idx, frame_idx, 4] = detections[det_idx]["score"]

        return NpzArray(bbox_2d, bbox_2d_axes)

    def _compute_bbox_overlap(self, bbox_2d: NpzArray) -> NpzArray:
        # create_iou_all_pairs expects (S, C, F, 1, 5); insert the "full_body" label axis.
        overlap = create_iou_all_pairs(bbox_2d.data[:, :, :, None, :])
        overlap_labels = [subj for subj in self.subjects]
        overlap_axes = FLOAT.make_axes(self.subjects, self.cameras, self.frame_names, labels=overlap_labels)
        return NpzArray(overlap, overlap_axes)

    def _apply_iou_discard(self, bbox_2d: NpzArray, bbox_overlap: NpzArray) -> None:
        # Drop any bbox that overlaps another subject's bbox above iou_thresh.
        # Diagonal (self-overlap) is 1.0 by construction, so mask it out before the reduction.
        iou = np.nan_to_num(bbox_overlap.data, nan=0.0)  # (S, C, F, S)
        for s in range(iou.shape[0]):
            iou[s, :, :, s] = 0.0
        max_other = iou.max(axis=-1)  # (S, C, F)
        discard_mask = max_other > self.detector_config.iou_thresh
        bbox_2d.data[discard_mask] = np.nan
