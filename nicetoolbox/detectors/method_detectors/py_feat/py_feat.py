"""
Py-feat method detector class.
"""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nicetoolbox_core.data.array_schema import BBOX_2D_CONF, BoundedSchema
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes, select_array
from nicetoolbox_core.entrypoint import SubprocessResult
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ....configs.schemas.detectors_instances_configs import MethodDetectorRuntime
from ...detector_inputs import NpzDetectorInput
from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ...subjects import assign_detections
from ...utils.draw_2d import RenderContext, draw_bounding_boxes, draw_emotions
from ..base_method import BaseMethod

FEATURE_COLUMNS_METADATA_KEY = b"feature_columns"

# Circumplex affect: valence is pleasantness, arousal is activation, both in [-1, 1].
VALENCE_AROUSAL = BoundedSchema(labels_columns=("valence", "arousal"), minimum=-1.0, maximum=1.0)


class PyFeat(BaseMethod):
    """
    The Python - Facial Expression Analysis Toolbox (Py-feat) is a method
    detector that computes emotion_individual and face_bounding_box components.
    """

    algorithm_type = "py_feat"
    components = ["emotion_individual", "face_bounding_box"]

    inputs = [
        NpzDetectorInput("subject_regions", "subjects"),
    ]

    outputs = [
        NpzDetectorOutput("face_bounding_box", "bbox_2d", schema=BBOX_2D_CONF),
        NpzDetectorOutput("emotion_individual", "emotions"),
        NpzDetectorOutput("emotion_individual", "valence_arousal", schema=VALENCE_AROUSAL),
    ]

    def _initialize_detector(self) -> MethodDetectorRuntime:
        self.subjects = self.data.subjects_descr
        self.frame_names = self.data.frame_labels
        self.cameras = self.detector_config.camera_names
        self.subject_regions = select_array(self.loaded_inputs["subjects"].array, cameras=self.cameras)

        self.render_context = RenderContext(
            dataloader=ImagePathsByFrameIndexLoader(
                config=self.data.get_input_recipes(), expected_cameras=self.cameras
            ),
            fps=self.data.fps,
            video_start=self.data.video_start_frame_index,
        )

        return super()._initialize_detector()

    def post_inference(self, result_manifest: SubprocessResult) -> DetectorOutput:
        out = DetectorOutput()
        # read raw prediction from native side
        raw_path = result_manifest.get_raw_output()
        # Which columns form each feature block. Written by the inference script, since Fex keeps
        # these groupings as python attributes that parquet cannot carry.
        meta_json = pq.read_schema(raw_path).metadata[FEATURE_COLUMNS_METADATA_KEY]
        self.feature_columns = json.loads(meta_json)
        raw_table = pd.read_parquet(raw_path).reset_index(drop=True)

        # Identity is resolved once and written onto the table, so every component below reads
        # the same assignment - they only differ in which columns they pull out.
        raw_table = self._assign_subjects(raw_table)

        out.add_array("face_bounding_box", "bbox_2d", self._face_bounding_box(raw_table))
        out.add_array("emotion_individual", "emotions", self._emotions(raw_table))
        out.add_array("emotion_individual", "valence_arousal", self._valence_arousal(raw_table))
        # TODO: post process the remaining blocks (AUs, gaze, pose)
        return out

    def visualization(self, out: DetectorOutput) -> None:
        bbox_2d = out.get("face_bounding_box", "bbox_2d")
        draw_bounding_boxes(bbox_2d, self.render_context, self.viz_folders["face_bounding_box"])

        emotions = out.get("emotion_individual", "emotions")
        draw_emotions(bbox_2d, emotions, self.render_context, self.viz_folders["emotion_individual"])

    def _face_bounding_box(self, table: pd.DataFrame) -> NpzArray:
        columns = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight", "FaceScore"]
        bbox_raw = self._select_columns(table, columns)
        # convert to toolbox x1, y1, x2, y2, conf convention
        # take last dimension from table
        x1, y1, w, h, conf = (bbox_raw[..., i] for i in range(5))

        axes = BBOX_2D_CONF.make_axes(self.subjects, self.cameras, self.frame_names)
        bbox = np.stack([x1, y1, x1 + w, y1 + h, conf], axis=-1)
        return NpzArray(bbox, axes)

    def _emotions(self, table: pd.DataFrame) -> NpzArray:
        # per-emotion probabilities, summing to 1 across the labels axis
        labels = self.feature_columns["emotions"]
        axes = NpzArrayAxes(self.subjects, self.cameras, self.frame_names, labels)
        return NpzArray(self._select_columns(table, labels), axes)

    def _valence_arousal(self, table: pd.DataFrame) -> NpzArray:
        # the schema fixes the label order, so the raw columns are read in that order
        columns = self.feature_columns["valence_arousal"]
        axes = VALENCE_AROUSAL.make_axes(self.subjects, self.cameras, self.frame_names)
        return NpzArray(self._select_columns(table, columns), axes)

    def _assign_subjects(self, raw_table: pd.DataFrame) -> pd.DataFrame:
        """Add a subject column telling which subject each detected face belongs to."""
        n_frames = len(self.frame_names)

        n_raw_frames = raw_table["frame"].max() + 1 if len(raw_table) else 0
        if n_raw_frames != n_frames:
            raise ValueError(
                f"Raw table holds {n_raw_frames} frames but the subsequence has {n_frames}. "
                "The table was produced for different data - re-run inference (skip_inference)."
            )

        # -1 marks faces that stay unassigned, so they drop out when reshaping to the grid.
        subject = np.full(len(raw_table), -1, dtype=int)
        regions = self.subject_regions.data
        assignment_config = self.subsequence_context.video_subject_assignment

        for (cam_name, frame_idx), detections in raw_table.groupby(["camera", "frame"], sort=False):
            if cam_name not in self.cameras:
                continue
            cam_idx = self.cameras.index(cam_name)

            x = detections["FaceRectX"].to_numpy(dtype=float)
            y = detections["FaceRectY"].to_numpy(dtype=float)
            w = detections["FaceRectWidth"].to_numpy(dtype=float)
            h = detections["FaceRectHeight"].to_numpy(dtype=float)
            boxes = np.stack([x, y, x + w, y + h], axis=-1)
            scores = detections["FaceScore"].to_numpy(dtype=float)

            subject_regions_frame = regions[:, cam_idx, frame_idx, :4]
            assigned = assign_detections(boxes, scores, subject_regions_frame, assignment_config)
            subject[detections.index] = assigned

        return raw_table.assign(subject=subject)

    def _select_columns(self, table: pd.DataFrame, columns: list[str]) -> np.ndarray:
        """
        Reshape assigned detections onto the canonical (subjects, cameras, frames) grid.

        Args:
            table (pd.DataFrame): Table carrying the 'subject' column set by _assign_subjects.
            columns (list[str]): Columns to read, in output order.

        Returns:
            np.ndarray: (subjects, cameras, frames, len(columns)), NaN where no detection.
        """
        assigned = table[table["subject"] >= 0].set_index(["subject", "camera", "frame"])
        # Rows only exist where a camera saw a subject, so reindexing over the full grid fills
        # the rest with NaN - "no measurement", which the detector itself never returns.
        full_grid = pd.MultiIndex.from_product([range(len(self.subjects)), self.cameras, range(len(self.frame_names))])
        values = assigned[columns].reindex(full_grid).to_numpy(dtype=float)
        return values.reshape(len(self.subjects), len(self.cameras), len(self.frame_names), len(columns))
