import logging

import numpy as np

from nicetoolbox_core.data.array_schema import BBOX_2D
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ...detector_outputs import DetectorOutput, NpzDetectorOutput
from ...utils.draw_2d import RenderContext, draw_bounding_boxes
from ..base_feature import BaseFeature
from .video_rules import AssignVideoRule, BBoxRule, HSplitHandler, NoneVideoRule, PathVideoRule, VSplitHandler

# same dict lookup like in detectors
# put your subject rule handler here to register it
HANDLER_CLASSES = {
    NoneVideoRule,
    AssignVideoRule,
    BBoxRule,
    VSplitHandler,
    HSplitHandler,
    PathVideoRule,
}
SUBJECT_RULE_HANDLERS = {cls.rule_type: cls for cls in HANDLER_CLASSES}


class VideoSubjectDetector(BaseFeature):
    """Compute per-subject regions per (camera, frame) from configured rules."""

    algorithm_type = "video_subject_detector"
    components = ["subject_regions"]

    outputs = [
        NpzDetectorOutput("subject_regions", "regions", schema=BBOX_2D),
    ]

    def compute(self) -> DetectorOutput:
        # TODO: move subjects regions to json?
        # because of limitations of npz file, we can output data only with same dimensions
        # that means that we can't have bounding box for one camera and polygon/segmentation mask for another
        # we can't efficiently serialize them as npz anyway
        subjects, cameras, frames = self.data.canonical_axes
        npz_axes = BBOX_2D.make_axes(subjects, cameras, frames)
        regions = np.full(npz_axes.shape, np.nan, dtype=float)

        # Loop over all cameras
        cameras_cfg = self.subsequence_context.sequence_properties.video.cameras
        for cam_idx, cam_name in enumerate(cameras):
            # Get rule for this camera
            rule_config = cameras_cfg[cam_name].subjects
            handler_cls = SUBJECT_RULE_HANDLERS.get(rule_config.rule)
            if handler_cls is None:
                raise ValueError(f"No handler registered for subject rule: '{rule_config.rule}'")

            # Apply rule for this camera
            cam_regions = handler_cls(self, cam_idx, rule_config).compute()
            regions[:, cam_idx, :, :] = cam_regions

        logging.info(f"Computed subject_regions for shape {regions.shape}.")

        out = DetectorOutput()
        out.add("subject_regions", "regions", data=regions, axes=npz_axes)
        return out

    def visualization(self, out: DetectorOutput) -> None:
        """Draw each subject's region as a bounding box on the source frames, per camera."""
        regions = out.get("subject_regions", "regions")
        cameras = list(regions.axes.cameras)

        render_context = RenderContext(
            dataloader=ImagePathsByFrameIndexLoader(config=self.data.get_input_recipes(), expected_cameras=cameras),
            fps=self.data.fps,
            video_start=self.data.video_start_frame_index,
            save_frames=self.save_frames,
        )
        draw_bounding_boxes(regions, render_context, self.viz_folders["subject_regions"])
