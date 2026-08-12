"""
Shared 2d overlay rendering for detectors.

Every detector that draws per-subject 2d results walks the same skeleton: for each camera, read
each source frame, draw whatever the subjects visible to that camera have, write the frame, and
encode a video. That skeleton lives in `render_per_camera_videos`; the drawing itself is a
callback, so a detector only writes what is specific to it.

Bounding boxes and keypoints are common enough to be provided directly.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from nicetoolbox_core.data.loaded_array import NpzArray

from ...utils import video as vd

# Per-subject BGR colours, cycled when there are more subjects than colours.
SUBJECT_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

# Default arrow length in pixels. A direction is a heading only, so arrows are drawn at a fixed
# on-screen size rather than in any metric unit.
ARROW_LENGTH_PX = 200


def subject_color(subject_idx: int) -> tuple[int, int, int]:
    """BGR colour for a subject, so every overlay gives the same subject the same colour."""
    return SUBJECT_COLORS[subject_idx % len(SUBJECT_COLORS)]


@dataclass
class RenderContext:
    """Everything the render loop needs that is not in the array being drawn.

    This is per-subsequence, so a detector builds it once in _initialize_detector() and passes
    it to every overlay it draws. The subject and camera axes are not here: they come from the
    NpzArray, so they always describe the data actually being rendered.
    """

    dataloader: object  # yields (real_frame_idx, {camera_name: frame_path})
    cam_sees_subjects: dict[str, list[int]]
    fps: int
    video_start: int


def render_per_camera_videos(
    draw_subject: Callable[[np.ndarray, int, int, int], None],
    axes,
    context: RenderContext,
    viz_folder: str,
) -> None:
    """
    Walk every camera/frame/visible-subject and let `draw_subject` paint the overlay.

    Frames are written to <viz_folder>/<camera>/ and combined into <viz_folder>/<camera>.mp4.

    Args:
        draw_subject (callable): Called as (image, subject_idx, cam_idx, frame_idx); draws in place.
        axes (NpzArrayAxes): Axes of the array being drawn; supplies the cameras and subjects.
        context (RenderContext): Frame source and timing for this subsequence.
        viz_folder (str): Visualization folder of the component being rendered.
    """
    for cam_idx, camera_name in enumerate(axes.cameras):
        os.makedirs(os.path.join(viz_folder, camera_name), exist_ok=True)

        for frame_idx, (real_frame_idx, frame_paths_per_camera) in enumerate(context.dataloader):
            image = cv2.imread(frame_paths_per_camera[camera_name])

            for subject_idx in range(len(axes.subjects)):
                if subject_idx not in context.cam_sees_subjects[camera_name]:
                    continue
                draw_subject(image, subject_idx, cam_idx, frame_idx)

            cv2.imwrite(os.path.join(viz_folder, camera_name, f"{real_frame_idx:09d}.jpg"), image)

        # create and save video. frames_to_video returns an ffmpeg exit code: 0 is success.
        exit_code = vd.frames_to_video(
            os.path.join(viz_folder, camera_name),
            os.path.join(viz_folder, f"{camera_name}.mp4"),
            fps=context.fps,
            start_frame=context.video_start,
        )
        if exit_code != 0:
            logging.error(f"Failed to encode '{camera_name}.mp4' in {viz_folder} (exit code {exit_code}).")

    logging.info(f"Visualization finished: {viz_folder}")


def draw_keypoints(keypoints_2d: NpzArray, context: RenderContext, viz_folder: str, radius: int = 2) -> None:
    """
    Draw each subject's 2d keypoints on the source frames and stitch a video per camera.

    Individual keypoints are skipped when missing, so a partially interpolated set still
    shows whichever points exist.

    Args:
        keypoints_2d (NpzArray): (subjects, cameras, frames, keypoints, x/y[/conf]). The
            confidence column is optional - only the first two columns are read.
        context (RenderContext): Frame source and timing for this subsequence.
        viz_folder (str): Visualization folder of the component being rendered.
        radius (int): Point radius in pixels; dense conventions want a smaller dot.
    """
    data = keypoints_2d.data

    def draw_subject(image, subject_idx, cam_idx, frame_idx):
        color = subject_color(subject_idx)
        for point in data[subject_idx, cam_idx, frame_idx]:
            if np.isnan(point[:2]).any():
                continue
            center = np.round(point[:2]).astype(np.int32)
            cv2.circle(image, center, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    render_per_camera_videos(draw_subject, keypoints_2d.axes, context, viz_folder)


def draw_direction_vector(
    direction_2d: NpzArray,
    origins_2d: NpzArray,
    context: RenderContext,
    viz_folder: str,
    arrow_length: int = ARROW_LENGTH_PX,
) -> None:
    """
    Draw each subject's direction as a single arrow, and stitch a video per camera.

    `direction_2d` is a unitless heading, so it is scaled to a fixed on-screen length rather than
    drawn in any metric unit. The camera axis is taken from `direction_2d`.

    Frames are written to <viz_folder>/<camera>/ and combined into <viz_folder>/<camera>.mp4.

    Args:
        direction_2d (NpzArray): (subjects, cameras, frames, dx/dy[/conf]) image-plane heading.
        origins_2d (NpzArray): (subjects, cameras, frames, x/y[/conf]) arrow anchor in pixels.
        context (RenderContext): Frame source and timing for this subsequence.
        viz_folder (str): Visualization folder of the component being rendered.
        arrow_length (int): On-screen arrow length in pixels.
    """
    directions = direction_2d.data
    origins = origins_2d.data

    def draw_subject(image, subject_idx, cam_idx, frame_idx):
        origin = origins[subject_idx, cam_idx, frame_idx, :2]
        direction = directions[subject_idx, cam_idx, frame_idx, :2]
        if np.isnan(origin).any() or np.isnan(direction).any():
            return

        tip = origin + direction * arrow_length
        cv2.arrowedLine(
            image,
            np.round(origin).astype(np.int32),
            np.round(tip).astype(np.int32),
            subject_color(subject_idx),
            thickness=2,
            tipLength=0.2,
            line_type=cv2.LINE_AA,
        )

    render_per_camera_videos(draw_subject, direction_2d.axes, context, viz_folder)


def draw_bounding_boxes(bbox_2d: NpzArray, context: RenderContext, viz_folder: str) -> None:
    """
    Draw each subject's bounding box on the source frames and stitch a video per camera.

    Frames where no detection was assigned to the subject's slot (NaN box) are skipped.

    Args:
        bbox_2d (NpzArray): (subjects, cameras, frames, x0/y0/x1/y1/conf).
        context (RenderContext): Frame source and timing for this subsequence.
        viz_folder (str): Visualization folder of the component being rendered.
    """
    data = bbox_2d.data
    subjects = bbox_2d.axes.subjects

    def draw_subject(image, subject_idx, cam_idx, frame_idx):
        # x0, y0, x1, y1, confidence; NaN whenever no detection was assigned to this slot.
        box = data[subject_idx, cam_idx, frame_idx]
        if np.isnan(box[:4]).any():
            return

        top_left = np.round(box[:2]).astype(np.int32)
        bottom_right = np.round(box[2:4]).astype(np.int32)
        color = subject_color(subject_idx)
        cv2.rectangle(image, top_left, bottom_right, color, thickness=2)
        cv2.putText(
            image,
            f"{subjects[subject_idx]} {box[4]:.2f}",
            (int(top_left[0]), int(top_left[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    render_per_camera_videos(draw_subject, bbox_2d.axes, context, viz_folder)
