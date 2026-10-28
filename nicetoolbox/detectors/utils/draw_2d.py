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
from typing import Callable, Optional

import cv2
import numpy as np

from nicetoolbox_core.data.loaded_array import NpzArray

from ...utils import video as vd
from ...utils import visual_utils as vis_ut

# Per-subject BGR colours, cycled when there are more subjects than colours.
SUBJECT_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

# Per-label BGR colours for categorical overlays (emotions, classes, ...), cycled the same way.
# Kept distinct from SUBJECT_COLORS so a label overlay is never mistaken for a subject one.
# Mirrored in visualizer_config.toml as RGB, so a label keeps its colour between the
# detector's mp4 and the rerun visualizer.
LABEL_COLORS = [
    (128, 128, 128),  # grey
    (0, 215, 255),  # gold
    (139, 61, 72),  # dark slate blue
    (255, 191, 0),  # deep sky blue
    (128, 0, 128),  # purple
    (47, 170, 85),  # olive green
    (0, 0, 255),  # red
    (255, 0, 255),  # magenta
    (0, 128, 128),  # olive
]

# Default arrow length in pixels. A direction is a heading only, so arrows are drawn at a fixed
# on-screen size rather than in any metric unit.
ARROW_LENGTH_PX = 200


def subject_color(subject_idx: int) -> tuple[int, int, int]:
    """BGR colour for a subject, so every overlay gives the same subject the same colour."""
    return SUBJECT_COLORS[subject_idx % len(SUBJECT_COLORS)]


def label_color(label_idx: int) -> tuple[int, int, int]:
    """BGR colour for a categorical label, by its position on the array's labels axis."""
    return LABEL_COLORS[label_idx % len(LABEL_COLORS)]


@dataclass
class RenderContext:
    """Everything the render loop needs that is not in the array being drawn.

    This is per-subsequence, so a detector builds it once in _initialize_detector() and passes
    it to every overlay it draws. The subject and camera axes are not here: they come from the
    NpzArray, so they always describe the data actually being rendered. Per-subject skipping is
    driven by NaN slots on the array itself, so no explicit "who does this camera see" list is
    carried here unless optionally specified in cam_sees_subjects.
    """

    dataloader: object  # yields (real_frame_idx, {camera_name: frame_path})
    fps: int
    video_start: int
    save_frames: bool = False
    cam_sees_subjects: Optional[dict[str, list[int]]] = None


def open_video_writer(viz_folder: str, name: str, size: tuple[int, int], context: RenderContext) -> vd.FFmpegPipeWriter:
    """
    Open a pipe writer streaming BGR frames into <viz_folder>/<name>.mp4.

    If context.save_frames is True, frames are also saved to <viz_folder>/images/<name>/.

    Args:
        viz_folder (str): Visualization folder of the component being rendered.
        name (str): Video name relative to viz_folder, without extension (e.g. "view_top").
        size (tuple[int, int]): Frame (width, height).
        context (RenderContext): Supplies fps and the save_frames flag.

    Returns:
        vd.FFmpegPipeWriter: The open writer; the caller closes it.
    """
    out_mp4 = os.path.join(viz_folder, f"{name}.mp4")
    frames_dir = os.path.join(viz_folder, "images", name) if context.save_frames else None
    return vd.FFmpegPipeWriter(out_mp4, size=size, fps=context.fps, save_frames_dir=frames_dir)


def render_per_camera_videos(
    draw_subject: Callable[[np.ndarray, int, int, int], None],
    axes,
    context: RenderContext,
    viz_folder: str,
) -> None:
    """
    Walk every camera/frame/visible-subject and let `draw_subject` paint the overlay.

    Frames are streamed in memory directly into <viz_folder>/<camera>.mp4.
    If context.save_frames is True, frames are also saved to <viz_folder>/images/<camera>/.

    Args:
        draw_subject (callable): Called as (image, subject_idx, cam_idx, frame_idx); draws in place.
        axes (NpzArrayAxes): Axes of the array being drawn; supplies the cameras and subjects.
        context (RenderContext): Frame source and timing for this subsequence.
        viz_folder (str): Visualization folder of the component being rendered.
    """
    os.makedirs(viz_folder, exist_ok=True)
    writers: dict[str, vd.FFmpegPipeWriter] = {}

    try:
        for frame_idx, (real_frame_idx, frame_paths_per_camera) in enumerate(context.dataloader):
            for cam_idx, camera_name in enumerate(axes.cameras):
                frame_path = frame_paths_per_camera.get(camera_name)
                if not frame_path:
                    continue

                image = cv2.imread(frame_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image from '{frame_path}'")

                if camera_name not in writers:
                    height, width = image.shape[:2]
                    writers[camera_name] = open_video_writer(viz_folder, camera_name, (width, height), context)

                visible_subjects = (
                    context.cam_sees_subjects.get(camera_name) if context.cam_sees_subjects is not None else None
                )
                for subject_idx in range(len(axes.subjects)):
                    if visible_subjects is not None and subject_idx not in visible_subjects:
                        continue
                    draw_subject(image, subject_idx, cam_idx, frame_idx)

                writers[camera_name].write_frame(image, frame_idx=real_frame_idx)
    finally:
        for writer in writers.values():
            writer.close()

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


def draw_skeleton(
    keypoints_2d: NpzArray,
    connections: list[list[str]],
    context: RenderContext,
    viz_folder: str,
    radius: int = 2,
    thickness: int = 2,
) -> None:
    """
    Draw each subject's 2d keypoints and skeleton edges on the source frames.

    Connections are polylines of label names as stored in predictions_mapping (e.g.
    [["left_wrist", "left_elbow", "left_shoulder"], ...]); each consecutive pair becomes
    an edge. Labels not present on the array's labels axis are skipped, and any edge with
    a missing endpoint is skipped so partial data still renders.

    Args:
        keypoints_2d (NpzArray): (subjects, cameras, frames, keypoints, x/y[/conf]).
        connections (list[list[str]]): Polylines of label names to connect.
        context (RenderContext): Frame source and timing for this subsequence.
        viz_folder (str): Visualization folder of the component being rendered.
        radius (int): Point radius in pixels.
        thickness (int): Edge line thickness in pixels.
    """
    data = keypoints_2d.data
    label_to_idx = {label: i for i, label in enumerate(keypoints_2d.axes.labels)}
    edges = [
        (label_to_idx[a], label_to_idx[b])
        for polyline in connections
        for a, b in zip(polyline, polyline[1:])
        if a in label_to_idx and b in label_to_idx
    ]

    def draw_subject(image, subject_idx, cam_idx, frame_idx):
        color = subject_color(subject_idx)
        points = data[subject_idx, cam_idx, frame_idx, :, :2]
        for start_idx, end_idx in edges:
            pa, pb = points[start_idx], points[end_idx]
            if np.isnan(pa).any() or np.isnan(pb).any():
                continue
            cv2.line(
                image,
                tuple(np.round(pa).astype(np.int32)),
                tuple(np.round(pb).astype(np.int32)),
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        for point in points:
            if np.isnan(point).any():
                continue
            center = tuple(np.round(point).astype(np.int32))
            cv2.circle(image, center, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(image, center, radius=radius + 1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

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
        bbox_2d (NpzArray): (subjects, cameras, frames, x0/y0/x1/y1[/conf]). The
            confidence column is optional - when present it is appended to the label.
        context (RenderContext): Frame source and timing for this subsequence.
        viz_folder (str): Visualization folder of the component being rendered.
    """
    data = bbox_2d.data
    subjects = bbox_2d.axes.subjects
    has_confidence = data.shape[-1] >= 5

    def draw_subject(image, subject_idx, cam_idx, frame_idx):
        # x0, y0, x1, y1[, confidence]; NaN whenever no detection was assigned to this slot.
        box = data[subject_idx, cam_idx, frame_idx]
        if np.isnan(box[:4]).any():
            return

        top_left = np.round(box[:2]).astype(np.int32)
        bottom_right = np.round(box[2:4]).astype(np.int32)
        color = subject_color(subject_idx)
        cv2.rectangle(image, top_left, bottom_right, color, thickness=2)
        label = f"{subjects[subject_idx]} {box[4]:.2f}" if has_confidence else subjects[subject_idx]
        (_, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Prefer above the box; drop inside when the top edge would clip the label.
        text_y = int(top_left[1]) - 5
        if text_y - text_h < 0:
            text_y = int(top_left[1]) + text_h + 5
        cv2.putText(
            image,
            label,
            (int(top_left[0]), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    render_per_camera_videos(draw_subject, bbox_2d.axes, context, viz_folder)


def draw_emotions(bbox_2d: NpzArray, emotions: NpzArray, context: RenderContext, viz_folder: str) -> None:
    """
    Draw each subject's bounding box coloured by their strongest emotion, labelled with its score.

    Frames where no detection was assigned to the subject's slot (NaN box) are skipped.

    Args:
        bbox_2d (NpzArray): (subjects, cameras, frames, x0/y0/x1/y1/conf).
        emotions (NpzArray): (subjects, cameras, frames, emotions) scores; the labels axis names
            the emotions, so any label set works.
        context (RenderContext): Frame source and timing for this subsequence.
        viz_folder (str): Visualization folder of the component being rendered.
    """
    boxes = bbox_2d.data
    scores = emotions.data
    labels = emotions.axes.labels

    def draw_subject(image, subject_idx, cam_idx, frame_idx):
        box = boxes[subject_idx, cam_idx, frame_idx]
        subject_scores = scores[subject_idx, cam_idx, frame_idx]
        if np.isnan(box[:4]).any() or np.isnan(subject_scores).all():
            return

        strongest = int(np.nanargmax(subject_scores))
        color = label_color(strongest)
        top_left = np.round(box[:2]).astype(np.int32)
        cv2.rectangle(image, top_left, np.round(box[2:4]).astype(np.int32), color, thickness=2)
        cv2.putText(
            image,
            f"{labels[strongest]} {subject_scores[strongest]:.2f}",
            (int(top_left[0]), int(top_left[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    render_per_camera_videos(draw_subject, bbox_2d.axes, context, viz_folder)


def draw_eye_closure_scores(
    landmarks,
    scores,
    left_eye_indices: list[int],
    right_eye_indices: list[int],
    axes,
    context: RenderContext,
    viz_folder: str,
) -> None:
    """
    Draw eye contours and EAR scores on frames and stitch videos per camera.
    """
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    lm_data = np.asarray(landmarks.data if hasattr(landmarks, "data") else landmarks)
    sc_data = np.asarray(scores.data if hasattr(scores, "data") else scores)
    subjects = axes.subjects

    def draw_subject(image, subject_idx, cam_idx, frame_idx):
        subject_name = subjects[subject_idx]
        if subject_name not in axes.subjects:
            return
        color = colors[subject_idx % len(colors)]

        for eye_idx, indices in enumerate([left_eye_indices, right_eye_indices]):
            eye_coords = lm_data[subject_idx, cam_idx, frame_idx, indices, :2]
            valid_coords = eye_coords[~np.isnan(eye_coords).any(axis=1)]
            if len(valid_coords) == 0:
                continue

            x_min, y_min = np.min(valid_coords, axis=0)
            cv2.polylines(image, [valid_coords.astype(np.int32)], isClosed=True, color=color, thickness=1)
            for pt in valid_coords:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 1, color, -1)

            score = sc_data[subject_idx, cam_idx, frame_idx, eye_idx]
            cv2.putText(
                image,
                f"{score:.2f}",
                (int(x_min), int(y_min) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    render_per_camera_videos(draw_subject, axes, context, viz_folder)


def draw_eye_closed_states(
    landmarks,
    scores,
    eye_closed,
    left_eye_indices: list[int],
    right_eye_indices: list[int],
    axes,
    context: RenderContext,
    viz_folder: str,
) -> None:
    """
    Draw eye contours coloured open (green) / closed (red) with EAR scores and stitch videos per camera.
    """
    lm_data = np.asarray(landmarks.data if hasattr(landmarks, "data") else landmarks)
    sc_data = np.asarray(scores.data if hasattr(scores, "data") else scores)
    cl_data = np.asarray(eye_closed.data if hasattr(eye_closed, "data") else eye_closed)
    subjects = axes.subjects

    def draw_subject(image, subject_idx, cam_idx, frame_idx):
        subject_name = subjects[subject_idx]
        if subject_name not in axes.subjects:
            return

        for eye_idx, indices in enumerate([left_eye_indices, right_eye_indices]):
            eye_coords = lm_data[subject_idx, cam_idx, frame_idx, indices, :2]
            valid_coords = eye_coords[~np.isnan(eye_coords).any(axis=1)]
            if len(valid_coords) == 0:
                continue

            x_min, y_min = np.min(valid_coords, axis=0)
            closed_state = cl_data[subject_idx, cam_idx, frame_idx, eye_idx]
            color = (0, 0, 255) if closed_state == 1 else (0, 255, 0)

            cv2.polylines(image, [valid_coords.astype(np.int32)], isClosed=True, color=color, thickness=1)
            for pt in valid_coords:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 1, color, -1)

            score = sc_data[subject_idx, cam_idx, frame_idx, eye_idx]
            if np.isnan(score):
                continue
            cv2.putText(
                image,
                f"{score:.2f}",
                (int(x_min), int(y_min) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    render_per_camera_videos(draw_subject, axes, context, viz_folder)


def draw_gaze_interaction_arrows(
    heads: np.ndarray,
    gaze: np.ndarray,
    looks_at_anyone: np.ndarray,
    radii: np.ndarray,
    axes,
    context: RenderContext,
    viz_folder: str,
) -> None:
    """
    Draw per-subject gaze interaction arrows (green for look-at, red for no-look) and target circles.
    """
    color_look_at = (0, 200, 0)
    color_no_look = (0, 0, 255)
    color_target = (0, 200, 200)

    def draw_subject(image, subject_idx, cam_idx, frame_idx):
        head = heads[subject_idx, cam_idx, frame_idx]
        vec = gaze[subject_idx, cam_idx, frame_idx]
        looks = looks_at_anyone[subject_idx, cam_idx, frame_idx]
        if np.isnan(head).any() or np.isnan(vec).any() or np.isnan(looks):
            return

        arrow_length = vis_ut.gaze_arrow_pixel_length(image.shape[1])
        color = color_look_at if looks == 1 else color_no_look
        head_point = np.round(head).astype(np.int32)

        radius = radii[subject_idx, cam_idx, frame_idx]
        if not np.isnan(radius):
            cv2.circle(
                image,
                tuple(head_point),
                max(1, int(round(radius))),
                color=color_target,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        end_point = np.round(head + arrow_length * vec).astype(np.int32)
        cv2.arrowedLine(
            image,
            tuple(head_point),
            tuple(end_point),
            color=color,
            thickness=2,
            line_type=cv2.LINE_AA,
            tipLength=0.2,
        )

    render_per_camera_videos(draw_subject, axes, context, viz_folder)


def draw_gaze_multiview_fusion(
    gaze_2d: np.ndarray,
    origins: np.ndarray,
    per_view_2d: dict[str, np.ndarray],
    image_sizes: dict[str, tuple[int, int]],
    axes,
    context: RenderContext,
    viz_folder: str,
) -> None:
    """
    Draw per-view yellow gaze arrows and fused red gaze arrows per subject.
    """
    n_views = next(iter(per_view_2d.values())).shape[1] if per_view_2d else 0

    def draw_subject(img, sub_id, cam_idx, frame_idx):
        cam_name = axes.cameras[cam_idx]
        arrow_length = vis_ut.gaze_arrow_pixel_length(image_sizes[cam_name][0])

        vec = gaze_2d[sub_id, cam_idx, frame_idx]
        if np.isnan(vec).any():
            return

        origin = origins[sub_id, cam_idx, frame_idx]
        if np.isnan(origin).any():
            return
        origin_i = np.round(origin).astype(np.int32)

        for view_idx in range(n_views):
            view_vec = per_view_2d[cam_name][sub_id, view_idx, frame_idx]
            if np.isnan(view_vec).any():
                continue
            view_end = np.round(origin + 0.5 * arrow_length * view_vec).astype(np.int32)
            cv2.arrowedLine(
                img,
                origin_i,
                view_end,
                color=(0, 255, 255),
                thickness=1,
                line_type=cv2.LINE_AA,
                tipLength=0.2,
            )

        end_point = np.round(origin + arrow_length * vec).astype(np.int32)
        cv2.arrowedLine(
            img,
            origin_i,
            end_point,
            color=(0, 0, 255),
            thickness=2,
            line_type=cv2.LINE_AA,
            tipLength=0.2,
        )

    render_per_camera_videos(draw_subject, axes, context, viz_folder)
