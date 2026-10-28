import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3d projection

from nicetoolbox_core.data.loaded_array import NpzArray

from .draw_2d import RenderContext, open_video_writer, subject_color


def _opencv_to_matplotlib(kpts: np.ndarray) -> np.ndarray:
    """OpenCV camera frame (x-right, y-down, z-forward) -> matplotlib world frame (x-right, y-forward, z-up)."""
    out = kpts[..., [0, 2, 1]].copy()
    out[..., 0] *= -1
    out[..., 2] *= -1
    return out


def _bgr_to_rgb(bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


def draw_skeleton_3d(
    keypoints_3d: NpzArray,
    connections: list[list[str]],
    context: RenderContext,
    viz_folder: str,
    elev: float = 15.0,
    azim: float = 70.0,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 100,
    marker_size: int = 20,
    linewidth: float = 2.0,
) -> None:
    """
    Render each subject's 3d skeleton per frame; one video per (camera, subject).

    Input frame is assumed OpenCV-camera-style (x-right, y-down, z-forward); axes are remapped
    to matplotlib's default so the person stands upright.

    Frames are streamed in memory directly into <viz_folder>/<camera>/subject_<idx>.mp4.
    If context.save_frames is True, frames are also saved to <viz_folder>/images/<camera>/subject_<idx>/.

    Args:
        keypoints_3d (NpzArray): (subjects, cameras, frames, keypoints, x/y/z[/conf]).
        connections (list[list[str]]): Polylines of label names to connect (same convention as draw_skeleton).
        context (RenderContext): Frame source and timing for this subsequence.
        viz_folder (str): Visualization folder of the component being rendered.
        elev, azim (float): Matplotlib 3d viewpoint in degrees.
        figsize, dpi: Matplotlib figure size / dpi.
        marker_size (int): Joint marker size.
        linewidth (float): Skeleton edge width.
    """
    data = _opencv_to_matplotlib(np.asarray(keypoints_3d.data)[..., :3])

    label_to_idx = {label: i for i, label in enumerate(keypoints_3d.axes.labels)}
    edges = [
        (label_to_idx[a], label_to_idx[b])
        for polyline in connections
        for a, b in zip(polyline, polyline[1:])
        if a in label_to_idx and b in label_to_idx
    ]

    n_frames = len(keypoints_3d.axes.frames)
    real_frame_indices = [real_idx for real_idx, _ in context.dataloader]
    if len(real_frame_indices) != n_frames:
        raise ValueError(
            f"draw_skeleton_3d: dataloader yielded {len(real_frame_indices)} frames but the array has {n_frames}."
        )

    cameras = list(keypoints_3d.axes.cameras)
    subjects = list(keypoints_3d.axes.subjects)
    for cam_idx, cam_name in enumerate(cameras):
        for subj_idx, subj_name in enumerate(subjects):
            # Fix view bounds from all valid frames of this (cam, subject) so the camera doesn't jitter.
            valid_pts = data[subj_idx, cam_idx].reshape(-1, 3)
            valid_pts = valid_pts[np.all(np.isfinite(valid_pts), axis=-1)]
            if valid_pts.size == 0:
                logging.info(f"draw_skeleton_3d: no valid frames for {cam_name}/{subj_name}, skipping.")
                continue

            mins, maxs = valid_pts.min(axis=0), valid_pts.max(axis=0)
            center = (mins + maxs) / 2.0
            half = max(float(np.max(maxs - mins)) / 2.0, 1e-6)
            xlim = (center[0] - half, center[0] + half)
            ylim = (center[1] - half, center[1] + half)
            zlim = (center[2] - half, center[2] + half)

            color = _bgr_to_rgb(subject_color(subj_idx))

            fig = plt.figure(figsize=figsize, dpi=dpi)
            width, height = fig.canvas.get_width_height()
            try:
                video_name = os.path.join(cam_name, subj_name)
                with open_video_writer(viz_folder, video_name, (width, height), context) as writer:
                    for frame_idx in range(n_frames):
                        fig.clear()
                        ax = fig.add_subplot(111, projection="3d")
                        ax.set_xlim(*xlim)
                        ax.set_ylim(*ylim)
                        ax.set_zlim(*zlim)
                        ax.set_box_aspect((1, 1, 1))
                        ax.view_init(elev=elev, azim=azim)
                        ax.grid(True)
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_zlabel("z")
                        ax.tick_params(labelsize=6)

                        points = data[subj_idx, cam_idx, frame_idx]
                        if not np.all(np.isnan(points)):
                            for start, end in edges:
                                pa, pb = points[start], points[end]
                                if np.isnan(pa).any() or np.isnan(pb).any():
                                    continue
                                ax.plot(
                                    [pa[0], pb[0]],
                                    [pa[1], pb[1]],
                                    [pa[2], pb[2]],
                                    color=color,
                                    linewidth=linewidth,
                                )
                            valid = np.all(np.isfinite(points), axis=-1)
                            if valid.any():
                                vp = points[valid]
                                ax.scatter(vp[:, 0], vp[:, 1], vp[:, 2], color=color, s=marker_size)

                        fig.canvas.draw()
                        # RGBA canvas -> BGR (the writer and saved frames expect OpenCV channel order)
                        bgr = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., 2::-1])
                        writer.write_frame(bgr, frame_idx=real_frame_indices[frame_idx])
            finally:
                plt.close(fig)

    logging.info(f"3d visualization finished: {viz_folder}")
