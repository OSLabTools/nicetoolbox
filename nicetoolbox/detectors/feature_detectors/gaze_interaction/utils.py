"""
Utility functions for visualizing gaze interaction data.
"""

import logging
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ....utils import video as vd
from ....utils import visual_utils as vis_ut

# BGR colors for the per-subject gaze arrow.
_COLOR_LOOK_AT = (0, 200, 0)  # green: this subject looks at someone this frame
_COLOR_NO_LOOK = (0, 0, 255)  # red: not looking at anyone this frame
_COLOR_TARGET = (0, 200, 200)  # yellow: the look-at target zone (radius = threshold)


def visualize_gaze_arrows(
    heads,
    gaze,
    looks_at_anyone,
    radii,
    dataloader_config,
    camera_names,
    subjects_descr,
    output_folder,
    fps,
) -> None:
    """Render a per-camera video overlaying each subject's gaze arrow on the source frames.

    For every frame and camera, draws an arrow from each subject's head along their image-plane
    gaze direction, colored green when that subject looks at another subject this frame and red
    otherwise (mirroring the fused-gaze arrow viz, with detection color added). A circle of the
    per-head pixel radius in `radii` is drawn around each head — the target zone the gaze line must
    pass within to count as a look-at. In 2D that radius is the constant pixel threshold; in 3D it
    is the projected silhouette of the metric threshold sphere, so it shrinks with depth.

    Args:
        heads (ndarray): 2D head anchors, image coords, shape (subjects, cameras, frames, 2).
        gaze (ndarray): unit image-plane gaze direction vectors (u, v), same shape as heads. They
            are scaled to a pixel arrow length per camera, derived from the drawn frame's width.
        looks_at_anyone (ndarray): per-subject bool (as float 0/1/nan), whether the subject looks
            at any other subject, shape (subjects, cameras, frames).
        radii (ndarray): per-head target-circle radius in pixels, shape (subjects, cameras, frames).
        dataloader_config: video input recipe for ImagePathsByFrameIndexLoader.
        camera_names (list): cameras carried on axis1 of the arrays, in order.
        subjects_descr (list): subject names on axis0 (only used for length here).
        output_folder (str): visualization folder; one <camera>/ subfolder + <camera>.mp4 each.
        fps (float): output video frame rate.
    """
    dataloader = ImagePathsByFrameIndexLoader(dataloader_config, expected_cameras=camera_names)
    num_subjects = len(subjects_descr)

    for frame_idx, (real_idx, files) in enumerate(dataloader):
        for cam_name, path in files.items():
            if cam_name not in camera_names:
                continue
            cam_idx = camera_names.index(cam_name)

            img = cv2.imread(str(path))
            if img is None:
                continue

            # Gaze arrives as a unit direction; scale it to a visible arrow for this frame width.
            arrow_length = vis_ut.gaze_arrow_pixel_length(img.shape[1])

            for sub_id in range(num_subjects):
                head = heads[sub_id, cam_idx, frame_idx]
                vec = gaze[sub_id, cam_idx, frame_idx]
                looks = looks_at_anyone[sub_id, cam_idx, frame_idx]
                # Skip subjects with no head, no gaze, or no look_at result this frame.
                if np.isnan(head).any() or np.isnan(vec).any() or np.isnan(looks):
                    continue

                color = _COLOR_LOOK_AT if looks == 1 else _COLOR_NO_LOOK
                head_point = np.round(head).astype(np.int32)

                # Target zone: gaze counts as a look-at if it passes within this pixel radius of a head.
                radius = radii[sub_id, cam_idx, frame_idx]
                if not np.isnan(radius):
                    cv2.circle(
                        img,
                        tuple(head_point),
                        max(1, int(round(radius))),
                        color=_COLOR_TARGET,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                end_point = np.round(head + arrow_length * vec).astype(np.int32)
                cv2.arrowedLine(
                    img,
                    tuple(head_point),
                    tuple(end_point),
                    color=color,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                    tipLength=0.2,
                )

            out_dir = os.path.join(output_folder, cam_name)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{real_idx:09d}.jpg"), img)

    for cam_name in camera_names:
        vd.frames_to_video(
            os.path.join(output_folder, cam_name),
            os.path.join(output_folder, f"{cam_name}.mp4"),
            fps=fps,
            start_frame=int(dataloader.start),
        )
    logging.info(f"Saved per-camera gaze arrow videos to '{output_folder}'.")


def _per_subject(matrix: np.ndarray) -> np.ndarray:
    """Collapse a (subjects, cam, frames, subjects) interaction matrix to one value per looking
    subject: the mean over the other subjects (the NaN self-diagonal is ignored)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(matrix, axis=-1, keepdims=True)


def visualize_gaze_interaction(distances, look_at, mutual, output_folder, people_names, camera_names) -> None:
    """
    Visualizes the gaze interaction results across frames for multiple people and cameras.

    Saves one PNG per camera, with one subplot per person: the continuous gaze distance to
    the other person's head on the left axis, and the boolean look_at / mutual markers on a
    secondary axis.

    Args:
        distances, look_at, mutual (ndarray): The subject x subject interaction matrices of
            shape (subjects, cameras, frames, subjects); each is collapsed to one value per
            looking subject (mean over the others, ignoring the NaN self-diagonal).
        output_folder (str): Directory where the per-camera plots are saved.
        people_names (list): Name for each person (one per subplot).
        camera_names (list): Name for each camera (one PNG each). The "3d" slot switches the
            axis units to real-world.

    Returns:
        None
    """
    categories_list = ["distance_gaze", "gaze_look_at", "gaze_mutual"]
    # Collapse each matrix to (subjects, cam, frames, 1), then stack into the (…, 3) plot input.
    data = np.concatenate((_per_subject(distances), _per_subject(look_at), _per_subject(mutual)), axis=-1)
    num_people = len(data)

    for camera_idx, camera_name in enumerate(camera_names):
        unit = "(in pixels)" if camera_name != "3d" else "(real-world units: m/cm/mm)"
        _, axs = plt.subplots(num_people, 1, figsize=(10, 15))
        # Ensure axs is a list in case num_people is 1
        if num_people == 1:
            axs = [axs]
        for i, (ax, dat) in enumerate(zip(axs, data)):
            # create secondary axis for boolean values
            ax2 = ax.twinx()
            # A camera may not see a person's head (all-NaN distance row); fall back to a small
            # positive max so the axis limits below stay finite.
            distance_row = dat[camera_idx, :, 0]
            gaze_distance_max = np.nanmax(distance_row) if not np.isnan(distance_row).all() else 1.0
            for j, category in enumerate(categories_list):
                # first category is gaze distance and it is continuous
                if j == 0:
                    ax.plot(
                        dat[camera_idx, :, j],
                        label=f"{category} (left axis)",
                        color="tab:blue",
                    )

                # other categories are boolean
                elif j == 1 or j == 2:
                    x = (np.arange(dat.shape[1]),)
                    y = (dat[camera_idx, :, j],)
                    ax2.scatter(
                        x,
                        y,
                        label=f"{category} (right axis)",
                        marker="_" if j == 1 else ".",
                        color="tab:orange" if j == 1 else "tab:green",
                        alpha=0.5,
                        zorder=2 if j == 1 else 1,
                        s=1 if gaze_distance_max > 100 else 5,
                    )

            ax.set_title(f"Gaze Interaction Across Frames - {people_names[i]}")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel(f"Gaze Distance to the Other Person Head \n {unit}")
            ax2.set_ylabel("True/False (1/0)")

            if gaze_distance_max < 5:
                increment = 0.2
            elif gaze_distance_max < 100:
                increment = 2
            else:
                increment = 20

            # Set specific ticks for axis
            ax.set_ylim(-0.1, gaze_distance_max + increment)
            yticks = np.arange(0, gaze_distance_max + increment, increment)
            ax.set_yticks(yticks)
            ax2.set_yticks([0, 1])
            ax2.set_ylim(-0.1, 1.1)
            # Add legends from both axes
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(
                lines + lines2,
                labels + labels2,
                loc="upper left",
                bbox_to_anchor=(1.03, 1),
            )

        # Save one plot per camera.
        plt.subplots_adjust(right=0.85)
        plt.savefig(
            os.path.join(output_folder, f"gaze_interaction_{camera_name}.png"),
            bbox_inches="tight",
            dpi=500,
        )
