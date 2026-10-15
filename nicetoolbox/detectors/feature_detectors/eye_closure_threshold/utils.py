"""
Utility functions for visualizing eye closed states.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

from ...utils.draw_2d import RenderContext, draw_eye_closed_states


def plot_eye_closed_states(
    viz_folder,
    eye_closed,
    subjects_descr,
    camera_names,
    threshold=None,
) -> None:
    n_subj = len(subjects_descr)

    for cam_idx, camera_name in enumerate(camera_names):
        fig, axes = plt.subplots(n_subj, 1, figsize=(12, 3.5 * n_subj), sharex=True, squeeze=False)

        for subject_idx in range(n_subj):
            ax = axes[subject_idx, 0]
            subject_name = subjects_descr[subject_idx]

            # For real cameras, if every frame is NaN the subject was never observed here
            # the pseudo-camera "3d" holds the fused global state and always plots.
            if camera_name != "3d" and np.isnan(eye_closed[subject_idx, cam_idx, :, :]).all():
                ax.set_title(f"Eye Closed State - {subject_name} ({camera_name}) - Not visible")
                ax.grid(True, linestyle="--", alpha=0.5)
                continue

            n_channels = eye_closed.shape[-1]
            if n_channels == 3:
                left_eye_state = eye_closed[subject_idx, cam_idx, :, 0]
                right_eye_state = eye_closed[subject_idx, cam_idx, :, 1]
                both_eye_state = eye_closed[subject_idx, cam_idx, :, 2]

                ax.step(range(len(left_eye_state)), left_eye_state, label="Left Eye", color="blue", where="post")
                ax.step(range(len(right_eye_state)), right_eye_state, label="Right Eye", color="green", where="post")
                ax.step(
                    range(len(both_eye_state)),
                    both_eye_state,
                    label="Both Eyes",
                    color="red",
                    linestyle="--",
                    where="post",
                )

            ax.set_ylabel("Closed State")
            ax.set_ylim(-0.2, 1.2)

            title = f"Eye Closed State - {subject_name} ({camera_name})"
            if threshold is not None:
                title += f" (Threshold: {threshold})"
            ax.set_title(title)
            ax.legend(loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.5)

        axes[-1, 0].set_xlabel("Frame Index")
        plt.tight_layout()

        # Save plot to visualization folder
        save_path = os.path.join(
            viz_folder,
            f"eye_closed_state_{camera_name}.png",
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def visualize_eye_closed_videos(
    viz_folder,
    dataloader,
    landmarks,
    landmarks_axes,
    eye_closure_score,
    eye_closed,
    left_eye_indices,
    right_eye_indices,
    fps,
    video_start_frame_index,
    save_frames,
    cam_sees_subjects=None,
) -> None:
    """
    Creates annotated video outputs drawing eye bboxes, writing eye scores,
    and color-coding green if open (0) and red if closed (1).

    Args:
        landmarks (np.ndarray): upstream 2D face landmarks, (subjects, cameras, frames, landmarks, 2+).
        landmarks_axes (NpzArrayAxes): the landmarks' named axes (subject/camera order).
    """
    context = RenderContext(
        dataloader=dataloader,
        cam_sees_subjects=cam_sees_subjects,
        fps=int(fps),
        video_start=video_start_frame_index,
        save_frames=save_frames,
    )
    draw_eye_closed_states(
        landmarks=landmarks,
        scores=eye_closure_score,
        eye_closed=eye_closed,
        left_eye_indices=left_eye_indices,
        right_eye_indices=right_eye_indices,
        context=context,
        viz_folder=viz_folder,
        axes=landmarks_axes,
    )


def filter_duration(
    binary_array: np.ndarray, fps: float, min_duration: float, max_duration: float = None, axis: int = -1
) -> np.ndarray:
    """
    Filters contiguous runs of 1s in a binary array along the specified axis based on their duration in seconds.
    If a run meets the criteria, it is kept as 1. Otherwise, it is reset to 0.

    The array can be of any shape. Values must be exactly 0 or 1.
    """
    if binary_array.ndim == 1:
        flat = binary_array[np.newaxis, :]
    else:
        moved_array = np.moveaxis(binary_array, axis, -1)
        original_shape = moved_array.shape
        flat = moved_array.reshape(-1, original_shape[-1])

    # Connect adjacent components only horizontally (along the time/frame axis)
    structure = np.zeros((3, 3), dtype=int)
    structure[1, :] = 1

    labeled, num_features = ndimage.label(flat, structure=structure)
    if num_features == 0:
        return np.zeros_like(binary_array)

    sizes = ndimage.sum(np.ones_like(flat), labeled, range(1, num_features + 1))
    durations = sizes / fps
    keep = durations >= min_duration
    if max_duration is not None:
        keep = keep & (durations <= max_duration)

    map_array = np.zeros(num_features + 1, dtype=flat.dtype)
    map_array[1:] = np.where(keep, 1.0, 0.0)
    flat_filtered = map_array[labeled]

    if binary_array.ndim == 1:
        return flat_filtered[0]
    reshaped = flat_filtered.reshape(original_shape)
    return np.moveaxis(reshaped, -1, axis)
