"""
Utility functions for visualizing eye closed states.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

from ....utils import video as vd


def plot_eye_closed_states(
    viz_folder,
    eye_closed,
    subjects_descr,
    camera_names,
    cam_sees_subjects,
    threshold=None,
) -> None:
    n_subj = len(subjects_descr)

    for cam_idx, camera_name in enumerate(camera_names):
        fig, axes = plt.subplots(n_subj, 1, figsize=(12, 3.5 * n_subj), sharex=True, squeeze=False)

        for subject_idx in range(n_subj):
            ax = axes[subject_idx, 0]
            subject_name = subjects_descr[subject_idx]

            # If it is a camera view, check visibility. Skip for pseudo-camera "3d"
            if camera_name != "3d" and subject_idx not in cam_sees_subjects.get(camera_name, []):
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
    camera_names,
    subjects_descr,
    cam_sees_subjects,
    landmarks,
    landmarks_axes,
    eye_closure_score,
    eye_closed,
    left_eye_indices,
    right_eye_indices,
    fps,
    video_start_frame_index,
) -> None:
    """
    Creates annotated video outputs drawing eye bboxes, writing eye scores,
    and color-coding green if open (0) and red if closed (1).

    Args:
        landmarks (np.ndarray): upstream 2D face landmarks, (subjects, cameras, frames, landmarks, 2+).
        landmarks_axes (NpzArrayAxes): the landmarks' named axes (subject/camera order).
    """
    landmarks_camera_names = landmarks_axes.cameras
    landmarks_subject_names = landmarks_axes.subjects

    n_subj = len(subjects_descr)

    for cam_idx, camera_name in enumerate(camera_names):
        if camera_name not in landmarks_camera_names:
            continue
        landmarks_cam_idx = landmarks_camera_names.index(camera_name)

        cam_viz_dir = os.path.join(viz_folder, camera_name)
        os.makedirs(cam_viz_dir, exist_ok=True)

        for frame_idx, (real_frame_idx, frame_paths_per_camera) in enumerate(dataloader):
            image_path = frame_paths_per_camera.get(camera_name)
            if not image_path:
                continue
            image = cv2.imread(image_path)
            if image is None:
                continue

            for subject_idx in range(n_subj):
                subject_name = subjects_descr[subject_idx]
                if subject_idx not in cam_sees_subjects.get(camera_name, []):
                    continue
                if subject_name not in landmarks_subject_names:
                    continue
                landmarks_subject_idx = landmarks_subject_names.index(subject_name)

                # Draw Left and Right eyes
                for eye_idx, indices in enumerate([left_eye_indices, right_eye_indices]):
                    eye_coords = landmarks[landmarks_subject_idx, landmarks_cam_idx, frame_idx, indices, :2]

                    valid_coords = eye_coords[~np.isnan(eye_coords).any(axis=1)]
                    if len(valid_coords) == 0:
                        continue

                    # Compute min coordinates for text positioning
                    x_min, y_min = np.min(valid_coords, axis=0)

                    # State-based Color coding: Red if closed (1), Green if open (0)
                    closed_state = eye_closed[subject_idx, cam_idx, frame_idx, eye_idx]
                    color = (0, 0, 255) if closed_state == 1 else (0, 255, 0)

                    # Draw eye landmarks contour and individual landmark points
                    cv2.polylines(image, [valid_coords.astype(np.int32)], isClosed=True, color=color, thickness=1)
                    for pt in valid_coords:
                        cv2.circle(image, (int(pt[0]), int(pt[1])), 1, color, -1)

                    # Write score on top of eye contour
                    score = eye_closure_score[subject_idx, cam_idx, frame_idx, eye_idx]
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

            cv2.imwrite(os.path.join(cam_viz_dir, f"{real_frame_idx:09d}.jpg"), image)

        # Compile frames into video
        vd.frames_to_video(
            cam_viz_dir,
            os.path.join(viz_folder, f"{camera_name}.mp4"),
            fps=fps,
            start_frame=int(video_start_frame_index),
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
