"""
Utility functions for visualizing and calculating eye closure scores (EAR method).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from ...utils.draw_2d import RenderContext, draw_eye_closure_scores


def resolve_eye_indices(face_indices: dict, eye_key: str) -> list[int]:
    """
    Resolves eye keypoint indices within sliced face landmarks.
    If keypoints are global whole-body IDs (e.g., COCO-WholeBody), maps them to their relative
    0-indexed positions within the face_landmarks subset.
    Otherwise returns the 0-indexed positions directly.
    """
    target_indices = face_indices[eye_key]
    if "face_landmarks" in face_indices:
        face_landmarks_all = face_indices["face_landmarks"]
        if all(idx in face_landmarks_all for idx in target_indices):
            return [face_landmarks_all.index(idx) for idx in target_indices]
    return target_indices


def calculate_ear(eye_coords, eye_layout) -> np.ndarray:
    """Calculates the Eye Aspect Ratio (EAR) for a given eye's coordinates.

    Args:
        eye_coords (np.ndarray): (..., n_points, 2) coordinates for one eye.
        eye_layout: EyeLayout naming the vertical pairs and horizontal pair, as
            positions within the eye's own point slice.

    Returns:
        np.ndarray: EAR of shape eye_coords.shape[:-2]. NaN where the score cannot be
        computed: one of the six measured points is missing, or the eye has zero width.
        Points the layout does not measure are ignored, even when missing.
    """
    v1 = eye_coords[..., eye_layout.vertical_pairs[0][0], :]
    v2 = eye_coords[..., eye_layout.vertical_pairs[0][1], :]
    v3 = eye_coords[..., eye_layout.vertical_pairs[1][0], :]
    v4 = eye_coords[..., eye_layout.vertical_pairs[1][1], :]
    h1 = eye_coords[..., eye_layout.horizontal_pair[0], :]
    h2 = eye_coords[..., eye_layout.horizontal_pair[1], :]

    dist_v1 = np.linalg.norm(v1 - v2, axis=-1)
    dist_v2 = np.linalg.norm(v3 - v4, axis=-1)
    dist_h = np.linalg.norm(h1 - h2, axis=-1)

    # Division by zero will result inf, so we replace zeroes with nan
    # In numpy x / nan == nan, which exactly what we want
    dist_h = np.where(dist_h == 0, np.nan, dist_h)

    # Calculate final metric
    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return ear


def visualize_eye_closure_scores(
    viz_folder,
    dataloader,
    subjects_descr,
    landmarks,
    landmarks_axes,
    eye_closure_score,
    left_eye_indices,
    right_eye_indices,
    fps,
    video_start_frame_index,
    save_frames,
) -> None:
    """Draw eye contours + EAR scores on each frame and stitch a per-camera video.

    Walks each frame per camera, extracts eye keypoints, computes left and right EAR,
    and draws eye boundary polygons along with EAR text annotations on the frames.
    """
    context = RenderContext(
        dataloader=dataloader,
        fps=int(fps),
        video_start=video_start_frame_index,
        save_frames=save_frames,
    )
    draw_eye_closure_scores(
        landmarks=landmarks,
        scores=eye_closure_score,
        left_eye_indices=left_eye_indices,
        right_eye_indices=right_eye_indices,
        context=context,
        viz_folder=viz_folder,
        axes=landmarks_axes,
    )

    # Generate EAR raw score line plot over time (subfigures for each subject in one figure per camera)
    n_subj = len(landmarks_axes.subjects)
    for cam_idx, camera_name in enumerate(landmarks_axes.cameras):
        fig, axes = plt.subplots(n_subj, 1, figsize=(12, 3.5 * n_subj), sharex=True, squeeze=False)
        for subject_idx in range(n_subj):
            ax = axes[subject_idx, 0]
            subject_name = subjects_descr[subject_idx]

            left_scores = eye_closure_score[subject_idx, cam_idx, :, 0]
            right_scores = eye_closure_score[subject_idx, cam_idx, :, 1]

            if np.isnan(left_scores).all() and np.isnan(right_scores).all():
                ax.set_title(f"EAR Score over Time - {subject_name} ({camera_name}) - Not visible")
                ax.grid(True, linestyle="--", alpha=0.5)
                continue

            ax.plot(range(len(left_scores)), left_scores, label="Left Eye", color="blue")
            ax.plot(range(len(right_scores)), right_scores, label="Right Eye", color="green")
            ax.set_ylabel("Eye Aspect Ratio (EAR)")
            ax.set_title(f"EAR Score over Time - {subject_name} ({camera_name})")
            ax.legend(loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.5)

        axes[-1, 0].set_xlabel("Frame Index")
        plt.tight_layout()

        plot_name = f"ear_score_{camera_name}.png"
        plt.savefig(os.path.join(viz_folder, plot_name), dpi=300, bbox_inches="tight")
        plt.close(fig)
