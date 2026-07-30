"""
Utility functions for visualizing and calculating eye closure scores (EAR method).
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ....utils import video as vd


def resolve_eye_indices(face_indices: dict, eye_key: str) -> list[int]:
    """
    Resolves eye keypoint indices within sliced face landmarks.
    If keypoints are global whole-body IDs, maps them to their relative 0-indexed positions
    in the sliced face landmark array. Otherwise returns the 0-indexed positions directly.
    """
    target_indices = face_indices[eye_key]
    face_indices_all = []
    for val in face_indices.values():
        if isinstance(val, list):
            face_indices_all.extend(val)
        else:
            face_indices_all.append(val)

    if all(idx in face_indices_all for idx in target_indices):
        return [face_indices_all.index(idx) for idx in target_indices]
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
    camera_names,
    subjects_descr,
    cam_sees_subjects,
    landmarks,
    landmarks_axes,
    eye_closure_score,
    left_eye_indices,
    right_eye_indices,
    fps,
    video_start_frame_index,
) -> None:
    """Draw eye contours + EAR scores on each frame and stitch a per-camera video.

    Args:
        landmarks (np.ndarray): upstream 2D face landmarks, (subjects, cameras, frames, landmarks, 3).
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
                break

            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red

            for subject_idx in range(n_subj):
                subject_name = subjects_descr[subject_idx]
                if subject_idx not in cam_sees_subjects.get(camera_name, []):
                    continue
                if subject_name not in landmarks_subject_names:
                    continue
                landmarks_subject_idx = landmarks_subject_names.index(subject_name)

                color = colors[subject_idx % len(colors)]

                # Draw Left and Right eyes
                for eye_idx, indices in enumerate([left_eye_indices, right_eye_indices]):
                    eye_coords = landmarks[landmarks_subject_idx, landmarks_cam_idx, frame_idx, indices, :2]

                    valid_coords = eye_coords[~np.isnan(eye_coords).any(axis=1)]
                    if len(valid_coords) == 0:
                        continue

                    # Compute min coordinates for text positioning
                    x_min, y_min = np.min(valid_coords, axis=0)

                    # Draw eye landmarks contour and individual landmark points
                    cv2.polylines(image, [valid_coords.astype(np.int32)], isClosed=True, color=color, thickness=1)
                    for pt in valid_coords:
                        cv2.circle(image, (int(pt[0]), int(pt[1])), 1, color, -1)

                    # Write score on top of eye contour
                    score = eye_closure_score[subject_idx, cam_idx, frame_idx, eye_idx]
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

        # Generate EAR raw score line plot over time (subfigures for each subject in one figure per camera)
        fig, axes = plt.subplots(n_subj, 1, figsize=(12, 3.5 * n_subj), sharex=True, squeeze=False)
        for subject_idx in range(n_subj):
            ax = axes[subject_idx, 0]
            subject_name = subjects_descr[subject_idx]

            if subject_idx not in cam_sees_subjects.get(camera_name, []):
                ax.set_title(f"EAR Score over Time - {subject_name} ({camera_name}) - Not visible")
                ax.grid(True, linestyle="--", alpha=0.5)
                continue

            left_scores = eye_closure_score[subject_idx, cam_idx, :, 0]
            right_scores = eye_closure_score[subject_idx, cam_idx, :, 1]

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
