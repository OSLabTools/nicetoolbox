"""
Helper functions for triangulation of 3D points from stereo images.
"""

import cv2
import numpy as np


def undistort_points_pinhole(point_coords, intrinsics, distortions):
    """
    Undistorts the given points based on intrinsics and distortions parameters.

    Args:
        point_coords (np.array): The x and y values of the points.
        intrinsics (np.array): The intrinsics matrix.
        distortions (np.array): The distortions.

    Returns:
        np.array: The undistorted 2D point coordinates.
    """
    undistorted = cv2.undistortPoints(point_coords, intrinsics, distortions, P=intrinsics)
    return undistorted


def triangulate_stereo(projection_matrix1, projection_matrix2, undistorted_points1, undistorted_points2):
    """
    Triangulates the 3D position of points in Euclidean coordinates from two camera
    views.

    Args:
        projection_matrix1 (np.array): Projection matrix of the first camera.
        projection_matrix2 (np.array): Projection matrix of the second camera.
        undistorted_points1 (np.array): Coordinates of undistorted points from camera 1.
        undistorted_points2 (np.array): Coordinates of undistorted points from camera 2.

    Returns:
        np.array: The 3D position of points in Euclidean coordinates.
    """
    # Triangulate the 3D point from the two camera views
    X_homogeneous = cv2.triangulatePoints(
        projection_matrix1, projection_matrix2, undistorted_points1, undistorted_points2
    )
    # Convert the 3D point from homogeneous coordinates to Euclidean coordinates
    X_euclidean = X_homogeneous / X_homogeneous[3]
    # Return 3D coordinates of the point in world coordinates
    return X_euclidean[:3]


def project_points_to_camera(world_points, projection_matrix):
    """
    Projects world-space 3D points into a camera's image plane.

    The inverse of triangulate_stereo: applies the camera's projection matrix and the
    perspective divide. Lens distortion is not re-applied, so the result is in the same
    undistorted image space that undistort_points_pinhole produces - points triangulated
    from undistorted coordinates reproject onto their own inputs.

    Points behind the camera (non-positive depth) are returned as NaN: the perspective
    divide would otherwise yield a plausible-looking coordinate for a point that the
    camera cannot see. NaN inputs propagate to NaN outputs.

    Args:
        world_points (np.array): 3D points, shape (..., 3).
        projection_matrix (np.array): The camera's 3x4 projection matrix.

    Returns:
        np.array: Image coordinates, shape (..., 2).
    """
    world_points = np.asarray(world_points, dtype=float)
    projection_matrix = np.asarray(projection_matrix, dtype=float)

    homogeneous = np.concatenate([world_points, np.ones((*world_points.shape[:-1], 1))], axis=-1)
    image_points = homogeneous @ projection_matrix.T

    depth = image_points[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = image_points[..., :2] / depth[..., None]
    return np.where(depth[..., None] > 0, uv, np.nan)
