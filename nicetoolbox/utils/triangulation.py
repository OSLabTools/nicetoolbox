"""
Helper functions for triangulation of 3D points from stereo images.
"""

import cv2


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
    undistorted = cv2.undistortPoints(
        point_coords, intrinsics, distortions, P=intrinsics
    )
    return undistorted


def triangulate_stereo(
    projection_matrix1, projection_matrix2, undistorted_points1, undistorted_points2
):
    """
    Triangulates the 3D position of points in Euclidean coordinates from two camera views.

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
