"""
Linear algebra utility functions.
"""

import numpy as np
from .logging_utils import assert_and_log


def distance_line_point(line_point, line_direction, point):
    """
    Calculate the Euclidean distance from a point to a line in n-dimensional space.

    Args:
        line_point (numpy.ndarray): A 2D array representing the coordinates of
            a point on the line.
        line_direction (numpy.ndarray): A 2D array representing the direction
            vector of the line.
        point (numpy.ndarray): A 2D array representing the coordinates of the point.

    Returns:
        numpy.ndarray: A 2D array containing the Euclidean distance from each
            point to the corresponding line.

    Raises:
        AssertionError: If the shapes of line_point, line_direction, or point
            are inconsistent.
    """
    assert_and_log(
        point.shape[-1] in [1, 2, 3], f"invalid point dimensions: {point.shape}"
    )
    assert_and_log(
        line_direction.shape == line_point.shape == point.shape,
        f"inconsistent shapes inputted to distance_line_point",
    )

    scalarprod = (line_direction * (point - line_point)).sum(axis=-1, keepdims=True)
    norm2 = (line_direction * line_direction).sum(axis=-1, keepdims=True)
    factor = scalarprod / norm2

    distance_vec = point - line_point - line_direction * factor
    distance2 = (distance_vec * distance_vec).sum(axis=-1, keepdims=True)
    return np.sqrt(distance2)
