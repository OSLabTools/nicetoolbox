"""
Check and exception handling functions.
"""

import logging
import os

import numpy as np

from . import filehandling as fh


def check_token_in_filepath(folder_name: str, token: str, description: str) -> None:
    """
    Check if a given token is present in a folder name.

    Args:
        folder_name (str): The name of the folder to check.
        token (str): The token to search for in the folder name.
        description (str): A description of the folder for error messages.

    Raises:
        ValueError: If the token is not found in the folder name.

    Returns:
        None
    """
    if token not in folder_name:
        raise ValueError(
            f"The given {description} '{folder_name}' does not contain "
            f"the required token '{token}'."
        )


def check_options(object, object_type, options) -> None:
    """
    Check if an object is of a specific type and if it is within a given
    list of options.

    Args:
        object (any): The object to be checked.
        object_type (type): The expected type of the object.
        options (list): A list of valid options for the object.

    Raises:
        TypeError: If the object is not of the expected type.
        ValueError: If the object is not within the list of valid options.

    Returns:
        None
    """
    if not isinstance(object, object_type):
        raise TypeError(
            f"Expected object of type {object_type.__name__}, "
            f"got {type(object).__name__}"
        )
    if object not in options:
        raise ValueError(
            f"Object {object} is not in the list of valid options: {options}"
        )


def check_value_bounds(
    object, object_type=None, object_min=None, object_max=None
) -> None:
    """
    Check if an object's value falls within specified bounds.

    Args:
        object (any): The object to be checked.
        object_type (type, optional): The expected type of the object. Defaults to None.
        object_min (any, optional): The minimum value allowed for the object.
            Defaults to None.
        object_max (any, optional): The maximum value allowed for the object.
            Defaults to None.

    Raises:
        TypeError: If the object is not of the expected type (if object_type is
            provided).
        ValueError: If the object's value is less than object_min (if object_min
            is provided).
                If the object's value is greater than object_max (if object_max
                is provided).

    Returns:
        None
    """
    if object_type is not None and not isinstance(object, object_type):
        raise TypeError(
            f"Expected object of type {object_type.__name__}, "
            f"got {type(object).__name__}"
        )
    if object_min is not None and object < object_min:
        raise ValueError(
            f"Object value {object} is less than the minimum allowed value {object_min}"
        )
    if object_max is not None and object > object_max:
        raise ValueError(
            f"Object value {object} is greater than the maximum allowed "
            f"value {object_max}"
        )


def file_exists(file: str) -> None:
    """
    Check if a file exists at the given path.

    Args:
        file (str): The path to the file to check.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.

    Returns:
        None
    """
    if not os.path.exists(file):
        raise FileNotFoundError


def error_log_and_raise(error, name, message):
    """
    This function logs an error message and then raises a specific error with
    a formatted message.

    Args:
        error (Exception): The type of error to be raised.
        name (str): The name of the function or method where the error occurred.
        message (str): The detailed error message.

    Raises:
        error: The specific error type raised with a formatted error message.

    """
    logging.error(f"{name}: {error.__name__}. {message}")
    raise error(f"{name}: {message}")


def check_zeros(arr: np.ndarray) -> None:
    """
    Check if any vectors in the last dimension of an array are zero vectors.

    Args
        arr (ndarray): The input array.

    Raises
        AssertionError: If there are any zero vectors found in the last
        dimension of the array.

    Examples
    >>> arr_3d_example = np.random.randint(0, 255, (10, 10, 3))
    >>> check_zeros(arr_3d_example)
    """

    # Determine the size of the last dimension from the input array
    last_dim_size = arr.shape[-1]

    zero_vector = np.zeros(last_dim_size)
    zero_cells = np.all(arr == zero_vector, axis=-1)

    if np.any(zero_cells):
        logging.warning(
            f" Warning. Data array contains zero cells at {np.argwhere(zero_cells)}"
        )
