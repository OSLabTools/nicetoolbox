"""
    Functions to check created data
"""
import sys
import numpy as np
import logging



def check_zeros(arr):
    """
    Check if any vectors in the last dimension of an array are zero vectors.

    Parameters
    ----------
    arr : ndarray
        The input array.

    Raises
    ------
    AssertionError
        If there are any zero vectors found in the last dimension of the array.

    Examples
    --------
    >>> arr_3d_example = np.random.randint(0, 255, (10, 10, 3))
    >>> check_zeros(arr_3d_example)
    """

    # Determine the size of the last dimension from the input array
    last_dim_size = arr.shape[-1]

    zero_vector = np.zeros(last_dim_size)
    zero_cells = np.all(arr == zero_vector, axis=-1)

    try:
        assert not np.any(zero_cells), f"Data array contains zero cells at {np.argwhere(zero_cells)}"
    except  AssertionError as e:
        logging.error(f"Assertion failed: {e}")
        sys.exit(1)


