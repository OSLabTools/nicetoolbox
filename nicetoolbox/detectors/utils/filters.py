"""
Savitzky-Golay filter for smoothing pose data.

Also exposes adaptive_savgol_filter for generic arrays: same Savitzky-Golay
math with an odd window length capped to the extent along a chosen axis (used
by SAM 3D Body post-processing on stacked MHR parameters).
"""

import numpy as np
from scipy import signal


def adaptive_savgol_filter(
    data: np.ndarray,
    window_length: int,
    polyorder: int,
    *,
    axis: int = 0,
    mode: str = "interp",
) -> np.ndarray:
    """
    Apply a 1D Savitzky-Golay filter along one axis of a multi-dimensional array.

    SciPy applies the filter independently along 1D slices parallel to axis.
    Before calling SciPy, the effective window length is reduced so it does not
    exceed the slice length, forced odd, and at least 3; polyorder is capped
    by window_length - 1. Empty arrays or window_length < 3 are returned
    unchanged.

    Args:
        data: Input array (any shape); cast to float64 for filtering.
        window_length: Requested Savitzky-Golay window length (will be capped and
            adjusted to an odd value ≤ slice length along axis).
        polyorder: Polynomial order (capped so it is < effective window length).
        axis: Axis interpreted as time (or the dimension along which to smooth).
        mode: Passed to scipy.signal.savgol_filter (default "interp").

    Returns:
        Filtered array, same shape as data (float64).

    References:
        - Savitzky, A., and Golay, M.J.E. (1964). Smoothing and Differentiation of Data
          by Simplified Least Squares Procedures. Analytical Chemistry, 36(8),
          pp.1627-1639.
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    """
    x = np.asarray(data, dtype=np.float64)
    if x.size == 0 or window_length < 3:
        return x
    wl = min(window_length, x.shape[axis])
    if wl % 2 == 0:
        wl -= 1
    if wl < 3:
        return x
    po = min(polyorder, wl - 1)
    return signal.savgol_filter(x, wl, po, axis=axis, mode=mode)


class SGFilter:
    """
    A class designed to apply a 1D Savitzky-Golay filter to smooth and/or differentiate
    data.

    The Savitzky-Golay filter is a digital filter that can smooth or differentiate a
    set of digital data points by fitting successive subsets of adjacent data points
    with a low-degree polynomial by the method of linear least squares. This class
    encapsulates the functionality of the Savitzky-Golay filter, allowing for easy
    application to data arrays. It is particularly useful for smoothing noisy data
    while preserving features of the signal such as relative maxima, minima, and width,
    which are usually flattened by other types of filters.

    Implementation note: filtering is vectorized along the frame dimension using
    scipy.signal.savgol_filter(..., axis=2) on each coordinate slab, rather than
    nested Python loops over person, camera, and keypoint.

    Attributes:
        window_length (int): The length of the filter window (i.e., the number of
            coefficients). window_length should be a positive odd integer for the
            intended design; SciPy may adjust the effective length when a series is
            shorter than the requested window. The size of the window affects the
            smoothness of the output signal, with larger windows providing smoother
            results but less sensitivity to small variations in the input data.
        polyorder (int): The order of the polynomial used to fit the samples.
            polyorder must be less than the effective window length. A higher
            polynomial order can fit the data more closely, but if too high, it may
            lead to overfitting, causing artifacts in the filtered signal.

    References:
        - Savitzky, A., and Golay, M.J.E. (1964). Smoothing and Differentiation of Data
          by Simplified Least Squares Procedures. Analytical Chemistry, 36(8),
          pp.1627-1639.
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    """

    def __init__(self, window_length, polyorder) -> None:
        """
        Initializes the Savitzky-Golay filter with the specified window length and
        polynomial order.
        """
        self.window_length = window_length
        self.polyorder = polyorder

    def apply(self, data, is_3d=False):
        """
        Applies the Savitzky-Golay filter to the input data.

        The filter is applied to each dimension (X, Y, (Z)) of each keypoint separately
        along time (the frame axis, index 2). The filtered data is stored in a
        new array.

        Args:
            data (np.array): The input data to be filtered. The shape of the array
                should be [#Persons, #Cameras, #Frames, #Keypoints, XYZ] for 5D data,
                or [#Persons, #Cameras, #Frames, XY] for 4D data.

            is_3d (bool, optional): A flag indicating whether the input data is 3D
                or not.
                If True, the filter will be applied to the Z dimension.
                If False, the filter will only be applied to the X and Y dimensions.
                Default is False.

        Returns:
            np.array: The filtered data, same shape as data input parameter.
        """
        smooth_data = np.copy(data)
        num_dim = 3 if is_3d else 2
        # Frame axis is index 2 for (P, C, F, K, D) and (P, C, F, D).
        if len(data.shape) == 5 or len(data.shape) == 4:
            for dim in range(num_dim):
                smooth_data[..., dim] = signal.savgol_filter(
                    data[..., dim],
                    window_length=self.window_length,
                    polyorder=self.polyorder,
                    axis=2,
                )
        return smooth_data
