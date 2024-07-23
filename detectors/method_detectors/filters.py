"""
Savitzy-Golay filter for smoothing pose data.
"""
import numpy as np
import scipy.signal as signal
from abc import ABC, abstractmethod

# TODO: Remove base class if not needed
class BaseFilter(ABC):
    """
    Class to apply filters to the 3D pose estimation results
    """

    def __init__(self):
        pass
    def apply(self, data):
        pass

class SGFilter(BaseFilter):
    """
    A class designed to apply a 1D Savitzky-Golay filter to smooth and/or differentiate data.

    The Savitzky-Golay filter is a digital filter that can smooth or differentiate a set of digital
    data points by fitting successive subsets of adjacent data points with a low-degree polynomial 
    by the method of linear least squares. This class encapsulates the functionality of the 
    Savitzky-Golay filter, allowing for easy application to data arrays. It is particularly useful 
    for smoothing noisy data while preserving features of the signal such as relative maxima, 
    minima, and width, which are usually flattened by other types of filters.

    Attributes:
        window_size (int): The length of the filter window (i.e., the number of coefficients). 
            `window_size` must be a positive odd integer. The size of the window will affect the 
            smoothness of the output signal, with larger windows providing smoother results but less 
            sensitivity to small variations in the input data.
        polyorder (int): The order of the polynomial used to fit the samples. `polyorder` must be 
            less than `window_size`. A higher polynomial order can fit the data more closely, but if 
            too high, it may lead to overfitting, causing artifacts in the filtered signal.

    References:
        - Savitzky, A., and Golay, M.J.E. (1964). Smoothing and Differentiation of Data by Simplified 
            Least Squares Procedures. Analytical Chemistry, 36(8), pp.1627-1639.
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    """

    def __init__(self, window_length, polyorder)-> None:
        """
        Initializes the Savitzky-Golay filter with the specified window length and polynomial order.
        """
        self.window_length = window_length
        self.polyorder = polyorder

    def apply(self, data, is_3d=False):
        """
        Applies the Savitzky-Golay filter to the input data.

        The filter is applied to each dimension (X, Y, (Z)) of each keypoint separately.
        The filtered data is stored in a new array.

        Args:
            data (np.array): The input data to be filtered. The shape of the array should be
                [#Frames, #Keypoints, XYZ].

            is_3d (bool, optional): A flag indicating whether the input data is 3D or not.
                If True, the filter will be applied to the Z dimension.
                If False, the filter will only be applied to the X and Y dimensions.
                Default is False.

        Returns:
            np.array: The filtered data. The shape of the array will be [#Frames, #Keypoints, XYZ].
        """
        # Create an empty array to store the filtered data
        smooth_data = np.copy(data)
        for person_idx in range(data.shape[0]):
            for camera_idx in range(data.shape[1]):
                # apply the Savitzky-Golay filter to each dimension (X, Y, (Z)) of each keypoint
                if is_3d:
                    num_dim = 3
                else:
                    num_dim = 2
                for dim in range(num_dim):
                    if len(smooth_data.shape) == 5:
                        for kp in range(data.shape[3]):  # Loop through labels
                            smooth_data[person_idx, camera_idx, :, kp, dim] = signal.savgol_filter(data[person_idx,camera_idx, :, kp, dim], window_length=self.window_length,
                                                                             polyorder=self.polyorder)
                    elif len(smooth_data.shape) == 4:
                        smooth_data[person_idx, camera_idx, :, dim] = signal.savgol_filter(data[person_idx,camera_idx, :, dim], window_length=self.window_length,
                                                                             polyorder=self.polyorder)
        return smooth_data
