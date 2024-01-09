import numpy as np
import scipy.signal as signal
from abc import ABC, abstractmethod

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
    Class to apply 1D savgol filter
    savgol_filter :https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html.

    Parameters
    ----------
    window_size (float): The length of the filter window. window_length must be a positive odd integer.

    polyorder (int):The order of the polynomial used to fit the samples. polyorder must be less than window_length.

    """

    def __init__(self, window_length, polyorder)-> None:

        self.window_length = window_length
        self.polyorder = polyorder

    def apply(self, data):
        """
        Takes data as an input and apply filter to each dimension of each keypoint separately.

        Parameters
        ----------
        data: (np array, shape - [#Frames, #Keypoints, XYZ]

        Returns:
        smoothed data(np array, shape - [#Frames, #Keypoints, XYZ]
        """
        # Create an empty array to store the filtered data
        smooth_data = np.zeros(data.shape)

        # apply the Savitzky-Golay filter to each dimension (X, Y, Z) of each keypoint
        for dim in range(data.shape[-1]):
            for kp in range(data.shape[1]):  # Loop through keypoints
                smooth_data[:, kp, dim] = signal.savgol_filter(data[:, kp, dim], window_length=self.window_length,
                                                                 polyorder=2)
        return smooth_data
