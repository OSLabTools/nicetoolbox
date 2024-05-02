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

    def apply(self, data, is_3d=False):
        """
        Takes data as an input and apply filter to each dimension of each keypoint separately.

        Parameters
        ----------
        data: (np array, shape - [#Frames, #Keypoints, XYZ]

        Returns:
        smoothed data(np array, shape - [#Frames, #Keypoints, XYZ]
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
                    for kp in range(data.shape[3]):  # Loop through keypoints
                        smooth_data[person_idx, camera_idx, :, kp, dim] = signal.savgol_filter(data[person_idx,camera_idx, :, kp, dim], window_length=self.window_length,
                                                                         polyorder=self.polyorder)
                return smooth_data
