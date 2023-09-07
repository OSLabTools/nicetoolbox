"""
A template class for Detectors.
"""

import os
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Class to setup and run existing computer vision research code.
    """

    def __init__(self, config, io) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        config : dict
            the method-specific configurations dictionary
        io: class
            a class instance that handles in-output folders
        """
        # shared config entries
        self.input_data_format = config['input_data_format']
        self.camera_ids = config['camera_ids']

        # output folders for inference and visualizations
        self.viz_folder = io.get_output_folder(self.name, 'visualization')
        self.result_folder = io.get_output_folder(self.name, 'result')

    def __str__(self):
        """ description of the class instance for printing

        Returns
        -------
        str
            all attributes and their values, written in plain text
        """
        return f"Instance of class {self.name} \n\t" \
               f"behavior = {self.behavior} \n\t" + \
               " \n\t".join([f"{attr} = {value}"
                             for (attr, value) in self.__dict__.items()])

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def behavior(self):
        raise NotImplementedError

    @abstractmethod
    def inference(self, data):
        """Run inference of the method on the given data

        Should save the results as one of our 'shared data formats' in
        self.result_folder as .hdf5
        """
        pass

    @abstractmethod
    def visualization(self, data):
        """ Visualize the output of the method, preferably as a video

        Should save the visualization in the self.viz_folder
        """
        pass



