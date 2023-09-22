"""
A template class for Detectors.
"""

import os
import logging
import data
from abc import ABC, abstractmethod
from oslab_utils.config import save_config
import oslab_utils.logging_utils as log_ut


class BaseFeature(ABC):
    """
    Class to setup and run follow-up computations. Input is always the output of any detector
    """

    def __init__(self, config, io) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        config : dict
            the feature-specific configurations dictionary
        io: class
            a class instance that handles in-output folders
        """

        # input folder of the feature is the result folder of detector
        self.input_folder = io.get_output_folder(config['input_name'], 'result')
        input_file_list = [f for f in os.listdir(self.input_folder) if ".hdf5" in f]
        log_ut.assert_and_log(len(input_file_list) != 0, f"Input file could not find.")
        log_ut.assert_and_log(len(input_file_list) == 1, f"There is more than one input file")
        self.input_file = os.path.join(self.input_folder, input_file_list[0])
        self.input_data_folder = config['input_data_folder']

        #output folders
        self.result_folder = io.get_output_folder(self.name, 'result')
        self.viz_folder = io.get_output_folder(self.name, 'visualization')

        # save this method config
        self.config_path = os.path.join(io.get_output_folder(self.name, 'tmp'),
                                        f'run_config.toml')

        save_config(config, self.config_path)
        logging.info(f"STARTING Computing - {self.name}")

    def __str__(self):
        """ description of the class instance for printing

        Returns
        -------
        str
            all attributes and their values, written in plain text
        """
        return (f"Instance of class {self.name} \n\t"
                f"behavior = {self.behavior} \n\t "
                f"\n\t").join([f"{attr} = {value}" for (attr, value) in self.__dict__.items()])

    def compute(self):
        """ Compute a feature based on a detector output.
        """
        pass

    def post_compute(self):
        """ Post-processing after computation
        """
        pass

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError


    @property
    @abstractmethod
    def behavior(self):
        raise NotImplementedError


    @abstractmethod
    def visualization(self, data):
        """ Visualize the output of the method, preferably as a video

        Should save the visualization in the self.viz_folder
        """
        pass




