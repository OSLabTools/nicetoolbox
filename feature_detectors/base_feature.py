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

    def __init__(self, config, io, data, requires_out_folder=True) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        config : dict
            the feature-specific configurations dictionary
        io: class
            a class instance that handles in-output folders
        """
        # input folder of the feature is the result folder of detector
        self.input_folders, self.input_files = [], []
        for (comp, alg) in config['input_detector_names']:
            input_folder = io.get_detector_output_folder(comp, alg, 'result')
            input_file = os.path.join(input_folder, f"{alg}.npz")
            if not os.path.isfile(input_file):
                raise FileNotFoundError(f"Feature detector {self.components}: File '{input_file}' does not exist!")
            self.input_folders.append(input_folder)
            self.input_files.append(input_file)

        #output folders
        self.result_folders = dict((comp, io.get_detector_output_folder(comp, self.algorithm, 'result')) 
                                   for comp in self.components)
        main_component = self.components[0]
        if requires_out_folder:
            self.out_folder = io.get_detector_output_folder(main_component, self.algorithm, 'output')
        if config['visualize']:
            self.viz_folder = io.get_detector_output_folder(main_component, self.algorithm, 'visualization')
        
        self.subjects_descr = data.subjects_descr

        # save this method config
        self.config_path = os.path.join(
            io.get_detector_output_folder(main_component, self.algorithm, 'run_config'), 
            'run_config.toml'
            )
        save_config(config, self.config_path)

    def get_input(self, input_list, token, listdir=True):
        if listdir:
            input_list = sorted(os.listdir(input_list))
        detected = [f for f in input_list if token in f]
        log_ut.assert_and_log(len(detected) != 0,
                              f"Input file could not find.")
        log_ut.assert_and_log(len(detected) == 1,
                              f"There is more than one input file")
        return detected[0]

    def __str__(self):
        """ description of the class instance for printing

        Returns
        -------
        str
            all attributes and their values, written in plain text
        """
        return (f"Instance of component {self.components} \n\t"
                f"algorithm = {self.algorithm} \n\t "
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
    def components(self):
        raise NotImplementedError


    @property
    @abstractmethod
    def algorithm(self):
        raise NotImplementedError


    @abstractmethod
    def visualization(self, data):
        """ Visualize the output of the method, preferably as a video

        Should save the visualization in the self.viz_folder
        """
        pass




