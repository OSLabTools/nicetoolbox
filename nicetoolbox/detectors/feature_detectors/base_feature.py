"""
A template class for Detectors.
"""

import os
from abc import ABC, abstractmethod

from ...utils import logging_utils as log_ut
from ...utils.config import save_config


class BaseFeature(ABC):
    """
    Abstract class to setup and run follow-up computations, called features detectors. 
    Input is always the output of any method detector.

    Attributes:
        input_folders (list): A list of input folders.
        input_files (list): A list of input files.
        result_folders (dict): A dictionary of result folders.
        out_folder (str): The output folder.
        viz_folder (str): The visualization folder.
        subjects_descr (str): The subjects description.
        config_path (str): The path to the configuration file.
    """

    def __init__(self, config, io, data, requires_out_folder=True) -> None:
        """
        Sets up the input and output folders based on the provided configurations
        and handles any necessary file checks. Input folders contain the the results
        of the method detectors. Saves a copy of the configuration file for the feature 
        detector.

        Args:
            config (dict): The feature-specific configurations dictionary.
            io (class): A class instance that handles input and output folders.
            data (class): A class instance that contains the data.
            requires_out_folder (bool, optional): Whether the output folder is required.
                Defaults to True.

        Returns:
            None
        """
        # input folder of the feature is the result folder of detector
        self.input_folders, self.input_files = [], []
        for comp, alg in config["input_detector_names"]:
            input_folder = io.get_detector_output_folder(comp, alg, "result")
            input_file = os.path.join(input_folder, f"{alg}.npz")
            if not os.path.isfile(input_file):
                raise FileNotFoundError(
                    f"Feature detector {self.components}: File '{input_file}' does "
                    "not exist!"
                )
            self.input_folders.append(input_folder)
            self.input_files.append(input_file)

        # output folders
        self.result_folders = dict(
            (comp, io.get_detector_output_folder(comp, self.algorithm, "result"))
            for comp in self.components
        )
        main_component = self.components[0]
        if requires_out_folder:
            self.out_folder = io.get_detector_output_folder(
                main_component, self.algorithm, "output"
            )
        if config["visualize"]:
            self.viz_folder = io.get_detector_output_folder(
                main_component, self.algorithm, "visualization"
            )

        self.subjects_descr = data.subjects_descr

        # save this method config
        self.config_path = os.path.join(
            io.get_detector_output_folder(main_component, self.algorithm, "run_config"),
            "run_config.toml",
        )
        save_config(config, self.config_path)

    def get_input(self, input_list: list, token: str, listdir: bool = True) -> str:
        """
        This method is used to find a specific file in a list of files based on a token.

        Args:
            input_list (str or list): The path to the directory or a list of files.
            token (str): The token to search for in the file names.
            listdir (bool, optional): If True, the input_list is treated as a directory
            and its contents are listed. Defaults to True.

        Returns:
            str: The file that contains the token.

        Raises:
            AssertionError: If no file is found or if more than one file is found.
        """
        if listdir:
            input_list = sorted(os.listdir(input_list))
        detected = [f for f in input_list if token in f]
        log_ut.assert_and_log(len(detected) != 0, "Input file could not find.")
        log_ut.assert_and_log(len(detected) == 1, "There is more than one input file")
        return detected[0]

    def __str__(self):
        """
        Returns a description of the feature detector for printing.

        Returns:
            str: A string representation of the feature detector, including its 
                components, and the associated algorithm.
        """
        return (
            f"Instance of component {self.components} \n\t"
            f"algorithm = {self.algorithm} \n\t "
            f"\n\t"
        ).join([f"{attr} = {value}" for (attr, value) in self.__dict__.items()])

    @abstractmethod
    def compute(self):
        """
        Compute the components assiciated to the given feature detector.

        This method is responsible for performing the main computation logic of the 
        feature detector. It should take the method detector output as input, process 
        it, and generate the desired components.
        """
        pass

    @abstractmethod
    def post_compute(self):
        """
        Post-processing after computation.

        This method is intended to perform any necessary post-processing tasks
        after the main computation method (compute) has been executed. It is
        designed to be overridden in derived classes to provide specific
        post-processing logic.
        """
        pass

    @property
    @abstractmethod
    def components(self):
        """
        Abstract property that returns the components of the feature.

        This property should be implemented in the derived classes to specify the 
        components that the feature detector is associated with.

        Returns:
            list: A list of strings representing the components associated with the 
                feature detector.

        Raises:
            NotImplementedError: If the property is not set in the derived classes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def algorithm(self):
        """
        Abstract property that returns the algorithm of the feature detector.

        This property should be implemented in the derived classes to specify the 
        algorithm that the feature detector is associated with.

        Returns:
            str: A string representing the algorithm associated with the feature 
                detector.

        Raises:
            NotImplementedError: If the property is not set in the derived classes.
        """
        raise NotImplementedError

    @abstractmethod
    def visualization(self, data):
        """
        Abstract method to visualize the output of the method, preferably as a video.

        This method is intended to generate a visual representation of the feature 
        detector's output. The visualization should be saved in the self.viz_folder.

        Args:
            data (any): The data to be visualized. The type and content of this 
                parameter depend on the specific implementation of the feature detector.

        Returns:
            None: This method does not return any value. However, it should save the 
                visualization in the self.viz_folder.

        Raises:
            NotImplementedError: If this method is not implemented in the derived 
                classes.
        """
        pass
