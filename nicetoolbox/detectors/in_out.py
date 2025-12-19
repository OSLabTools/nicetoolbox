"""
IO module for the NICE toolbox.
"""

import copy
import logging
import os

from ..utils import check_and_exception as exc


class IO:
    """
    The IO class handles input/output operations and folder management for the
    NICE toolbox.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Attributes:
        out_folder (str): The path to the output folder.
        log_level (str): The log level.
        out_sub_folder (str): The path to the output sub-folder.
        data_folder (str): The path to the data folder.
        data_input_folder (str): The path to the input data folder.
        calibration_file (str): The path to the calibration file.
        conda_path (str): The path to the Conda installation directory.
        algorithm_names (list): A list of algorithm names.
        detector_out_folder (str): The path to the detector output folder.
        detector_visualization_folder (str): The path to the detector visualization
            folder.
        detector_additional_output_folder (str): The path to the detector additional
            output folder.
        detector_run_config_path (str): The path to the detector run configuration file.
        detector_final_result_folder (str): The path to the detector final result
            folder.
    """

    def __init__(self, config):
        """
        Initializes IO by creating the output folder and setting the log level.

        Args:
            config (dict): A dictionary containing configuration parameters.

        Raises:
            OSError: If there is an error creating the output folder or the base data
                folder.

        """
        # create folders
        self.out_folder = config["out_folder"]
        try:
            os.makedirs(self.out_folder, exist_ok=True)
        except OSError:
            logging.exception("Failed creating the output folder.")
            raise

        try:
            os.makedirs(os.path.dirname(config["data_folder"]), exist_ok=True)
        except OSError:
            logging.exception(
                f"Failed creating the base data folder {config['data_folder']}."
            )
            raise

        self.log_level = config["log_level"]

    def get_log_file_level(self):
        """
        Returns the path of the log file and the log level.

        Returns:
            tuple: A tuple containing the path of the log file and the log level.
        """
        return os.path.join(self.out_folder, "nicetoolbox.log"), self.log_level

    def get_config_file(self):
        """
        Returns the path of the configuration file.

        Returns:
            str: The path of the configuration file.
        """
        return self.out_folder

    def initialization(self, config, algorithm_names):
        """
        Initializes the necessary variables and folders for data processing.

        Args:
            config (dict): A dictionary containing the configuration parameters.
            algorithm_names (list): A list of algorithm names.

        Returns:
            None
        """

        self.out_sub_folder = config["out_sub_folder"]
        if config["process_data_to"] == "data_folder":
            self.data_folder = config["data_folder"]
        self.create_folders()

        # check the given io-config
        self.check_config(config)

        # get the relevant config entries
        self.data_input_folder = config["data_input_folder"]
        self.calibration_file = config["path_to_calibrations"]
        self.conda_path = config["conda_path"]
        self.csv_folder = config["csv_out_folder"]

        self.algorithm_names = algorithm_names
        self.detector_out_folder = config["detector_out_folder"]
        self.detector_visualization_folder = config["detector_visualization_folder"]
        self.detector_additional_output_folder = config[
            "detector_additional_output_folder"
        ]
        self.detector_run_config_path = config["detector_run_config_path"]
        self.detector_final_result_folder = config["detector_final_result_folder"]

    def get_input_folder(self):
        """
        Returns the input folder path.

        Returns:
            str: The path to the input folder.
        """
        return self.data_input_folder

    def get_calibration_file(self):
        """
        Returns the calibration file path.

        Returns:
            str: The path of the calibration file.
        """
        return self.calibration_file

    def get_data_folder(self):
        """
        Returns the data folder associated with the current instance.

        Returns:
            str: The path to the data folder.
        """
        return self.data_folder

    def get_conda_path(self):
        """
        Returns the path to the Conda installation directory.

        Returns:
            str: The path to the Conda installation directory.
        """
        return self.conda_path

    def get_output_folder(self, token):
        """
        Returns the output folder based on the given name and token.

        Args:
            token (str): The token specifying the type of output folder
            ('tmp', 'output', 'main', or 'csv').

        Returns:
            str: The path to the output folder.

        Raises:
            NotImplementedError: If the token is not 'tmp', 'output', 'main', or 'csv'.
        """
        if token == "output":
            return self.out_sub_folder
        if token == "main":
            return self.out_folder
        if token == "csv":
            os.makedirs(self.csv_folder, exist_ok=True)
            return self.csv_folder
        raise NotImplementedError(
            f"IO return output folder: Token '{token}' unknown! "
            f"Supported are 'tmp', 'output', 'main', 'csv'."
        )

    def get_detector_output_folder(self, component, algorithm, token):
        """
        Get the output folder path for a specific component, algorithm, and token.

        Args:
            component (str): The name of the component.
            algorithm (str): The name of the algorithm.
            token (str): The token indicating the type of output folder.

        Returns:
            str: The path of the output folder.

        Raises:
            NotImplementedError: If the algorithm or token is not supported.

        """
        if not any([m_name in algorithm for m_name in self.algorithm_names]):
            raise NotImplementedError(
                f"IO return component output folder: Algorithm '{algorithm}' unknown! "
                f"Supported are: {self.algorithm_names}."
            )

        if token == "output":
            folder_name = copy.deepcopy(self.detector_out_folder)
        elif token == "visualization":
            folder_name = copy.deepcopy(self.detector_visualization_folder)
        elif token == "additional":
            folder_name = copy.deepcopy(self.detector_additional_output_folder)
        elif token == "result":
            folder_name = copy.deepcopy(self.detector_final_result_folder)
        elif token == "run_config":
            folder_name = copy.deepcopy(self.detector_run_config_path)
        else:
            raise NotImplementedError(
                f"IO return output folder: Token '{token}' unknown! "
                f"Supported are 'output', 'visualization', "
                f"'additional', 'tmp', 'result'."
            )

        folder_name = folder_name.replace("<cur_component_name>", component).replace(
            "<cur_algorithm_name>", algorithm
        )
        os.makedirs(folder_name, exist_ok=True)
        return folder_name

    def create_folders(self):
        """
        Creates the necessary output and data folders.

        This method creates the output folder and data folder if they don't already
        exist. If the folders cannot be created, an exception is raised.

        Raises:
            OSError: If the output folder or data folder cannot be created.

        """
        # create output folders
        try:
            os.makedirs(self.out_sub_folder, exist_ok=True)
        except OSError:
            logging.exception("Failed creating the output folder.")
            raise
        # create the data folders
        try:
            os.makedirs(self.data_folder, exist_ok=True)
        except OSError:
            logging.exception("Failed creating the data folder.")
            raise

    def check_config(self, config):
        """
        Check the validity of the given configuration.

        Args:
            config (dict): The configuration dictionary.

        Raises:
            TypeError: If the 'process_data_to' value is not a string.
            ValueError: If the 'process_data_to' value is not 'data_folder'.
            ValueError: If any of the detector input folders are invalid.
            OSError: If any of the detector input folders are not accessible.

        """
        # check the config['process_data_to'] input
        try:
            exc.check_options(config["process_data_to"], str, ["data_folder"])
        except (TypeError, ValueError):
            logging.exception(
                "Unsupported 'process_data_to' in io. " "Valid options: 'data_folder'."
            )
            raise

        # check all given detector output folders
        def check_base_folder(folder_name, token, description):
            try:
                exc.check_token_in_filepath(folder_name, token, description)
            except ValueError:
                logging.exception("The given detector input folder is invalid.")
                raise

            base = folder_name.split(token)[0][:-1]
            try:
                _ = sorted(os.listdir(base))
            except OSError:
                logging.exception(f"'{base}' is not an accessible directory.")
                raise

        check_base_folder(
            config["detector_out_folder"], "<cur_component_name>", "detector_out_folder"
        )
        check_base_folder(
            config["detector_visualization_folder"],
            "<cur_component_name>",
            "detector_visualization_folder",
        )
        check_base_folder(
            config["detector_additional_output_folder"],
            "<cur_component_name>",
            "detector_additional_output_folder",
        )
        check_base_folder(
            config["detector_final_result_folder"],
            "<cur_component_name>",
            "detector_final_result_folder",
        )
