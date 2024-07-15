"""

"""
import sys
import os
import logging
from contextlib import contextmanager
import shutil
from utils.config import config_fill_auto, save_config


def log_configs(configs: dict, out_folder: str, file_name: str = 'log_config') -> None:
    """
    Logs the configurations to a file.

    The function takes in a dictionary of configurations, an output folder path, and an optional
    file name. It fills in placeholders in the code_config dictionary with actual values using 
    the config_fill_auto function. Then, it constructs the log file path by joining the output 
    folder path and the file name with a .toml extension. After that, it updates the configs 
    dictionary with the filled code_config and saves the configurations to the log file using the
    save_config function.
    
    Args:
        configs (dict): A dictionary containing the configurations to be logged.
        out_folder (str): The path to the output folder where the log file will be saved.
        file_name (str, optional): The name of the log file. Defaults to 'log_config'.

    Returns:
        None
    """
    code_config = dict(
            user="<me>",
            git_hash="<git_hash>",
            commit_message="<commit_message>",
            date="<today>",
            time="<time>",
    )
    # fill placeholders
    code_config = config_fill_auto(code_config)

    file_name = config_fill_auto(dict(file=file_name))['file']
    config_log_file = os.path.join(out_folder, f"{file_name}.toml")

    configs.update(code_config=code_config)
    save_config(configs, config_log_file)


def setup_logging(log_path: str, level=logging.DEBUG) -> None:
    """
    Start logger.

    Args:
        log_path (str): The path to the log file.
        level (int, optional): Determines from which level the logger will record the messages.
            For instance, when the level is set as logging.INFO, the messages with a severity 
            below INFO (i.e. DEBUG) will be ignored.
            The possible levels are:
                - logging.DEBUG: Detailed information, typically of interest only when 
                diagnosing problems.
                - logging.INFO: Confirmation that things are working as expected.
                - logging.WARNING: An indication that something unexpected happened, or 
                indicative of some problem in the near future (e.g. 'disk space low').
                    The software is still working as expected.
                - logging.ERROR: Due to a more serious problem, the software has not been able 
                to perform some function.
                - logging.CRITICAL: A serious error, indicating that the program itself may be 
                unable to continue running.

    Returns:
        None

    """
    logging.basicConfig(
        filename=log_path, 
        level=level, 
        format='%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s'
    )

# def log_message(message, level="info"):
#     if level == "debug":
#         logging.debug(message)
#     elif level == "info":
#         logging.info(message)
#     elif level == "warning":
#         logging.warning(message)
#     elif level == "error":
#         logging.error(message)
#     elif level == "critical":
#         logging.critical(message)
#     else:
#         logging.info(message)  # Default to info if no valid level is provided

def assert_and_log(condition, message):
    """
    Asserts a condition and logs an error message if the condition is not met.

    Args:
        condition (bool): The condition to be checked.
        message (str): The error message to be logged if the condition is not met.

    Returns:
        None

    Raises:
        AssertionError: If the condition is not met.
        SystemExit: If the condition is not met, the function will terminate the program with a status code of 1.
    """
    try:
        assert condition, message
    except AssertionError as e:
        logging.error(f"Assertion failed: {e}")
        sys.exit(1)


class LoggingSetup:
    """
    A class that provides utilities for setting up logging.

    Args:
        tmp_log_path (str): The temporary path for the log file.
        log_level (int): The log level to be set.

    Attributes:
        tmp_log_path (str): The temporary path for the log file.
        log_level (int): The log level to be set.
        log_path (str): The final path for the log file.

    Methods:
        tmp_logging: A context manager that sets up logging and yields control.
        set_path: Sets the final path for the log file.
        set_level: Sets the log level.

    """

    def __init__(self, tmp_log_path, log_level):
        self.tmp_log_path = tmp_log_path
        self.log_level = log_level
        self.log_path = None

    @contextmanager
    def tmp_logging(self):
        """
        A context manager that sets up logging and yields control.

        Yields:
            None

        """
        setup_logging(self.tmp_log_path, level=self.log_level)
        logging.debug('\n\n\n')
        logging.debug(f"Start logging to path {self.tmp_log_path} with log "
                      f"level {self.log_level}.")

        try:
            yield
        finally:
            if self.log_path is not None:
                shutil.move(self.tmp_log_path, self.log_path)
                self.set_path()
                self.set_level()
            else:
                logging.warning("logging handler move not successful!")

    def set_path(self):
        """
        Sets the final path for the log file.

        Raises:
            AssertionError: If the number of logging handlers is not equal to 1.

        """
        log_handlers = logging.getLogger().handlers
        assert_and_log(len(log_handlers) == 1,
                       f'Found {len(log_handlers)} logging handlers, expected 1.')
        log_handlers[0].setStream(open(self.log_path, 'a'))

        if logging.getLogger().handlers[0].baseFilename == self.log_path:
            logging.debug(f"Successfully set the logging path to {self.log_path}.")
        else:
            logging.error(f"Failed to set the logging path to {self.log_path}.")

    def set_level(self):
        """
        Sets the log level.

        Raises:
            AssertionError: If the number of logging handlers is not equal to 1.

        """
        log_handlers = logging.getLogger().handlers
        assert_and_log(len(log_handlers) == 1,
                       f'Found {len(log_handlers)} logging handlers, expected 1.')
        log_handlers[0].setLevel(self.log_level)

        if logging.getLogger().handlers[0].level == self.log_level:
            logging.debug(f"Successfully set the log level to {self.log_level}.")
        else:
            logging.error(f"Failed to set the log level to {self.log_level}.")

