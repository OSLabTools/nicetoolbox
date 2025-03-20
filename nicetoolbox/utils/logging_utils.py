"""
Helper functions for logging.
"""

import logging
import os
import sys

from .config import config_fill_auto, save_config


def log_configs(configs: dict, out_folder: str, file_name: str = "log_config") -> None:
    """
    Logs the configurations to a file.

    The function takes in a dictionary of configurations, an output folder path, and an
    optional file name. It fills in placeholders in the code_config dictionary with
    actual values using the config_fill_auto function. Then, it constructs the log file
    path by joining the output folder path and the file name with a .toml extension.
    After that, it updates the configs dictionary with the filled code_config and saves
    the configurations to the log file using the save_config function.

    Args:
        configs (dict): A dictionary containing the configurations to be logged.
        out_folder (str): The path to the output folder where the log file will be
            saved.
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

    file_name = config_fill_auto(dict(file=file_name))["file"]
    config_log_file = os.path.join(out_folder, f"{file_name}.toml")

    configs.update(code_config=code_config)
    save_config(configs, config_log_file)


def setup_logging(log_path: str, level=logging.DEBUG) -> None:
    """
    Start logger.

    Args:
        log_path (str): The path to the log file.
        level (int, optional): Determines from which level the logger will record the
            messages.
            For instance, when the level is set as logging.INFO, the messages with a
            severity below INFO (i.e. DEBUG) will be ignored.
            The possible levels are:
                - logging.DEBUG: Detailed information, typically of interest only when
                diagnosing problems.
                - logging.INFO: Confirmation that things are working as expected.
                - logging.WARNING: An indication that something unexpected happened, or
                indicative of some problem in the near future (e.g. 'disk space low').
                    The software is still working as expected.
                - logging.ERROR: Due to a more serious problem, the software has not
                been able to perform some function.
                - logging.CRITICAL: A serious error, indicating that the program itself
                may be unable to continue running.

    Returns:
        None

    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def setup_custom_logging(log_path: str, name: str, level=logging.DEBUG) -> None:
    """
    Start logger.

    Args:
        log_path (str): The path to the log file.
        level (int, optional): Determines from which level the logger will record the
            messages.
            For instance, when the level is set as logging.INFO, the messages with a
            severity below INFO (i.e. DEBUG) will be ignored.
            The possible levels are:
                - logging.DEBUG: Detailed information, typically of interest only when
                diagnosing problems.
                - logging.INFO: Confirmation that things are working as expected.
                - logging.WARNING: An indication that something unexpected happened, or
                indicative of some problem in the near future (e.g. 'disk space low').
                    The software is still working as expected.
                - logging.ERROR: Due to a more serious problem, the software has not
                been able to perform some function.
                - logging.CRITICAL: A serious error, indicating that the program itself
                may be unable to continue running.

    Returns:
        None

    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s"
    )

    # create file and console handler
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler(sys.stdout)

    # add formatter to handlers
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(sh)


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
        SystemExit: If the condition is not met, the function will terminate the
            program with a status code of 1.
    """
    try:
        assert condition, message
    except AssertionError as e:
        logging.error(f"Assertion failed: {e}")
        sys.exit(1)
