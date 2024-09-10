"""
    Helper functions for reading, writing and parsing text files
"""
import os
import numpy as np
import json
import toml


def load_config(config_file: str) -> dict:
    """
    Load a configuration file in YAML or TOML format.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        dict: The configuration data loaded from the file.

    Raises:
        NotImplementedError: If the file type is not supported.

    Note:
        If the operating system is Windows, the paths in the configuration data will 
        be converted to Windows format.
    """

    if config_file.endswith('toml'):
        config = toml.load(config_file)
    else:
        raise NotImplementedError(
                f"config_file type {config_file} is not supported currently. "
                f"Implemented are yaml/yml and toml.")
    return config


def read_npz_file(filepath):
    """
    Reads and returns the data from an NPZ file.
    
    Supports loading data with the `allow_pickle=True` parameter.
    
    Parameters:
        filepath (str): The path to the NPZ file.
        
    Returns:
        data (numpy.ndarray): The data loaded from the NPZ file.
    """
    data = np.load(filepath, allow_pickle=True)
    return data


def find_npz_files(directory):
    """
    Recursively find all npz files in the given directory.
    
    Args:
        directory (str): The directory to search for npz files.
    
    Returns:
        npz_files (list): A list of paths to the npz files found in the directory.
    """
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(root, file))
    return npz_files


def load_toml(toml_path: str) -> dict:
    """
    Load TOML data from a file.

    Args:
        toml_path (str): The path to the TOML file.

    Returns:
        dict: The TOML data loaded from the file.
    """
    data = toml.load(toml_path)
    return data


def load_json_file(json_path: str) -> dict:
    """
    Load JSON data from a file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        dict: The JSON data loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON data.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
        return data