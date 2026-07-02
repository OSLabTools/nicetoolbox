"""
Helper functions for reading, writing and parsing files
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import toml


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
    for root, _dirs, files in os.walk(directory):
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
    with open(json_path) as file:
        data = json.load(file)
        return data


def resolve_single_file(path: Path, *, label: str) -> Path:
    """
    Resolve `path` to exactly one existing file.

    - If `path` contains a `*`, expand it as a glob; expect exactly one match.
    - Otherwise, expect the file to exist as-is.

    Zero matches or multiple matches raise ValueError. `label` is included in
    error messages to identify the track/source that produced the path.
    """
    path_str = str(path)
    if "*" in path_str:
        matches = [Path(p) for p in glob.glob(path_str)]
        matches = [p for p in matches if p.is_file()]
        if not matches:
            raise ValueError(f"{label}: no files match pattern '{path_str}'.")
        if len(matches) > 1:
            listing = ", ".join(str(m) for m in matches)
            raise ValueError(f"{label}: pattern '{path_str}' matches {len(matches)} files: {listing}")
        return matches[0]

    if not path.is_file():
        raise FileNotFoundError(f"{label}: file not found: '{path}'.")
    return path
