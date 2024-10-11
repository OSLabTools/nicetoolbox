"""
Helper functions for input/output operations.
"""

import os
from pathlib import Path
import glob


def list_files_under_root(root_path: str, ext: str = '') -> list:
    """
    Lists all files under a given root.

    Args:
        root_path (str): The root path.
        ext (str, optional): Extension of the requested file type. Default is an empty string.

    Returns:
        list: If ext parameter is an empty string, lists all files under the given root.
              Otherwise, lists the files with the given extension under the given root.
    """
    rootdir = Path(root_path)
    if ext:
        if not ext.startswith('.'):
            ext = f".{ext}"
        pattern = '/**/*' + ext
    else:
        pattern = '/**'
    # list all files under all subdirectories
    file_list = [path for path in glob.glob(f'{rootdir}{pattern}', recursive=True) if os.path.isfile(path)]

    return file_list


def delete_files_into_list(filepath_list: list) -> None:
    """
    Deletes the files specified in the given list of filepaths.

    Args:
        filepath_list (list): A list of filepaths to be deleted.

    Returns:
        None

    Raises:
        Exception: If the file cannot be deleted.
    """
    for filepath in filepath_list:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)  # Attempt to delete the file
            except Exception as e:
                print(f"Failed to delete {filepath}. Reason: {e}")
