"""
Helper functions for input/output operations.
"""

import os
from pathlib import Path
import shutil
import glob
from contextlib import contextmanager


@contextmanager
def create_tmp_folder(tmp_folders: str | list):
    """
    Create temporary run folders and yield control to the caller.
    
    Args:
        tmp_folders (str or list): The path(s) of the temporary folders to be created.
            If a string is provided, a single temporary folder will be created.
            If a list of strings is provided, multiple temporary folders will be created.
    
    Yields:
        None: The control is returned to the caller, 
            allowing the caller to execute a block of code.
    
    Raises:
        OSError: If there is an error creating the temporary folders.
    
    """
    if isinstance(tmp_folders, str):
        tmp_folders = [tmp_folders]

    # create temporary run folders
    for tmp_folder in tmp_folders:
        os.makedirs(tmp_folder, exist_ok=True)
        
    try:
        yield
    finally: 
        # delete temporary run folders
        for tmp_folder in tmp_folders:
            shutil.rmtree(tmp_folder)


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


def list_directories_under_root(root_path: str) -> list:
    """
    Lists all the subdirectories under the given root path.

    Args:
        root_path (str): The root path to search for subdirectories.

    Returns:
        list: A list of subdirectory paths.

    Raises:
        None
    """
    rootdir = Path(root_path)
    pattern = '/**/'
    # list all subdirectories
    directory_list = [path for path in glob.glob(f'{rootdir}{pattern}', recursive=True) ]
    if directory_list == []:
        print('Could not find any folder. Please check root_path')

    return directory_list


def create_directory_tree(source: str, dest: str) -> None:
    """
    Gets the folder structure from source and creates the same folder structure under the destination root.

    Args:
        source (str): The source root.
        dest (str): The destination root.

    Returns:
        None. Creates empty folders under the given destination root.
    """
    dir_list = list_directories_under_root(source)
    for dir in dir_list:
        temp_dir = dir.replace(source, dest)
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

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
