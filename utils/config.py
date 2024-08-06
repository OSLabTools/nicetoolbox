""" Handling configuration files, format: yaml

    for configs: supports using placeholders from two sources
    autofill:    <git_hash> <commit_message> <me> <today> <yyyymmdd> <time>
    path_config: <pis_folder_path> <local_input_path> <local_output_path>
"""

import os
import copy
import time
import yaml
import toml
import json
import numpy as np
import utils.system as os_sys
from utils.git_utils import CustomRepo


def default(obj):
    # serialize numpy array for saving to json
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def save_config(configs: dict, config_file: str) -> None:
    """
    Save the given configuration data to the specified file.

    Args:
        configs (dict): The configuration data to be saved.
        config_file (str): The path to the file where the configuration data will be saved.

    Raises:
        NotImplementedError: If the file type is not supported. 
            Supported types are yaml/yml and toml.

    Note:
        If the file type is Windows, it will convert the paths to Windows format.
    """

    if config_file.endswith('yml') or config_file.endswith('yaml'):
        with open(config_file, 'w') as file:
            yaml.dump_all(configs, file, default_flow_style=False, indent=4,
                          sort_keys=False)
    elif config_file.endswith('toml'):
        with open(config_file, 'w') as file:
            toml.dump(configs, file, encoder=toml.TomlNumpyEncoder())
    elif config_file.endswith('json'):
        with open(config_file, 'w') as file:
            json.dump(configs, file, default=default)
    else:
        raise NotImplementedError(
                f"config_file type {config_file} is not supported currently. "
                f"Implemented are yaml/yml and toml.")


def config_fill_placeholders(config, placeholders):
    """
    Replace placeholders in the given configuration data with their corresponding values.

    Args:
        config (dict or list): The configuration data to be processed.
        placeholders (dict): A dictionary containing the placeholder values.

    Returns:
        dict: The configuration data with placeholders replaced.
    """

    filled_config = copy.deepcopy(config)

    # iterate over all placeholders
    for ph_key, ph_value in placeholders.items():

        for config_key, config_value in filled_config.items():
            if isinstance(config_value, dict):
                filled_config[config_key] = config_fill_placeholders(
                        config_value, placeholders)
            elif isinstance(config_value, list):
                if any([f"<{ph_key}>" in val if isinstance(val, str) else False
                        for val in config_value]):
                    filled_config[config_key] = [
                        val.replace(f"<{ph_key}>", ph_value)
                        if (isinstance(val, str) and f"<{ph_key}>" in val)
                        else val for val in config_value]
            elif isinstance(config_value, str) and f"<{ph_key}>" in config_value:
                filled_config[config_key] = config_value.replace(
                        f"<{ph_key}>", f"{ph_value}")
            else:
                continue

    return filled_config


def config_fill_auto(config, working_directory=None):
    """
    This function fills placeholders in the configuration data with auto-generated values.

    Args:
        config (dict): The configuration data to be processed.
        working_directory (str, optional): The directory where the git repository is located. 
            Defaults to the current working directory.

    Returns:
        dict: The configuration data with placeholders replaced with auto-generated values.

    Note:
        This function uses the `oslab_utils.git_utils` module to get the git hash and commit 
        message. The placeholders filled are: <git_hash>, <commit_message>, <me>, <yyyymmdd>, 
        <today>, and <time>.
    """
    if working_directory is None:
        working_directory = os.getcwd()
    repo = CustomRepo(working_directory)
    git_hash, commit_message = repo.get_git_hash()

    placeholder_dict = dict(
            git_hash=git_hash[:7],
            commit_message=commit_message,
            me=os.getlogin(),
            yyyymmdd=time.strftime("%Y%m%d", time.localtime()),
            today=time.strftime("%Y%m%d", time.localtime()),
            time=time.strftime("%H_%M", time.localtime()),
            #pwd=working_directory,
    )

    return config_fill_placeholders(config, placeholder_dict)

