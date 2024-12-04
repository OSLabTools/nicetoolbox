"""
Check and exception handling functions.
"""

import logging
import os

import numpy as np

from . import filehandling as fh


def check_token_in_filepath(folder_name: str, token: str, description: str) -> None:
    """
    Check if a given token is present in a folder name.

    Args:
        folder_name (str): The name of the folder to check.
        token (str): The token to search for in the folder name.
        description (str): A description of the folder for error messages.

    Raises:
        ValueError: If the token is not found in the folder name.

    Returns:
        None
    """
    if token not in folder_name:
        raise ValueError(
            f"The given {description} '{folder_name}' does not contain "
            f"the required token '{token}'."
        )


def check_options(object, object_type, options) -> None:
    """
    Check if an object is of a specific type and if it is within a given
    list of options.

    Args:
        object (any): The object to be checked.
        object_type (type): The expected type of the object.
        options (list): A list of valid options for the object.

    Raises:
        TypeError: If the object is not of the expected type.
        ValueError: If the object is not within the list of valid options.

    Returns:
        None
    """
    if not isinstance(object, object_type):
        raise TypeError(
            f"Expected object of type {object_type.__name__}, "
            f"got {type(object).__name__}"
        )
    if object not in options:
        raise ValueError(
            f"Object {object} is not in the list of valid options: {options}"
        )


def check_value_bounds(
    object, object_type=None, object_min=None, object_max=None
) -> None:
    """
    Check if an object's value falls within specified bounds.

    Args:
        object (any): The object to be checked.
        object_type (type, optional): The expected type of the object. Defaults to None.
        object_min (any, optional): The minimum value allowed for the object.
            Defaults to None.
        object_max (any, optional): The maximum value allowed for the object.
            Defaults to None.

    Raises:
        TypeError: If the object is not of the expected type (if object_type is
            provided).
        ValueError: If the object's value is less than object_min (if object_min
            is provided).
                If the object's value is greater than object_max (if object_max
                is provided).

    Returns:
        None
    """
    if object_type is not None and not isinstance(object, object_type):
        raise TypeError(
            f"Expected object of type {object_type.__name__}, "
            f"got {type(object).__name__}"
        )
    if object_min is not None and object < object_min:
        raise ValueError(
            f"Object value {object} is less than the minimum allowed value {object_min}"
        )
    if object_max is not None and object > object_max:
        raise ValueError(
            f"Object value {object} is greater than the maximum allowed "
            f"value {object_max}"
        )


def file_exists(file: str) -> None:
    """
    Check if a file exists at the given path.

    Args:
        file (str): The path to the file to check.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.

    Returns:
        None
    """
    if not os.path.exists(file):
        raise FileNotFoundError


def error_log_and_raise(error, name, message):
    """
    This function logs an error message and then raises a specific error with
    a formatted message.

    Args:
        error (Exception): The type of error to be raised.
        name (str): The name of the function or method where the error occurred.
        message (str): The detailed error message.

    Raises:
        error: The specific error type raised with a formatted error message.

    """
    logging.error(f"{name}: {error.__name__}. {message}")
    raise error(f"{name}: {message}")


def check_user_input_config(config, check, config_name, var=None):
    """
    Check a given configuration dictionary for correct user inputs
    based on a template dictionary describing the valid inputs.

    Args:
        config (dict): The config that contains all inputs to be checked.
        check (dict): The template specifying which inputs are valid for
            each dict key.
        config_name (str): The name of the config to be checked,
            used for more descriptive logs.

    Note:
        The keys of 'check' must include the keys of 'config'.
        Syntax of the dict values:
            'type:<str/int/bool/...>': specifies the valid data type
            'folder:<base/full>': requires existence of the folder (in case of 'full')
                                  or parent-folder (in case of 'base')
            'file': requires that the file is existing on the system
            'keys:<toml_filepath>': valid options are all keys from the dict given
                                    by the toml_filepath
            'tbd': not yet defined in the template dict 'check',
                   will write a warning to log
            [<>, ...]: list of valid options, may contain all basetypes
    """
    # the config dict contains all test-keys and test-values (tkey, tval)
    # the check dict contains all check-keys and check-values (ckey, cval)
    for tkey, tval in config.items():
        # CHECKING KEYS
        # given a definition of valid keys
        if "_valid_keys_" in check:
            check_user_input_config(
                {"test": tkey}, {"test": check["_valid_keys_"]}, config_name, var
            )

        # if a key serves as a variable
        if list(check.keys()) == ["_var_"] or set(["_var_", "_valid_keys_"]) == set(
            list(check.keys())
        ):
            var = tkey
            cval = check["_var_"]

        # if not, the validity criterion for its value(s) needs to be defined in the
        # check dictionary
        elif tkey not in check:
            error_log_and_raise(
                LookupError,
                config_name,
                f"'{tkey}' not found in within the keys of the check dictionary.",
            )
            break

        # if key is no variable and validity criterion for its value is given, use
        # this for checks
        else:
            cval = check[tkey]

        # CHECKING VALUES

        # special case: the test value is a list
        if isinstance(tval, list):
            if tval == []:
                pass
            else:
                for tvalue in tval:
                    check_user_input_config({tkey: tvalue}, check, config_name, var)

        # go through all options for the check-value (cval)
        elif isinstance(cval, str):
            # check whether there is a variable in the string to replace
            if "_var_" in cval:
                if var is not None:
                    cval = cval.replace("_var_", var)
                else:
                    error_log_and_raise(
                        LookupError,
                        config_name,
                        f"value '{cval}' requires the variable (_var_) "
                        "to be defined.",
                    )

            # type check
            if cval.startswith("type"):
                check_type = cval.split(":")[1]
                if not isinstance(tval, __builtins__[check_type]):
                    error_log_and_raise(
                        TypeError,
                        config_name,
                        f"Key '{tkey}' requires value of type {check_type}. "
                        f"The given value is '{tval}'.",
                    )

            elif cval.startswith("folder"):
                folder_details = cval.split(":")[1]
                if folder_details == "base":
                    folder = os.path.dirname(tval)
                elif folder_details == "full":
                    folder = tval
                else:
                    error_log_and_raise(
                        ValueError,
                        config_name,
                        f"Unknown check value {folder_details}. "
                        "Options are 'base' or 'full'.",
                    )

                if not os.path.isdir(folder):
                    error_log_and_raise(
                        NotADirectoryError,
                        config_name,
                        f"For key '{tkey}', the folder '{folder}' "
                        "is not a directory.",
                    )

            elif cval.startswith("file"):
                if not os.path.isfile(tval):
                    error_log_and_raise(
                        FileNotFoundError,
                        config_name,
                        f"For key {tkey}, the file '{tval}' is not found.",
                    )

            elif cval.startswith("keys"):
                keys = load_dict_keys_values(cval, config_name)
                check_user_input_config({tkey: tval}, {tkey: keys}, config_name, var)

            elif cval.startswith("values"):
                values = load_dict_keys_values(cval, config_name)
                check_user_input_config({tkey: tval}, {tkey: values}, config_name, var)

            elif cval.startswith("tbd"):
                logging.warning(
                    f"{config_name}: For key '{tkey}', no check argument is "
                    "given. Skipping."
                )

            else:
                error_log_and_raise(
                    NotImplementedError,
                    config_name,
                    f"Check option '{cval.split(':')[0]}' is unknown. Currently "
                    "supported options in strings are 'type', 'folder', "
                    "'file', 'keys', 'tbd'.",
                )

        # check given options
        elif isinstance(cval, list):
            if tval not in cval:
                error_log_and_raise(
                    ValueError,
                    config_name,
                    f"Key '{tkey}' can take values {cval}. "
                    f"The given value is '{tval}'.",
                )

        # recursive strategy for dicts
        elif isinstance(cval, dict):
            check_user_input_config(tval, cval, config_name, var)


def load_dict_keys_values(command: str, config_name: str) -> list:
    """
    Load keys or values from a .toml file based on the given command.

    Args:
        command (str): A string in the format 'token:file:key(s)' or 'token:file'.
            'token' should be either 'keys' or 'values'.
            'file' is the path to the .toml file.
            'key(s)' is an optional parameter specifying the nested keys in the
                .toml file.
        config_name (str): The name of the configuration for error logging.

    Returns:
        list: A list of keys or values from the .toml file based on the given command.

    Raises:
        NotImplementedError: If the file format is not supported.
        AssertionError: If the token is neither 'keys' nor 'values'.
    """
    has_sub_keys = len(command.split(":")) != 2
    if not has_sub_keys:
        token, file = command.split(":")
    else:
        token, file, sub_keys = command.split(":")

    # ensure that token == 'keys' or 'values'
    check_options(token, str, ["keys", "values"])

    if file.endswith(".toml"):
        loaded_dict = fh.load_config(file)
        if has_sub_keys:
            for sub_key in sub_keys.split("."):
                loaded_dict = loaded_dict[sub_key]
        # get keys or values of the loaded dict
        keys_values = loaded_dict.keys() if token == "keys" else loaded_dict
    else:
        error_log_and_raise(
            NotImplementedError,
            config_name,
            f"Check option '{token}:<>' currently only supports '.toml' files "
            f"to retrieve keys. Given argument: '{file}'",
        )

    return list(keys_values)


def check_zeros(arr: np.ndarray) -> None:
    """
    Check if any vectors in the last dimension of an array are zero vectors.

    Args
        arr (ndarray): The input array.

    Raises
        AssertionError: If there are any zero vectors found in the last
        dimension of the array.

    Examples
    >>> arr_3d_example = np.random.randint(0, 255, (10, 10, 3))
    >>> check_zeros(arr_3d_example)
    """

    # Determine the size of the last dimension from the input array
    last_dim_size = arr.shape[-1]

    zero_vector = np.zeros(last_dim_size)
    zero_cells = np.all(arr == zero_vector, axis=-1)

    if np.any(zero_cells):
        logging.warning(
            f" Warning. Data array contains zero cells at {np.argwhere(zero_cells)}"
        )
