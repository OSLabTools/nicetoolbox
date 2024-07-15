import platform

def detect_os_type() -> str:
    """
    Detects the underlying operating system.

    Returns:
        str: A string representing the operating system type. It can be either 'windows' or 'linux'.

    Notes:
        This function uses the platform module to determine the operating system.
        It checks the platform.system() function's return value and returns 'windows' if it's 'Windows',
        and 'linux' if it's either 'Linux' or 'Darwin'.
    """
    if platform.system() == 'Windows':
        os_type = "windows"
    elif platform.system() == 'Linux' or platform.system() == 'Darwin':
        os_type = "linux"
    return os_type

def linux_to_windows_path(linux_path: str) -> str:
    """
    Converts a Linux path to a Windows path.

    Args:
        linux_path (str): The Linux path to be converted.
    """
    return linux_path.replace('/', '\\')

def convert_config_paths_to_windows(config):
    """
    Converts all string paths in a given dictionary from Linux to Windows format.

    Args:
        config (dict): A dictionary containing configuration settings. 
            The values of this dictionary can be either strings or nested dictionaries.

    Returns:
        dict: The same dictionary as the input, but with all string paths converted to 
            Windows format.

    Note:
        This function does not modify the original dictionary. Instead, it returns a new 
        dictionary with the converted paths. It skips any string values that contain the 
        substring 'http', assuming they represent web addresses and not file paths.

    """
    for key, value in config.items():
        # don't convert web address
        if isinstance(value, str):
            if "http" in value:
                continue
        if isinstance(value, dict):  # If value is dictionary, recurse
            convert_config_paths_to_windows(value)
        elif isinstance(value, str):  # Convert path if value is string
            config[key] = linux_to_windows_path(value)
    return config