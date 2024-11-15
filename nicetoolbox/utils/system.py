"""
Helper functions for system-related operations.
"""

import platform


def detect_os_type() -> str:
    """
    Detects the underlying operating system.

    Returns:
        str: A string representing the operating system type. 
            It can be either 'windows' or 'linux'.

    Notes:
        This function uses the platform module to determine the operating system.
        It checks the platform.system() function's return value and returns 
        'windows' if it's 'Windows', and 'linux' if it's either 'Linux' or 'Darwin'.
    """
    if platform.system() == "Windows":
        os_type = "windows"
    elif platform.system() == "Linux" or platform.system() == "Darwin":
        os_type = "linux"
    return os_type
