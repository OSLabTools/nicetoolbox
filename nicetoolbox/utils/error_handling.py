import logging
from contextlib import contextmanager

from nicetoolbox_core.errors import ErrorLevel, error_level_to_int


@contextmanager
def manage_error_scope(current_level: ErrorLevel, threshold_level: ErrorLevel, context_name: str):
    """
    Context manager that handles errors based on hierarchical error levels.

    - If the current_level is greater than or equal to the threshold_level, it suppresses exceptions,
      logs the error, and allows the outer loop to continue.
    - If the current_level is less than the threshold_level, it re-raises the exception for the outer loop.

    Args:
        current_level (ErrorLevel): The current error tolerance level (selected by the user in the config).
        threshold_level (ErrorLevel): The threshold level to compare against.
        context_name (str): Name of the current context for logging.

    Raises:
        Exception: Re-raises the caught exception if the current_level is less than the threshold_level
    """
    try:
        yield
    except Exception as e:
        logging.error(f"Failure in '{context_name}': {e}", exc_info=True)

        # Mathematical comparison works because of the IntEnum
        # If Config="DETECTOR" (20) and Threshold="VIDEO" (10) -> 20 >= 10 -> True
        if error_level_to_int[current_level.name] >= error_level_to_int[threshold_level.name]:
            logging.warning(
                f"\n\nSkipping '{context_name}' and continuing. (User-selected Error Level: {current_level.name}).\n\n"
            )
            return

        raise e
