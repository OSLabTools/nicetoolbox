# very barebone config handling
from pathlib import Path
from pprint import pformat
from typing import Optional, Type, TypeVar

import pydantic
import toml
from pydantic import BaseModel


class ConfigValidationError(Exception):
    """
    Custom exception for configuration validation errors.
    Provides better formatting for Pydantic validation errors.
    """

    def __init__(self, error: pydantic.ValidationError, filepath: Optional[Path] = None):
        message = f"Config validation error in {filepath.name}\n" if filepath else ""
        for err in error.errors():
            message += "=" * 40 + "\n"
            error_path = ".".join(str(loc) for loc in err["loc"])
            message += f"{err['msg']}: '{error_path}'.\n"
            message += pformat(err["input"], depth=2) + "\n"

        super().__init__(message)


def load_config(config_file: str) -> dict:
    """
    Load a configuration file in TOML format.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        dict: The configuration data loaded from the file.

    Raises:
        IOError: When an array with no valid (existing).
        FileNotFoundError: If the file does not exist.
        NotImplementedError: If the file type is not supported.
        TomlDecodeError: Error while decoding toml.

    Note:
        If the operating system is Windows, the paths in the configuration data will
        be converted to Windows format.
    """

    if config_file.endswith(".toml"):
        config = toml.load(config_file)
    else:
        raise NotImplementedError(
            f"config_file type {config_file} is not supported. " f"Only toml config files are supported."
        )
    return config


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_validated_config_raw(config_filepath: str, schema: Type[ModelT]) -> dict:
    """
    Load a configuration file, validate it using a Pydantic model
    and return the raw dictionary.

    Args:
        config_filepath (str): Path to the config file.
        schema (Type[ModelT]): Pydantic model class used to validate.

    Returns:
        dict: Raw config dictionary.

    Raises:
        IOError: When an array with no valid (existing)
        FileNotFoundError: If the file does not exist.
        NotImplementedError: If the file type is not supported.
        TomlDecodeError: Error while decoding toml
        ConfigValidationError: If validation fails.
    """
    config_raw = load_config(config_filepath)
    try:
        schema.model_validate(config_raw, extra="forbid")
    except pydantic.ValidationError as e:
        raise ConfigValidationError(e, Path(config_filepath)) from None
    return config_raw
