# shared config utils functions

import getpass
import os
import time
from typing import TypeVar

import pydantic
from pydantic import BaseModel

from ..configs.config_handler import ConfigValidationError
from ..utils.git_utils import try_get_toolbox_git_metadata

# general type for all pydantic models
ModelT = TypeVar("ModelT", bound=BaseModel)


def model_to_dict(model: BaseModel) -> dict:
    """Converts pydantic model into the primitives dict"""
    # pydantic tries to keep Paths, UUID and other fields as python types
    # that confuses our resolver. json mode force to convert them to primitives
    # we also serialize_as_any to force nested structured to be serialized
    # and some of our fields has aliases, so we force it use original names
    return model.model_dump(mode="json", serialize_as_any=True, by_alias=True)


def dict_to_model(config_raw: dict, schema: type[ModelT]) -> ModelT:
    """Converts and validate dict to pydantic model"""
    try:
        config = schema.model_validate(config_raw)
    except pydantic.ValidationError as e:
        raise ConfigValidationError(e) from None
    return config


def keys_collision_dict(a: dict, b: dict) -> set:
    """Returns key collision between two dictionaries"""
    return set(a) & set(b)


def merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dictionaries, raising an error if any keys overlap."""
    collision = keys_collision_dict(a, b)
    if collision:
        raise KeyError(f"Duplicate keys detected: {collision}")
    return {**a, **b}


def default_auto_placeholders(working_directory=None):
    if working_directory is None:
        working_directory = os.getcwd()

    git_metadata = try_get_toolbox_git_metadata(working_directory)
    if git_metadata is not None:
        git_hash, commit_message = git_metadata
        git_hash = git_hash[:7]
    else:
        git_hash, commit_message = "unknown", "unknown"

    placeholder_dict = dict(
        git_hash=git_hash,
        commit_message=commit_message,
        me=getpass.getuser(),
        yyyymmdd=time.strftime("%Y%m%d", time.localtime()),
        today=time.strftime("%Y%m%d", time.localtime()),
        time=time.strftime("%H_%M", time.localtime()),
        pwd=working_directory,
    )

    return placeholder_dict


def default_runtime_placeholders():
    return {
        "video_length",
        "video_start",
        "sequence_ID",
        "dataset_name",
        "session_ID",
        "cam_face1",
        "cam_face2",
        "cam_top",
        "cam_front",
        "algorithm_name",
        "component_name",
    }
