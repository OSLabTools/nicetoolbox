import re
from typing import Any, Optional, Union

from pydantic import BaseModel

from .utils import keys_collision_dict, model_to_dict

# Pattern matches <key> where key can contain alphanumeric, underscore and hyphen
# This is guaranteed to be valid TOML field name
PLACEHOLDER_PATTERN = re.compile(r"<([a-zA-Z0-9_-]+)>")

# Only this types are allowed to be placeholder values
# Other types will be ignored or cause undefined behaviour
PLACEHOLDERS_TYPE = Union[str, int, float, bool]


def get_placeholders_str(input: str) -> set[str]:
    """Extract all placeholder names from a string."""
    return set(re.findall(PLACEHOLDER_PATTERN, input))


def get_placeholders(input: Any) -> set[str]:
    """Extract all placeholder names from any data structure."""
    if isinstance(input, str):
        return get_placeholders_str(input)

    if isinstance(input, dict):
        placeholders = set()
        for value in input.values():
            placeholders.update(get_placeholders(value))
        return placeholders

    if isinstance(input, list):
        placeholders = set()
        for item in input:
            placeholders.update(get_placeholders(item))
        return placeholders

    if isinstance(input, BaseModel):
        # Convert to dict and recursively extract
        return get_placeholders(model_to_dict(input))

    # For other types (int, float, bool, None), no placeholders
    return set()


def resolve_placeholders_str(
    input: str,
    placeholders: dict[str, PLACEHOLDERS_TYPE],
    unresolved: Optional[set[str]] = None,
    unreachable: Optional[set[str]] = None,
) -> str:
    r"""
    Find and resolve placeholders in a string with corresponding values from dictionary.
    Placeholders should be formatted as \<var\> strings. If placeholders key isn't
    inside dict, it will be ignored and added to optional unresolved placeholders set.

    Args:
        input (str): The string containing placeholders to be replaced.
        placeholders (dict[str, PLACEHOLDERS_TYPE]): A dictionary containing
            the placeholder values.
        unresolved (set, optional): A set to collect unresolved placeholder keys.
            Placeholder is unresolved if it's not present in dict or its value
            contains unresolved placeholders (including unreachable placeholders).
        unreachable (set, optional): Set of placeholder names that are allowed to
            remain unresolved (e.g., runtime placeholders resolved later).

    Returns:
        str: The string with placeholders replaced.
    """
    if unresolved is None:
        unresolved = set()
    if unreachable is None:
        unreachable = set()

    # function to replace matched placeholders with dict values
    def replace_match(match: re.Match[str]) -> str:
        orig_str = match.group(0)  # full input string "<var>"
        key = match.group(1)  # actual name "var"

        # if placeholder's unknown - mark it unresolved and return as is
        if key not in placeholders:
            unresolved.add(key)
            return orig_str

        # if placeholder's known - check if replacement contains unknown placeholders
        # we mark all unknown placeholders replacement as unresolved
        replacement = str(placeholders[key])  # force placeholder to string
        unknown_placeholders = get_placeholders_str(replacement) - unreachable
        if unknown_placeholders:
            unresolved.add(key)
        return replacement

    # replace all known placeholders keys with provided values
    return re.sub(PLACEHOLDER_PATTERN, replace_match, input)


def resolve_placeholders_str_strict(
    input: str,
    placeholders: dict[str, PLACEHOLDERS_TYPE],
    unreachable: Optional[set[str]] = None,
) -> str:
    """
    Same as resolve_placeholders_str, but raise if finds unexpected placeholders.
    """
    if unreachable is None:
        unreachable = set()
    unresolved = set()

    result = resolve_placeholders_str(input, placeholders, unresolved, unreachable)
    unexpected = unresolved - unreachable
    if unexpected:
        raise ValueError(
            f"Could not resolve placeholders: {unexpected}. "
            f"Check for typos or circular dependencies."
        )
    return result


def resolve_placeholders_dict_mut(
    input: dict[str, PLACEHOLDERS_TYPE],
    placeholders: dict[str, PLACEHOLDERS_TYPE],
    unreachable: Optional[set[str]] = None,
    max_iterations: int = 5,
) -> dict[str, Any]:
    r"""
    Resolve placeholders in a dict using both provided context and self-references.
    Iteratively resolves \<var\> placeholders in string values using:
    1. Provided placeholders dict
    2. Other string values within the same dict (self-reference)

    Input placeholders dictionary will be mutated with new local placeholders.

    Resolution continues until no changes occur or max_iterations is reached.
    Non-string values (numbers, lists, nested dicts) are returned unchanged.
    Function expects to resolve all known placeholders or raise an error.

    Args:
        input(dict[str, PLACEHOLDERS_TYPE]): Dict containing string values to resolve.
        placeholders(dict[str, Any]): Context dict for resolution. Will be mutated
             to include resolved string values from input.
        unreachable(Optional[set[str]]): Placeholder names that are allowed to remain
            unresolved (e.g., runtime placeholders resolved later).
        max_iterations(int): Maximum resolution passes to handle chained dependencies.

    Returns:
        Dict with all resolvable placeholders replaced.

    Raises:
        ValueError: If unresolved placeholders remain that are not in unreachable
            (indicates typos or circular dependencies).
        KeyError: If placeholders and input dicts have names collision (including
            non-string fields names)
    """
    # check fields name collision
    collision = keys_collision_dict(placeholders, input)
    if collision:
        raise KeyError(
            f"Fields collision between local and "
            f"placeholders field names: {collision}"
        )

    # default values and copies
    if unreachable is None:
        unreachable = set()
    result = dict(input)

    for _ in range(max_iterations):
        # update placeholders context with local dict fields
        local_ctx = {
            k: v for k, v in result.items() if isinstance(v, PLACEHOLDERS_TYPE)
        }
        placeholders.update(local_ctx)

        # try to resolve placeholders with combined context
        unresolved: set[str] = set()
        new_result = {
            k: resolve_placeholders_str(v, placeholders, unresolved, unreachable)
            if isinstance(v, str)
            else v
            for k, v in result.items()
        }

        # no changes? either fully resolved or stuck
        if new_result == result:
            unexpected = unresolved - unreachable
            if unexpected:
                raise ValueError(
                    f"Could not resolve placeholders: {unexpected}. "
                    f"Check for typos or circular dependencies."
                )
            return new_result

        # update results and start next iteration
        result = new_result

    # Did all iterations - still didn't converged
    # Looks like we have circular dependency
    raise ValueError(
        f"Couldn't resolve placeholders after {max_iterations} iterations. "
        f"Unexpected placeholders: {unresolved - unreachable}"
    )


def resolve_placeholders(
    input: Any,
    placeholders: dict[str, PLACEHOLDERS_TYPE],
    unreachable: Optional[set[str]] = None,
) -> Any:
    r"""
    Recursively find and resolve placeholders in input data with corresponding values.
    Placeholders should be formatted as \<var\> strings inside data structure fields.
    If placeholders key isn't inside dict or can't be resolved from local context,
    it will rise resolution error.

    Args:
        input: The data structure to be processed. Only strings, lists, dicts and
            pydantics.BaseModel are processed. Other input is returned as is.
        placeholders (dict[str, PLACEHOLDERS_TYPE]): A dictionary containing the
            placeholder values.
        unreachable (set, optional): Set of placeholder names that are allowed to
            remain unresolved (e.g., runtime placeholders resolved later).

    Returns:
        The copy of processed data structure with placeholders replaced.

    Raises:
        ValueError: If dict-level placeholders cannot be resolved (circular dependencies
            or missing placeholders that are not in unreachable set).
        KeyError: If there's a field name collision between input dict and placeholders
            dict when processing inputs.
    """
    if isinstance(input, dict):
        new_placeholders = dict(placeholders)
        # resolve placeholders on a local level
        # copy for avoiding original placeholders mutation
        result = resolve_placeholders_dict_mut(input, new_placeholders, unreachable)

        # run recursively for all non-strings (strings should be already resolved)
        # we send a copy of placeholders to avoid contamination
        return {
            k: resolve_placeholders(v, new_placeholders, unreachable)
            if not isinstance(v, str)
            else v
            for k, v in result.items()
        }

    if isinstance(input, list):
        return [resolve_placeholders(item, placeholders, unreachable) for item in input]

    # TODO: support for sets and tuples?

    # for pydantic models: convert them to dict, recursively resolve it and convert back
    # we do it with validation in case of custom post-validation hooks
    if isinstance(input, BaseModel):
        input_dict = model_to_dict(input)
        processed_dict = resolve_placeholders(input_dict, placeholders, unreachable)
        return type(input).model_validate(processed_dict)

    # finally strings are just properly resolved
    if isinstance(input, str):
        return resolve_placeholders_str_strict(input, placeholders, unreachable)

    # if we got any other type during recursion (e.g. float or int), we return it as is
    return input
