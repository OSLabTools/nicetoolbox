import glob
import re
from pathlib import Path
from typing import Any

from .placeholders import get_placeholders_str

# regex to extract [named_wildcards] from path
BRACKET_TOKEN = re.compile(r"\[([a-zA-Z0-9_-]+)\]")


def discovery_to_regex(pattern: str) -> re.Pattern:
    """
    Compile a discover_sequences pattern into a regex that matches full candidate
    paths and yields the captured `[name]` values as named groups.

    Args:
        pattern (str): A `discover_sequences` string containing `[name]` named wildcards
            and literal path segments. All `<placeholder>` tokens must already
            be resolved to concrete strings before this is called.

    Returns:
        re.Pattern: Compiled regex ready to run against candidate paths via
            `regex.match(candidate)`. Extract captured values with `m.groupdict()`.

    Raises:
        ValueError: If the pattern contains unresolved `<placeholder>` tokens,
            has no `[name]` tokens to capture, or reuses the same `[name]` twice.
    """
    # check that pattern doesn't contain unnamed glob wildcards
    # (`*` is never valid inside a [name] token, so any `*` is a raw wildcard)
    if "*" in pattern:
        raise ValueError(
            f"discover_sequences pattern contains unnamed glob wildcards (`*`): {pattern!r}. "
            "Only named `[name]` wildcards are supported."
        )

    # check that all placeholders are already resolved
    unresolved = get_placeholders_str(pattern)
    if unresolved:
        raise ValueError(
            f"discover_sequences pattern still contains unresolved placeholders "
            f"{sorted(unresolved)}: {pattern!r}. "
            "Ensure all placeholders (e.g. <datasets_folder_path>) are defined before discovery."
        )
    # check that we have any named wildcards to discover
    matches = list(BRACKET_TOKEN.finditer(pattern))
    if not matches:
        raise ValueError(
            f"discover_sequences pattern has no [name] tokens: {pattern!r}. "
            "Nothing to discover — list sequences explicitly instead."
        )
    # extract all named wildcards names
    names = [m.group(1) for m in matches]

    # check that there are no duplicate named wildcards
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"discover_sequences pattern reuses capture names {sorted(duplicates)}: {pattern!r}.")

    # check that no two named wildcards are adjacent (would produce ambiguous capture)
    for prev, curr in zip(matches, matches[1:]):
        if prev.end() == curr.start():
            raise ValueError(
                f"discover_sequences pattern has adjacent [name] tokens with no separator: {pattern!r}. "
                "Add a literal character between them (e.g. `[a]_[b]`) to disambiguate the capture."
            )

    # convert each named placeholder match to regex
    regex_parts: list[str] = []
    last = 0
    for m in matches:
        regex_parts.append(re.escape(pattern[last : m.start()]))
        match_name = m.group(1)
        regex_parts.append(rf"(?P<{match_name}>[^/]+)")
        last = m.end()
    regex_parts.append(re.escape(pattern[last:]))

    # compile final regex
    regex = re.compile("^" + "".join(regex_parts) + "$")
    return regex


def _normalize(path: str) -> str:
    """Normalize a path string to an absolute, forward-slash-only form."""
    return Path(path).absolute().as_posix()


def discover_sequences_for_dataset(pattern: str) -> list[dict[str, Any]]:
    """
    Enumerate filesystem entries matching a `discover_sequences` pattern and
    return one dict of captured values per match.

    The pattern is normalized, compiled into a regex, and turned into a glob
    pattern. `glob.glob` finds candidate paths on disk; the regex extracts the
    `[name]` capture values from each candidate. Results are sorted for
    deterministic order across runs.

    Args:
        pattern (str): A `discover_sequences` string with `[name]` tokens and
            literal path segments (e.g. `"/data/mpi/[session]/[recording]/"`).
            Must not contain unresolved `<placeholder>` tokens.

    Returns:
        list[dict[str, Any]]: One dict per matched path, keyed by the `[name]`
            tokens (e.g. `{"session": "S1", "recording": "Seq1"}`).

    Raises:
        ValueError: If the pattern is malformed (see `discovery_to_regex`).
        FileNotFoundError: If no filesystem entries match the pattern.
    """
    # convert normalized path pattern to regex
    pattern = _normalize(pattern)
    regex = discovery_to_regex(pattern)

    # replace all [brackets] with regular wildcards for glob
    glob_pattern = BRACKET_TOKEN.sub("*", pattern)

    # do the filesystem walk
    matches: list[dict[str, Any]] = []
    for candidate in sorted(glob.glob(glob_pattern)):
        # found some path, extract named wildcards using regex
        candidate = _normalize(candidate)
        m = regex.match(candidate)
        matches.append(m.groupdict())

    if not matches:
        raise FileNotFoundError(f"discover_sequences pattern matched zero paths on disk: {pattern!r}.")

    return matches
