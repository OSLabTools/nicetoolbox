"""
Unit tests for placeholder resolution in strings.
"""

import pytest

from nicetoolbox.configs.placeholders import resolve_placeholders_str, resolve_placeholders_str_strict


def test_resolve_placeholders_str_basic():
    """Test basic string placeholder resolution."""
    # Empty string
    assert resolve_placeholders_str("", {}) == ""

    # No placeholders
    assert resolve_placeholders_str("bar", {}) == "bar"

    # Single placeholder
    assert resolve_placeholders_str("<name>", {"name": "John"}) == "John"
    assert resolve_placeholders_str("Hello <name>", {"name": "Alice"}) == "Hello Alice"

    # Multiple placeholders
    ret = resolve_placeholders_str(
        "<greeting> <name>, welcome to <place>!",
        {"greeting": "Hi", "name": "Bob", "place": "Earth"},
    )
    assert ret == "Hi Bob, welcome to Earth!"

    # Repeated placeholders
    ret = resolve_placeholders_str("<var> and <var> make two <var>s", {"var": "apple"})
    assert ret == "apple and apple make two apples"


def test_resolve_placeholders_str_unresolved():
    """Test unresolved placeholder handling."""
    # Without tracking
    ret = resolve_placeholders_str("Hello <name>, your age is <age>", {"name": "Alice"})
    assert ret == "Hello Alice, your age is <age>"

    # With tracking
    unresolved = set()
    ret = resolve_placeholders_str(
        "Hello <name>, your age is <age> and city is <city>",
        {"name": "Alice"},
        unresolved,
    )
    assert ret == "Hello Alice, your age is <age> and city is <city>"
    assert unresolved == {"age", "city"}

    # Preserves existing unresolved set
    unresolved = {"existing"}
    ret = resolve_placeholders_str("<new_unresolved>", {}, unresolved)
    assert unresolved == {"existing", "new_unresolved"}


def test_resolve_placeholders_str_special_cases():
    """Test special characters and edge cases."""
    # Underscores and hyphens
    ret = resolve_placeholders_str("<my_var> and <my-var>", {"my_var": "underscore", "my-var": "hyphen"})
    assert ret == "underscore and hyphen"

    # Empty placeholder value
    assert resolve_placeholders_str("Before<empty>After", {"empty": ""}) == "BeforeAfter"

    # Adjacent placeholders
    ret = resolve_placeholders_str("<first><second><third>", {"first": "A", "second": "B", "third": "C"})
    assert ret == "ABC"

    # Invalid formats
    ret = resolve_placeholders_str("<incomplete", {"incomplete": "value"})
    assert ret == "<incomplete"

    # cursed cases
    assert resolve_placeholders_str("<>", {}) == "<>"
    assert resolve_placeholders_str("<>", {"": "var"}) == "<>"
    assert resolve_placeholders_str("<my var>", {"my var": "value"}) == "<my var>"
    # this one is technically correct, but I hope no one will ever write like that
    assert resolve_placeholders_str("<<var>>", {"var": "value"}) == "<value>"


def test_resolve_placeholders_str_nonstring_values():
    """Test that non-string placeholder values are converted to strings."""
    # Integer placeholder value
    assert resolve_placeholders_str("<port>", {"port": 8080}) == "8080"

    # Float placeholder value
    assert resolve_placeholders_str("<version>", {"version": 1.5}) == "1.5"

    # Boolean placeholder values
    assert resolve_placeholders_str("<enabled>", {"enabled": True}) == "True"
    assert resolve_placeholders_str("<disabled>", {"disabled": False}) == "False"

    # Multiple non-string placeholders
    ret = resolve_placeholders_str(
        "Port: <port>, Version: <version>, Enabled: <enabled>",
        {"port": 8080, "version": 2.0, "enabled": True},
    )
    assert ret == "Port: 8080, Version: 2.0, Enabled: True"

    # Mixed string and non-string placeholders
    ret = resolve_placeholders_str("<host>:<port>/<name>", {"host": "localhost", "port": 3000, "name": "myapp"})
    assert ret == "localhost:3000/myapp"


def test_resolve_placeholders_str_unreachable_and_circular():
    """Test unreachable parameter and circular dependency detection."""
    # Single unreachable - not marked as unresolved
    unresolved = set()
    ret = resolve_placeholders_str("<resolved> and <runtime>", {"resolved": "VALUE"}, unresolved, {"runtime"})
    assert ret == "VALUE and <runtime>"
    assert unresolved == {"runtime"}

    # Multiple unreachable
    unresolved = set()
    ret = resolve_placeholders_str("<a> <b> <c>", {}, unresolved, {"a", "b", "c"})
    assert ret == "<a> <b> <c>"
    assert unresolved == {"a", "b", "c"}

    # Mix: resolved, missing, unreachable
    unresolved = set()
    ret = resolve_placeholders_str("<resolved> <missing> <runtime>", {"resolved": "OK"}, unresolved, {"runtime"})
    assert ret == "OK <missing> <runtime>"
    assert unresolved == {"missing", "runtime"}

    # Direct self-reference marked as unresolved
    unresolved = set()
    assert resolve_placeholders_str("<a>", {"a": "<a>"}, unresolved) == "<a>"
    assert "a" in unresolved

    # Value contains different placeholder
    unresolved = set()
    assert resolve_placeholders_str("<a>", {"a": "<b>"}, unresolved) == "<b>"
    assert "a" in unresolved

    # Value with multiple placeholders
    unresolved = set()
    ret = resolve_placeholders_str("<path>", {"path": "<base>/<file>"}, unresolved)
    assert ret == "<base>/<file>"
    assert "path" in unresolved

    # Unreachable placeholder in resolved value - NOT marked as unresolved
    unresolved = set()
    ret = resolve_placeholders_str("<path>", {"path": "/data/<runtime>"}, unresolved, {"runtime"})
    assert ret == "/data/<runtime>"
    assert unresolved == set()

    # Mix of unreachable and reachable in resolved value
    unresolved = set()
    ret = resolve_placeholders_str("<config>", {"config": "<base>/<runtime>"}, unresolved, {"runtime"})
    assert ret == "<base>/<runtime>"
    assert "config" in unresolved  # Due to '<base>'

    # All placeholders in resolved value are unreachable
    unresolved = set()
    ret = resolve_placeholders_str("<path>", {"path": "<v1>/<v2>/<v3>"}, unresolved, {"v1", "v2", "v3"})
    assert ret == "<v1>/<v2>/<v3>"
    assert unresolved == set()

    # Chained placeholders - single level resolution
    unresolved = set()
    ret = resolve_placeholders_str("<final>", {"final": "<mid>", "mid": "<base>", "base": "val"}, unresolved)
    assert ret == "<mid>"
    assert "final" in unresolved

    # Complex: resolved, circular, unreachable, missing
    unresolved = set()
    ret = resolve_placeholders_str(
        "<a> <b> <c> <d>",
        {"a": "VALUE_A", "b": "<b>", "c": "<runtime>", "d": "<missing>"},
        unresolved,
        {"runtime"},
    )
    assert ret == "VALUE_A <b> <runtime> <missing>"
    assert unresolved == {"b", "d"}


def test_resolve_placeholders_str_no_mutation():
    """Test that resolve_placeholders_str doesn't mutate input data."""
    # Original placeholders dict should not be modified
    original_placeholders = {"name": "Alice", "age": 30, "city": "NYC"}
    placeholders_copy = dict(original_placeholders)

    result = resolve_placeholders_str("Hello <name>, age <age> from <city>", original_placeholders)
    assert result == "Hello Alice, age 30 from NYC"
    assert original_placeholders == placeholders_copy

    # Original unresolved should contain new items
    unresolved_before = {"existing"}
    result = resolve_placeholders_str("<new>", {}, unresolved_before)
    assert "existing" in unresolved_before
    assert "new" in unresolved_before

    # Unreachable set should not be modified
    unreachable = {"runtime", "session"}
    unreachable_copy = set(unreachable)
    unresolved = set()

    result = resolve_placeholders_str("Hello <name> and <runtime>", {"name": "Test"}, unresolved, unreachable)
    assert result == "Hello Test and <runtime>"
    assert unreachable == unreachable_copy  # Not mutated

    # Complex case with all parameters
    placeholders = {"a": "VALUE_A", "b": "<c>"}
    placeholders_copy = dict(placeholders)
    unresolved = set()
    unreachable = {"runtime"}
    unreachable_copy = set(unreachable)

    result = resolve_placeholders_str("<a> <b> <runtime>", placeholders, unresolved, unreachable)
    assert result == "VALUE_A <c> <runtime>"
    assert placeholders == placeholders_copy  # Not mutated
    assert unreachable == unreachable_copy  # Not mutated
    assert unresolved == {"b", "runtime"}  # Modified (output parameter)


def test_resolve_placeholders_str_strict():
    """Test resolve_placeholders_str_strict raises on missing placeholders."""
    # Should work when all placeholders are resolved
    result = resolve_placeholders_str_strict("Hello <name>, age <age>", {"name": "Alice", "age": 30})
    assert result == "Hello Alice, age 30"

    # Should RAISE when placeholder is missing
    with pytest.raises(ValueError, match="Could not resolve placeholders: {'missing'}"):
        resolve_placeholders_str_strict("Hello <name> and <missing>", {"name": "Bob"})

    # Should NOT raise if missing placeholder is in unreachable set
    result = resolve_placeholders_str_strict("Hello <name> and <runtime>", {"name": "Test"}, {"runtime"})
    assert result == "Hello Test and <runtime>"

    # Should RAISE if only some missing are unreachable
    with pytest.raises(ValueError, match="Could not resolve placeholders: {'missing'}"):
        resolve_placeholders_str_strict("<runtime> <missing>", {}, {"runtime"})
