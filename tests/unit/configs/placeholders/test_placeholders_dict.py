"""
Unit tests for dicts placeholder resolution.
"""

from nicetoolbox.configs.placeholders import resolve_placeholders_dict_mut


def test_resolve_placeholder_dict_basic_and_self_reference():
    """Test basic resolution, self-references, and chained dependencies."""
    # Simple replacement from context
    result = resolve_placeholders_dict_mut(
        {"key1": "<value>", "key2": "Hello <name>"},
        {"value": "replaced", "name": "World"},
    )
    assert result == {"key1": "replaced", "key2": "Hello World"}

    # Non-string values preserved
    result = resolve_placeholders_dict_mut({"str": "<val>", "int": 42, "float": 3.14, "bool": True}, {"val": "TEXT"})
    assert result == {"str": "TEXT", "int": 42, "float": 3.14, "bool": True}

    # Simple self-reference
    result = resolve_placeholders_dict_mut({"base": "/home/user", "path": "<base>/data"}, {})
    assert result == {"base": "/home/user", "path": "/home/user/data"}

    # Multiple self-references
    result = resolve_placeholders_dict_mut({"a": "A", "b": "B", "ab": "<a><b>", "ba": "<b><a>"}, {})
    assert result == {"a": "A", "b": "B", "ab": "AB", "ba": "BA"}

    # Self-reference with external placeholders and chained dependencies
    result = resolve_placeholders_dict_mut(
        {
            "user": "<username>",
            "home": "/home/<user>",
            "data": "<home>/data",
            "v1": "start",
            "v2": "<v1>-2",
            "v3": "<v2>-3",
            "v4": "<v3>-4",
        },
        {"username": "alice"},
    )
    assert result == {
        "user": "alice",
        "home": "/home/alice",
        "data": "/home/alice/data",
        "v1": "start",
        "v2": "start-2",
        "v3": "start-2-3",
        "v4": "start-2-3-4",
    }

    # Mixed with external context
    result = resolve_placeholders_dict_mut({"base": "<root>/app", "config": "<base>/config.yml"}, {"root": "/var"})
    assert result == {"base": "/var/app", "config": "/var/app/config.yml"}


def test_resolve_placeholder_dict_unreachable():
    """Test unreachable placeholders that are allowed to remain unresolved."""
    # Single unreachable placeholder
    result = resolve_placeholders_dict_mut({"key1": "value", "key2": "<runtime_var>"}, {}, unreachable={"runtime_var"})
    assert result == {"key1": "value", "key2": "<runtime_var>"}

    # Multiple unreachable placeholders
    result = resolve_placeholders_dict_mut(
        {"resolved": "<base>/data", "unresolved1": "<var1>", "unresolved2": "<var2>"},
        {"base": "/home"},
        unreachable={"var1", "var2"},
    )
    assert result == {
        "resolved": "/home/data",
        "unresolved1": "<var1>",
        "unresolved2": "<var2>",
    }

    # Mixed resolved and unreachable
    result = resolve_placeholders_dict_mut({"a": "<b>", "b": "B", "c": "<runtime>"}, {}, unreachable={"runtime"})
    assert result == {"a": "B", "b": "B", "c": "<runtime>"}


def test_resolve_placeholder_dict_input_not_mutated():
    """Test that input dict is not mutated during resolution."""
    # Original input dict should remain unchanged
    input_dict = {"base": "/home/user", "path": "<base>/data", "port": 8080}
    input_original = input_dict.copy()

    result = resolve_placeholders_dict_mut(input_dict, {})

    # Result should have resolved placeholders
    assert result == {"base": "/home/user", "path": "/home/user/data", "port": 8080}
    # Original input should be unchanged
    assert input_dict == input_original, "Input dict was mutated!"

    # Test that placeholders dict is mutated
    input_dict2 = {"base": "/home/user", "path": "<base>/data"}
    placeholders = {"external": "value"}
    resolve_placeholders_dict_mut(input_dict2, placeholders)
    assert placeholders == {
        "external": "value",
        "base": "/home/user",
        "path": "/home/user/data",
    }

    # Test that unreachable set is not mutated
    input_dict3 = {"resolved": "<base>/data", "unresolved": "<runtime>"}
    placeholders3 = {"base": "/home"}
    unreachable = {"runtime"}
    unreachable_original = unreachable.copy()
    resolve_placeholders_dict_mut(input_dict3, placeholders3, unreachable=unreachable)
    assert unreachable == unreachable_original, "Unreachable set was mutated!"


def test_resolve_placeholder_dict_errors():
    """Test error conditions: circular dependencies, unresolved, and key collision."""
    import pytest

    # Circular dependency: a -> b -> c -> a
    with pytest.raises(ValueError, match="Could not resolve|Couldn't resolve"):
        resolve_placeholders_dict_mut({"a": "<b>", "b": "<c>", "c": "<a>"}, {})

    # Self-circular: a references itself
    with pytest.raises(ValueError, match="Could not resolve|Couldn't resolve"):
        resolve_placeholders_dict_mut({"a": "prefix_<a>"}, {})

    # Missing placeholders with partial resolution
    with pytest.raises(ValueError, match="Could not resolve placeholders"):
        resolve_placeholders_dict_mut(
            {"resolved": "<exists>", "missing": "<missing1>", "missing2": "<missing2>"},
            {"exists": "VALUE"},
        )

    # Key collision between input and placeholders (string and non-string fields)
    with pytest.raises(KeyError, match="Fields collision"):
        resolve_placeholders_dict_mut({"name": "value"}, {"name": "other"})

    with pytest.raises(KeyError, match="Fields collision"):
        resolve_placeholders_dict_mut({"port": 8080}, {"port": "8000"})


def test_resolve_placeholder_dict_edge_cases():
    """Test edge cases, empty dicts, context non-mutation, and max iterations."""
    import pytest

    # Empty dict
    assert resolve_placeholders_dict_mut({}, {}) == {}

    # Empty placeholders
    assert resolve_placeholders_dict_mut({"key": "value"}, {}) == {"key": "value"}

    # No placeholders in values
    assert resolve_placeholders_dict_mut({"a": "A", "b": "B"}, {"unused": "X"}) == {
        "a": "A",
        "b": "B",
    }

    # Empty strings
    result = resolve_placeholders_dict_mut({"empty": "", "filled": "<val>"}, {"val": "V"})
    assert result == {"empty": "", "filled": "V"}

    # Context dict is mutated
    placeholders = {"external": "EXT"}
    input_dict = {"internal": "INT", "combined": "<external>-<internal>"}
    result = resolve_placeholders_dict_mut(input_dict, placeholders)
    assert result == {"internal": "INT", "combined": "EXT-INT"}
    assert placeholders == {"internal": "INT", "combined": "EXT-INT", "external": "EXT"}

    # Max iterations exceeded with circular dependency
    with pytest.raises(ValueError, match="Could not resolve|Couldn't resolve"):
        resolve_placeholders_dict_mut({"a": "<b>", "b": "<a>"}, {}, max_iterations=2)


def test_resolve_placeholder_dict_complex_realistic():
    """Test realistic configuration scenarios."""
    # Application configuration
    config = {
        "app_name": "myapp",
        "env": "production",
        "base_dir": "/opt/<app_name>",
        "data_dir": "<base_dir>/data",
        "log_dir": "<base_dir>/logs",
        "db_host": "localhost",
        "db_port": "5432",
        "db_url": "postgresql://<db_host>:<db_port>/<app_name>",
        "log_file": "<log_dir>/<app_name>-<env>.log",
    }
    placeholders = {}

    result = resolve_placeholders_dict_mut(config, placeholders)

    assert result == {
        "app_name": "myapp",
        "env": "production",
        "base_dir": "/opt/myapp",
        "data_dir": "/opt/myapp/data",
        "log_dir": "/opt/myapp/logs",
        "db_host": "localhost",
        "db_port": "5432",
        "db_url": "postgresql://localhost:5432/myapp",
        "log_file": "/opt/myapp/logs/myapp-production.log",
    }


def test_resolve_placeholder_dict_misc_features():
    """Test misc features: unreachable, special chars, partial, nested, order."""
    import pytest

    # Mixed: unreachable, resolvable, missing
    with pytest.raises(ValueError, match="missing_var"):
        resolve_placeholders_dict_mut(
            {
                "resolved": "<exists>",
                "runtime": "<runtime_var>",
                "missing": "<missing_var>",
            },
            {"exists": "VALUE"},
            unreachable={"runtime_var"},
        )

    # Special characters (underscores, hyphens, alphanumeric, digits)
    result = resolve_placeholders_dict_mut(
        {
            "my_var": "value1",
            "my-var": "value2",
            "var123": "V3",
            "path": "<my_var>/<my-var>/<var123>",
        },
        {},
    )
    assert result == {
        "my_var": "value1",
        "my-var": "value2",
        "var123": "V3",
        "path": "value1/value2/V3",
    }

    # Partial resolution (start, middle, end)
    result = resolve_placeholders_dict_mut(
        {
            "var": "VALUE",
            "start": "<var>-suffix",
            "mid": "prefix-<var>-suffix",
            "end": "prefix-<var>",
        },
        {},
    )
    assert result == {
        "var": "VALUE",
        "start": "VALUE-suffix",
        "mid": "prefix-VALUE-suffix",
        "end": "prefix-VALUE",
    }

    # Nested structures NOT processed (only top-level strings)
    result = resolve_placeholders_dict_mut(
        {"top": "<val>", "nested": {"inner": "<val>"}, "list": ["<val>", "<other>"]},
        {"val": "REPLACED"},
    )
    assert result == {
        "top": "REPLACED",
        "nested": {"inner": "<val>"},
        "list": ["<val>", "<other>"],
    }

    # Order independence - placeholders can reference each other regardless of key order
    result = resolve_placeholders_dict_mut(
        {
            "final": "<intermediate>-end",
            "intermediate": "<base>-mid",
            "base": "start",
            "c": "<a>-<b>",
            "a": "A",
            "b": "B",
        },
        {},
    )
    assert result == {
        "final": "start-mid-end",
        "intermediate": "start-mid",
        "base": "start",
        "c": "A-B",
        "a": "A",
        "b": "B",
    }
