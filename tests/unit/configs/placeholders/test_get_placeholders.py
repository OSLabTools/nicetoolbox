"""
Unit tests for getting a set of placeholders from structures.
"""

from pathlib import Path

from pydantic import BaseModel

from nicetoolbox.configs.placeholders import get_placeholders, get_placeholders_str


def test_get_placeholders_str():
    """Test extracting placeholders from strings."""
    # Empty string
    assert get_placeholders_str("") == set()

    # No placeholders
    assert get_placeholders_str("plain text") == set()

    # Single placeholder
    assert get_placeholders_str("<name>") == {"name"}

    # Multiple placeholders
    assert get_placeholders_str("<first> and <second>") == {"first", "second"}

    # Duplicate placeholders
    assert get_placeholders_str("<name> and <name>") == {"name"}

    # Complex pattern
    assert get_placeholders_str("path/<dir>/<file>.txt") == {"dir", "file"}


def test_get_placeholders_basic_structures():
    """Test extracting placeholders from basic structures."""
    # String
    assert get_placeholders("Hello <name>") == {"name"}

    # Dict
    assert get_placeholders({"key": "<value>"}) == {"value"}
    assert get_placeholders({"k1": "<v1>", "k2": "<v2>"}) == {"v1", "v2"}

    # List
    assert get_placeholders(["<first>", "<second>"]) == {"first", "second"}

    # Empty structures
    assert get_placeholders({}) == set()
    assert get_placeholders([]) == set()

    # Primitives (no placeholders)
    assert get_placeholders(42) == set()
    assert get_placeholders(3.14) == set()
    assert get_placeholders(True) == set()
    assert get_placeholders(None) == set()


def test_get_placeholders_nested():
    """Test extracting placeholders from nested structures."""
    # Nested dicts
    result = get_placeholders(
        {
            "outer": "<outer_val>",
            "nested": {"inner": "<inner_val>", "deep": {"value": "<deep_val>"}},
        }
    )
    assert result == {"outer_val", "inner_val", "deep_val"}

    # Nested lists
    result = get_placeholders(["<first>", ["<nested1>", "<nested2>"], [["<deep>"]]])
    assert result == {"first", "nested1", "nested2", "deep"}

    # Mixed nesting
    result = get_placeholders(
        {
            "config": {
                "name": "<app_name>",
                "paths": ["<path1>", "<path2>"],
                "settings": {"key": "<setting>"},
            }
        }
    )
    assert result == {"app_name", "path1", "path2", "setting"}


def test_get_placeholders_mixed_types():
    """Test extracting placeholders from structures with mixed types."""
    result = get_placeholders(
        {
            "string": "<value>",
            "plain": "no placeholder",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": ["<item1>", 100, "<item2>"],
            "nested": {"field": "<nested_value>", "count": 5},
        }
    )
    assert result == {"value", "item1", "item2", "nested_value"}


def test_get_placeholders_pydantic_basic():
    """Test extracting placeholders from Pydantic models."""

    class SimpleConfig(BaseModel):
        name: str
        path: str
        port: int

    config = SimpleConfig(name="<app_name>", path="<base_dir>/data", port=8080)
    result = get_placeholders(config)
    assert result == {"app_name", "base_dir"}


def test_get_placeholders_pydantic_with_path():
    """Test extracting placeholders from Pydantic models with Path fields."""

    class ConfigWithPath(BaseModel):
        name: str
        data_dir: Path
        log_file: Path
        port: int

    config = ConfigWithPath(
        name="<app_name>",
        data_dir=Path("<base_dir>/data"),
        log_file=Path("<log_dir>/app.log"),
        port=8080,
    )
    result = get_placeholders(config)
    assert result == {"app_name", "base_dir", "log_dir"}


def test_get_placeholders_pydantic_nested():
    """Test extracting placeholders from nested Pydantic models."""

    class DatabaseConfig(BaseModel):
        host: str
        port: int
        user: str

    class AppConfig(BaseModel):
        name: str
        database: DatabaseConfig

    config = AppConfig(
        name="<app_name>",
        database=DatabaseConfig(host="<db_host>", port=5432, user="<db_user>"),
    )
    result = get_placeholders(config)
    assert result == {"app_name", "db_host", "db_user"}


def test_get_placeholders_pydantic_collections():
    """Test extracting placeholders from Pydantic models with collections."""

    class ConfigWithCollections(BaseModel):
        name: str
        paths: list[str]
        settings: dict[str, str]

    config = ConfigWithCollections(
        name="<app_name>",
        paths=["<path1>", "<path2>", "<path3>"],
        settings={"key1": "<val1>", "key2": "<val2>"},
    )
    result = get_placeholders(config)
    assert result == {"app_name", "path1", "path2", "path3", "val1", "val2"}


def test_get_placeholders_pydantic_with_path_list():
    """Test extracting placeholders from Pydantic models with list of Paths."""

    class ConfigWithPathList(BaseModel):
        name: str
        input_dirs: list[Path]
        output_file: Path

    config = ConfigWithPathList(
        name="<project>",
        input_dirs=[
            Path("<base>/input1"),
            Path("<base>/input2"),
            Path("<alt_base>/input3"),
        ],
        output_file=Path("<output_dir>/result.txt"),
    )
    result = get_placeholders(config)
    assert result == {"project", "base", "alt_base", "output_dir"}


def test_get_placeholders_pydantic_complex():
    """Test extracting placeholders from complex Pydantic structures."""

    class Server(BaseModel):
        host: str
        port: int

    class PathConfig(BaseModel):
        data_dir: Path
        log_dir: Path

    class AppConfig(BaseModel):
        name: str
        version: str
        servers: list[Server]
        paths: PathConfig
        settings: dict[str, str]

    config = AppConfig(
        name="<app_name>",
        version="<app_version>",
        servers=[Server(host="<host1>", port=8001), Server(host="<host2>", port=8002)],
        paths=PathConfig(data_dir=Path("<base>/data"), log_dir=Path("<base>/logs")),
        settings={"key1": "<setting1>", "key2": "<setting2>"},
    )
    result = get_placeholders(config)
    assert result == {
        "app_name",
        "app_version",
        "host1",
        "host2",
        "base",
        "setting1",
        "setting2",
    }


def test_get_placeholders_pydantic_in_structures():
    """Test extracting placeholders from Pydantic models within lists and dicts."""

    class Item(BaseModel):
        name: str
        path: Path

    # Models in list
    items = [
        Item(name="<name1>", path=Path("<dir1>/file1")),
        Item(name="<name2>", path=Path("<dir2>/file2")),
    ]
    result = get_placeholders(items)
    assert result == {"name1", "name2", "dir1", "dir2"}

    # Models in dict
    items_dict = {
        "first": Item(name="<name1>", path=Path("<dir1>/file1")),
        "second": Item(name="<name2>", path=Path("<dir2>/file2")),
    }
    result = get_placeholders(items_dict)
    assert result == {"name1", "name2", "dir1", "dir2"}
