"""
Unit tests for placeholder resolution.
"""

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from nicetoolbox.configs.placeholders import resolve_placeholders


def test_resolve_placeholders_basic_structures():
    """Test basic dict, list, and string resolution."""
    # Dictionary
    result = resolve_placeholders(
        {"key1": "<value>", "key2": "Hello <name>"},
        {"value": "replaced", "name": "World"},
    )
    assert result == {"key1": "replaced", "key2": "Hello World"}

    # List
    result = resolve_placeholders(
        ["<item1>", "Hello <item2>", "No placeholder"],
        {"item1": "first", "item2": "second"},
    )
    assert result == ["first", "Hello second", "No placeholder"]

    # String
    result = resolve_placeholders("<greeting> <name>!", {"greeting": "Hello", "name": "Alice"})
    assert result == "Hello Alice!"

    # Empty structures
    assert resolve_placeholders({}, {"key": "value"}) == {}
    assert resolve_placeholders([], {"key": "value"}) == []


def test_resolve_placeholders_nested():
    """Test nested structures."""
    # Nested dict
    result = resolve_placeholders(
        {
            "outer": "<outer_val>",
            "nested": {"inner": "<inner_val>", "another": "text <var>"},
        },
        {"outer_val": "OUTER", "inner_val": "INNER", "var": "VAR"},
    )
    assert result == {
        "outer": "OUTER",
        "nested": {"inner": "INNER", "another": "text VAR"},
    }

    # Nested list
    result = resolve_placeholders(
        ["<first>", ["<nested1>", "<nested2>"], "plain"],
        {"first": "A", "nested1": "B", "nested2": "C"},
    )
    assert result == ["A", ["B", "C"], "plain"]

    # Deep nesting
    result = resolve_placeholders(
        {
            "level1": {
                "level2": {"level3": ["<deep1>", "<deep2>"], "value": "<mid>"},
                "another": "<top>",
            }
        },
        {"deep1": "D1", "deep2": "D2", "mid": "M", "top": "T"},
    )
    assert result == {"level1": {"level2": {"level3": ["D1", "D2"], "value": "M"}, "another": "T"}}


def test_resolve_placeholders_mixed_types():
    """Test structures with mixed data types (non-strings pass through unchanged)."""
    result = resolve_placeholders(
        {
            "config": {
                "name": "<app_name>",
                "version": 1.5,
                "enabled": True,
                "settings": [
                    "<setting1>",
                    100,
                    {"key": "<nested_value>", "count": 5, "active": False},
                ],
            },
            "metadata": None,
            "port": 8080,
        },
        {"app_name": "MyApp", "setting1": "SETTING_ONE", "nested_value": "NESTED"},
    )
    assert result == {
        "config": {
            "name": "MyApp",
            "version": 1.5,
            "enabled": True,
            "settings": [
                "SETTING_ONE",
                100,
                {"key": "NESTED", "count": 5, "active": False},
            ],
        },
        "metadata": None,
        "port": 8080,
    }


def test_resolve_placeholders_unreachable():
    """Test unreachable placeholder handling in structures."""
    # Dict with unreachable placeholders
    result = resolve_placeholders(
        {"key1": "<resolved>", "key2": "<runtime>"},
        {"resolved": "VALUE"},
        unreachable={"runtime"},
    )
    assert result == {"key1": "VALUE", "key2": "<runtime>"}

    # List with unreachable placeholders
    result = resolve_placeholders(
        ["<resolved>", "<runtime1>", "<runtime2>"],
        {"resolved": "OK"},
        unreachable={"runtime1", "runtime2"},
    )
    assert result == ["OK", "<runtime1>", "<runtime2>"]

    # Without unreachable - should raise error
    import pytest

    with pytest.raises(ValueError, match="Could not resolve"):
        resolve_placeholders({"key1": "<resolved>", "key2": "<missing>"}, {"resolved": "VALUE"})


def test_resolve_placeholders_nonstring_values():
    """Test full resolution with non-string placeholder values in structures."""
    # Non-string value in dict referenced by nested dict
    result = resolve_placeholders({"outer": 2, "a": {"b": "<outer>"}}, {})
    assert result == {"outer": 2, "a": {"b": "2"}}

    # Dict with non-string placeholders
    result = resolve_placeholders(
        {"server": "<host>:<port>", "debug": "<debug_mode>"},
        {"host": "localhost", "port": 8080, "debug_mode": True},
    )
    assert result == {"server": "localhost:8080", "debug": "True"}

    # List with non-string placeholders
    result = resolve_placeholders(
        ["Port: <port>", "Version: <version>", "Active: <active>"],
        {"port": 3000, "version": 1.5, "active": True},
    )
    assert result == ["Port: 3000", "Version: 1.5", "Active: True"]

    # Nested structures with non-string placeholders
    result = resolve_placeholders(
        {
            "config": {
                "server": "<host>:<port>",
                "settings": ["timeout: <timeout_val>", "retries: <retry_count>"],
            },
            "metadata": {"version_str": "<ver>", "enabled_str": "<is_enabled>"},
        },
        {
            "host": "localhost",
            "port": 8080,
            "timeout_val": 30,
            "retry_count": 3,
            "ver": 2.0,
            "is_enabled": True,
        },
    )
    assert result == {
        "config": {
            "server": "localhost:8080",
            "settings": ["timeout: 30", "retries: 3"],
        },
        "metadata": {"version_str": "2.0", "enabled_str": "True"},
    }

    # With unreachable placeholders
    result = resolve_placeholders(
        {"server": "<host>:<port>", "runtime": "<runtime_var>"},
        {"host": "localhost", "port": 8080},
        unreachable={"runtime_var"},
    )
    assert result == {"server": "localhost:8080", "runtime": "<runtime_var>"}

    # Realistic example: database configuration
    config = {
        "db_url": "postgresql://<db_host>:<db_port>/<db_name>",
        "pool_size_str": "<pool_size>",
        "timeout_str": "<timeout_val>",
    }
    result = resolve_placeholders(
        config,
        {
            "db_host": "localhost",
            "db_port": 5432,
            "db_name": "mydb",
            "pool_size": 10,
            "timeout_val": 30,
        },
    )
    assert result == {
        "db_url": "postgresql://localhost:5432/mydb",
        "pool_size_str": "10",
        "timeout_str": "30",
    }


def test_resolve_placeholders_other_types():
    """Test handling of unsupported types."""
    # Numbers, None, booleans are returned as-is
    assert resolve_placeholders(123, {"key": "value"}) == 123
    assert resolve_placeholders(None, {"key": "value"}) is None
    assert resolve_placeholders(3.14, {"key": "value"}) == 3.14
    assert resolve_placeholders(True, {"key": "value"}) is True


# ============== PYDANTIC BASEMODEL TESTS ======================


def test_resolve_placeholders_pydantic_basic():
    """Test basic Pydantic model resolution."""

    class SimpleConfig(BaseModel):
        name: str
        path: str

    config = SimpleConfig(name="<app_name>", path="<base_dir>/data")
    result = resolve_placeholders(config, {"app_name": "MyApp", "base_dir": "/home/user"})

    assert isinstance(result, SimpleConfig)
    assert result.name == "MyApp"
    assert result.path == "/home/user/data"
    # Original not modified
    assert config.name == "<app_name>"


def test_resolve_placeholders_pydantic_sibling():
    """Model fields resolve against each other, same as dict siblings."""

    class Cfg(BaseModel):
        session_name: str
        recording_name: str
        folder: str

    config = Cfg(
        session_name="S1",
        recording_name="Seq1",
        folder="/data/<session_name>/<recording_name>",
    )
    result = resolve_placeholders(config, {})

    assert isinstance(result, Cfg)
    assert result.folder == "/data/S1/Seq1"


def test_resolve_placeholders_pydantic_complex():
    """Test complex nested Pydantic with collections, mixed types, and deep nesting."""

    class Credentials(BaseModel):
        user: str
        password: str

    class Database(BaseModel):
        host: str
        port: int
        credentials: Credentials

    class Server(BaseModel):
        host: str
        port: int

    class AppConfig(BaseModel):
        name: str
        version: str
        database: Database
        servers: list[Server]
        paths: list[str]
        settings: dict[str, str]

    config = AppConfig(
        name="<app_name>",
        version="<app_version>",  # Changed to avoid collision with "version" field
        database=Database(
            host="<db_host>",
            port=5432,
            credentials=Credentials(user="<db_user>", password="<db_pass>"),
        ),
        servers=[Server(host="<host1>", port=8001), Server(host="<host2>", port=8002)],
        paths=["<base>/data", "<base>/logs"],
        settings={"key": "<setting_value>"},
    )
    placeholders = {
        "app_name": "MyApp",
        "app_version": "1.0.0",  # Changed to avoid collision
        "db_host": "localhost",
        "db_user": "admin",
        "db_pass": "secret",
        "host1": "server1",
        "host2": "server2",
        "base": "/var/app",
        "setting_value": "VALUE",
    }
    result = resolve_placeholders(config, placeholders)

    assert isinstance(result, AppConfig)
    assert result.name == "MyApp"
    assert result.version == "1.0.0"
    assert result.database.host == "localhost"
    assert result.database.credentials.user == "admin"
    assert len(result.servers) == 2
    assert result.servers[0].host == "server1"
    assert result.servers[1].host == "server2"
    assert result.paths == ["/var/app/data", "/var/app/logs"]
    assert result.settings == {"key": "VALUE"}


def test_resolve_placeholders_pydantic_in_structures():
    """Test Pydantic models within lists and dicts."""

    class Item(BaseModel):
        name: str
        value: str

    # Models in list
    items = [Item(name="<name1>", value="<val1>"), Item(name="<name2>", value="<val2>")]
    placeholders = {"name1": "N1", "val1": "V1", "name2": "N2", "val2": "V2"}
    result = resolve_placeholders(items, placeholders)

    assert all(isinstance(item, Item) for item in result)
    assert result[0].name == "N1"
    assert result[1].value == "V2"

    # Models in dict
    class Config(BaseModel):
        host: str
        port: int

    config_dict = {
        "primary": Config(host="<primary_host>", port=8001),
        "secondary": Config(host="<secondary_host>", port=8002),
    }
    placeholders = {"primary_host": "server1", "secondary_host": "server2"}
    result = resolve_placeholders(config_dict, placeholders)

    assert isinstance(result["primary"], Config)
    assert result["primary"].host == "server1"
    assert result["secondary"].host == "server2"


def test_resolve_placeholders_pydantic_special_cases():
    """Test Pydantic models with optional fields and unreachable placeholders."""

    class ConfigWithOptional(BaseModel):
        name: str = Field()
        description: str | None = None
        path: str | None = None

    # Optional fields
    config = ConfigWithOptional(name="<app_name>", description="<desc>", path=None)
    result = resolve_placeholders(config, {"app_name": "MyApp", "desc": "My Application"})
    assert result.description == "My Application"
    assert result.path is None

    # Unreachable placeholders
    class SimpleConfig(BaseModel):
        name: str
        path: str

    config = SimpleConfig(name="<app_name>", path="<runtime_dir>/data")
    result = resolve_placeholders(config, {"app_name": "MyApp"}, unreachable={"runtime_dir"})
    assert result.name == "MyApp"
    assert result.path == "<runtime_dir>/data"

    # Empty strings
    class ConfigWithEmpty(BaseModel):
        name: str
        description: str

    config = ConfigWithEmpty(name="<app_name>", description="")
    result = resolve_placeholders(config, {"app_name": "MyApp"})
    assert result.description == ""

    # No placeholders
    config = SimpleConfig(name="MyApp", path="/data")
    result = resolve_placeholders(config, {"unused": "value"})
    assert result.name == "MyApp"


def test_resolve_placeholders_pydantic_simple_alias():
    """Test resolution with a simple field alias."""

    class ConfigWithAlias(BaseModel):
        custom_field: bool = Field(alias="custom-field")
        name: str

    config = ConfigWithAlias(**{"custom-field": True, "name": "<app_name>"})
    result = resolve_placeholders(config, {"app_name": "MyApp"})

    assert isinstance(result, ConfigWithAlias)
    assert result.custom_field is True
    assert result.name == "MyApp"


def test_resolve_placeholders_pydantic_alias_with_placeholder():
    """Test resolution when aliased field contains a placeholder."""

    class ConfigWithAlias(BaseModel):
        output_path: str = Field(alias="output-path")
        input_path: str = Field(alias="input-path")

    config = ConfigWithAlias(**{"output-path": "<base_dir>/output", "input-path": "<base_dir>/input"})
    result = resolve_placeholders(config, {"base_dir": "/home/user"})

    assert isinstance(result, ConfigWithAlias)
    assert result.output_path == "/home/user/output"
    assert result.input_path == "/home/user/input"


def test_resolve_placeholders_pydantic_mixed_alias_and_regular():
    """Test resolution with both aliased and regular fields."""

    class MixedConfig(BaseModel):
        regular_field: str
        aliased_field: str = Field(alias="aliased-field")
        another_regular: int

    config = MixedConfig(
        **{
            "regular_field": "<regular_val>",
            "aliased-field": "<aliased_val>",
            "another_regular": 42,
        }
    )
    result = resolve_placeholders(config, {"regular_val": "RegularValue", "aliased_val": "AliasedValue"})

    assert isinstance(result, MixedConfig)
    assert result.regular_field == "RegularValue"
    assert result.aliased_field == "AliasedValue"
    assert result.another_regular == 42


def test_resolve_placeholders_pydantic_nested_with_alias():
    """Test resolution with nested models containing aliases."""

    class InnerConfig(BaseModel):
        inner_value: str = Field(alias="inner-value")

    class OuterConfig(BaseModel):
        outer_name: str = Field(alias="outer-name")
        inner: InnerConfig

    config = OuterConfig(
        **{
            "outer-name": "<outer_placeholder>",
            "inner": {"inner-value": "<inner_placeholder>"},
        }
    )
    result = resolve_placeholders(config, {"outer_placeholder": "OuterResolved", "inner_placeholder": "InnerResolved"})

    assert isinstance(result, OuterConfig)
    assert result.outer_name == "OuterResolved"
    assert isinstance(result.inner, InnerConfig)
    assert result.inner.inner_value == "InnerResolved"


def test_resolve_placeholders_pydantic_alias_in_list():
    """Test resolution with aliased fields in models within a list."""

    class ItemConfig(BaseModel):
        item_name: str = Field(alias="item-name")
        item_value: int

    class ContainerConfig(BaseModel):
        items: List[ItemConfig]

    config = ContainerConfig(
        items=[
            ItemConfig(**{"item-name": "<name1>", "item_value": 1}),
            ItemConfig(**{"item-name": "<name2>", "item_value": 2}),
        ]
    )
    result = resolve_placeholders(config, {"name1": "First", "name2": "Second"})

    assert isinstance(result, ContainerConfig)
    assert len(result.items) == 2
    assert result.items[0].item_name == "First"
    assert result.items[1].item_name == "Second"


def test_resolve_placeholders_pydantic_alias_numeric_prefix():
    """Test resolution with numeric-prefixed aliases (like '3d_results')."""

    class FrameworkConfig(BaseModel):
        input_data_format: str
        camera_names: List[str]
        results_3d: bool = Field(alias="3d_results")
        device: str

    config = FrameworkConfig(
        **{
            "input_data_format": "<format>",
            "camera_names": ["<cam1>", "<cam2>"],
            "3d_results": True,
            "device": "<gpu>",
        }
    )
    result = resolve_placeholders(
        config,
        {"format": "RGB", "cam1": "front", "cam2": "top", "gpu": "cuda:0"},
    )

    assert isinstance(result, FrameworkConfig)
    assert result.input_data_format == "RGB"
    assert result.camera_names == ["front", "top"]
    assert result.results_3d is True
    assert result.device == "cuda:0"


def test_resolve_placeholders_pydantic_alias_with_path():
    """Test resolution with aliased Path fields."""

    class PathConfig(BaseModel):
        data_dir: Path = Field(alias="data-dir")
        output_dir: Path = Field(alias="output-dir")

    config = PathConfig(
        **{
            "data-dir": Path("<base>/data"),
            "output-dir": Path("<base>/output"),
        }
    )
    result = resolve_placeholders(config, {"base": "/home/user"})

    assert isinstance(result, PathConfig)
    assert result.data_dir == Path("/home/user/data")
    assert result.output_dir == Path("/home/user/output")


def test_resolve_placeholders_pydantic_alias_unreachable():
    """Test resolution with aliased fields containing unreachable placeholders."""

    class ConfigWithAlias(BaseModel):
        runtime_path: str = Field(alias="runtime-path")
        static_value: str

    config = ConfigWithAlias(**{"runtime-path": "<runtime_dir>/data", "static_value": "<static_val>"})
    result = resolve_placeholders(config, {"static_val": "StaticResolved"}, unreachable={"runtime_dir"})

    assert isinstance(result, ConfigWithAlias)
    assert result.runtime_path == "<runtime_dir>/data"
    assert result.static_value == "StaticResolved"


def test_resolve_placeholders_pydantic_multiple_aliases_same_model():
    """Test resolution with multiple aliased fields in the same model."""

    class MultiAliasConfig(BaseModel):
        first_field: str = Field(alias="first-field")
        second_field: str = Field(alias="second-field")
        third_field: str = Field(alias="third-field")
        regular_field: str

    config = MultiAliasConfig(
        **{
            "first-field": "<val1>",
            "second-field": "<val2>",
            "third-field": "<val3>",
            "regular_field": "<val4>",
        }
    )
    result = resolve_placeholders(config, {"val1": "V1", "val2": "V2", "val3": "V3", "val4": "V4"})

    assert isinstance(result, MultiAliasConfig)
    assert result.first_field == "V1"
    assert result.second_field == "V2"
    assert result.third_field == "V3"
    assert result.regular_field == "V4"
