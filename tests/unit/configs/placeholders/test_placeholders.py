"""
Unit tests for placeholder resolution.
"""

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
