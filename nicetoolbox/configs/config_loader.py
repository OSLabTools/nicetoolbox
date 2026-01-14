from typing import Any, Optional

from pydantic import BaseModel

from .config_handler import ModelT, load_config
from .placeholders import PLACEHOLDERS_TYPE, resolve_placeholders
from .utils import dict_to_model, merge_dicts, model_to_dict


class ConfigLoader:
    """
    Configuration loader with placeholder resolution and validation.

    Attributes:
        auto_placeholders (dict[str, PLACEHOLDERS_TYPE]): Placeholders initialized from
            functions, available to all configurations.
        runtime_placeholders (set[str]): Placeholder names that should remain unresolved
            until runtime (e.g., video_length, session_id).
        global_placeholders (dict[str, PLACEHOLDERS_TYPE]): Placeholders made available
            to all subsequent config loads, built up by adding contexts from loaded
            configs.
    """

    auto_placeholders: dict[str, PLACEHOLDERS_TYPE]
    runtime_placeholders: set[str]
    global_placeholders: dict[str, PLACEHOLDERS_TYPE]

    def __init__(self, auto: dict[str, PLACEHOLDERS_TYPE], runtime: set[str]):
        self.auto_placeholders = auto
        self.runtime_placeholders = runtime
        self.global_placeholders = {}

    def load_config(self, path: str, schema: type[ModelT], ignore_auto_and_global=False) -> ModelT:
        """
        Loads a TOML configuration file, resolves all placeholders using available
        contexts (auto, global and runtime) and validates the result against a Pydantic
        schema. The configuration is returned as a validated Pydantic model instance.

        Args:
            path (str): Path to the configuration file.
            schema (type[ModelT]): Pydantic model class to validate the configuration.
            ignore_auto_and_global (bool, optional): If True, excludes global and auto
                placeholders from resolution context. Used for self-contained configs
                like experiment that bundle their own dependencies. Defaults to False.

        Returns:
            ModelT: Validated Pydantic model instance populated with resolved
                config data.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            NotImplementedError: If the file type is not .toml
            toml.TomlDecodeError: If the TOML file has syntax errors.
            ValueError: If placeholders cannot be resolved due to missing
                values, circular dependencies, or exceed max iterations.
            KeyError: If there's a collision between placeholder contexts
                or between local config fields and placeholder names.
            ConfigValidationError: If the resolved configuration fails
                schema validation).
        """
        cfg_raw = load_config(path)
        cfg_raw_resolved = self.resolve(cfg_raw, None, ignore_auto_and_global)
        cfg = dict_to_model(cfg_raw_resolved, schema)
        return cfg

    def extend_global_ctx(self, add_ctx: dict[str, PLACEHOLDERS_TYPE] | BaseModel) -> None:
        """
        Extends the global placeholder context that will be available to all
        future load_config() calls. Typically used to register values from
        already-loaded configs (e.g., paths from machine_specific config).

        Args:
            add_ctx (dict[str, PLACEHOLDERS_TYPE]): Dictionary of placeholder key-value
            pairs to add to the global context.

        Raises:
            KeyError: If any keys in add_ctx already exist in global_placeholders.
        """
        if isinstance(add_ctx, BaseModel):
            add_ctx = model_to_dict(add_ctx)
        self.global_placeholders = merge_dicts(self.global_placeholders, add_ctx)

    def resolve(
        self,
        config: Any,
        runtime_ctx: Optional[dict[str, PLACEHOLDERS_TYPE]] = None,
        ignore_auto_and_global: bool = False,
    ) -> Any:
        r"""
        Recursively processes the configuration to replace all placeholder
        strings (formatted as \<key\>) with their corresponding values from
        the combined context (auto, global and runtime).

        Args:
            config: Configuration data to resolve. Can be dict, list,
                Pydantic model, string or any nested combination.
            runtime_ctx (Optional[dict[str, PLACEHOLDERS_TYPE]], optional): Additional
                placeholders to provide at runtime (e.g., current video name,
                session ID, active detector).
            ignore_auto_and_global (bool, optional): If True, excludes global and auto
                placeholders from resolution context. Used for self-contained configs
                like experiment that bundle their own dependencies. Defaults to False.

        Returns:
            Copy of config with all resolvable placeholders replaced.
                Type matches input type.

        Raises:
            ValueError: If placeholders cannot be resolved due to missing
                values, circular dependencies, or max iterations exceeded.
            KeyError: If there's a collision between placeholder contexts
                (auto, global, runtime) or between local config fields and
                placeholder names.
        """
        # prepare external context (auto, global and runtime)
        ctx = {}
        unreachable = self.runtime_placeholders
        if not ignore_auto_and_global:
            ctx = merge_dicts(self.auto_placeholders, self.global_placeholders)
        if runtime_ctx:
            ctx = merge_dicts(ctx, runtime_ctx)
            # if provided some runtime ctx - it should be reachable now
            unreachable = self.runtime_placeholders - set(runtime_ctx)

        # resolve all placeholders with provided context
        config_res = resolve_placeholders(config, ctx, unreachable)
        return config_res
