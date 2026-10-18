from typing import Any, Dict, Optional, Type

from pydantic import BaseModel


class ModelsRegistry:
    """
    Dictionary-like registry for Pydantic models.
    Used for dynamic model selection based
    on a string identifier (i.e. detectors configs)
    """

    _models: Dict[str, Type[BaseModel]]
    _default_model: Optional[Type[BaseModel]]

    def __init__(self, default_model: Optional[Type[BaseModel]] = None):
        self._models = {}
        self._default_model = default_model

    def register(self, name: str):
        def _wrap(model: Type[BaseModel]) -> Type[BaseModel]:
            self._models[name] = model
            return model

        return _wrap

    def parse(self, name: str, data: Dict[str, Any]) -> BaseModel:
        model = self._models.get(name, self._default_model)
        if model is None:
            known = ", ".join(sorted(self._models.keys()))
            raise ValueError(
                f"Unknown schema for {name!r}. "
                "If you added a new detector, reinstall nicetoolbox "
                "(e.g. `pip install -e .` or re-run `install_windows.bat` / `install_linux.sh`). "
                f"Registered keys: {known}"
            )
        return model.model_validate(data)

    def parse_dict(self, data: Dict[str, Any]) -> Dict[str, BaseModel]:
        return {k: self.parse(k, v) for k, v in data.items()}

    def parse_dict_by_type(self, data: Dict[str, Any], type_field_name: str) -> Dict[str, BaseModel]:
        parsed: Dict[str, BaseModel] = {}
        for instance_name, item in data.items():
            # Already a parsed model (e.g. on a re-validation pass) — pass through.
            if isinstance(item, BaseModel):
                parsed[instance_name] = item
                continue
            if type_field_name not in item:
                raise ValueError(f"'{instance_name}' is missing required '{type_field_name}' field.")
            type_value = item[type_field_name]
            parsed[instance_name] = self.parse(type_value, item)
        return parsed

    def get_model(self, name: str) -> Optional[Type[BaseModel]]:
        return self._models.get(name, self._default_model)
