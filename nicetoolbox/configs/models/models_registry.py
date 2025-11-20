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
            raise ValueError(f"Unknown schema for {name}")
        return model.model_validate(data)

    def parse_dict(self, data: Dict[str, Any]) -> Dict[str, BaseModel]:
        return {k: self.parse(k, v) for k, v in data.items()}

    def get_model(self, name: str) -> Optional[Type[BaseModel]]:
        return self._models.get(name, self._default_model)
