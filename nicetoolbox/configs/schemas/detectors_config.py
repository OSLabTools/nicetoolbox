from copy import deepcopy

from pydantic import BaseModel, model_validator

from ..models.models_registry import ModelsRegistry

# registries for detectors and frameworks
_DETECTORS_REGISTRY = ModelsRegistry()
_FRAMEWORKS_REGISTRY = ModelsRegistry()
detector_config = _DETECTORS_REGISTRY.register
framework_config = _FRAMEWORKS_REGISTRY.register


# Top-level config in detectors_config.toml
class DetectorsConfig(BaseModel):
    """
    Contains all detector and framework configurations.
    """

    algorithms: dict[str, BaseModel]
    frameworks: dict[str, BaseModel]

    @staticmethod
    def __resolve_frameworks_inheritance(frms, algos):
        for name, alg in algos.items():
            if "framework" not in alg:
                continue
            # check that framework exists
            frm = alg["framework"]
            if frm not in frms:
                raise ValueError(f"Invalid framework {frm} for algorithm {name}.")
            alg.update(frms[frm])

    @model_validator(mode="before")
    @classmethod
    def parse_frameworks_and_algos(cls, values):
        # because we will modify the input dict
        # it is safer to deepcopy it first
        values = deepcopy(values)
        # first we parse frameworks
        frms = values.get("frameworks", {})
        values["frameworks"] = _FRAMEWORKS_REGISTRY.parse_dict(frms)
        # next we "patch" algorithms with framework information
        # we need to this now because algorithms may inherit from frameworks
        algos = values.get("algorithms", {})
        cls.__resolve_frameworks_inheritance(frms, algos)
        # now we parse algorithms
        values["algorithms"] = _DETECTORS_REGISTRY.parse_dict(algos)

        return values
