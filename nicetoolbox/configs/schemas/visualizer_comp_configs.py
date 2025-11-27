from typing import Any, Dict, List

from pydantic import BaseModel

from ..models.models_registry import ModelsRegistry

# ================================================
#                 BASE CONFIG
# ================================================


class AppearanceConfig(BaseModel):
    """
    Appearance settings for visualizer components.
    Stores colors and other visual attributes.
    """

    colors: List[Any]  # TODO: inconsistent dimensions
    radii: dict[str, Any]


class VisualizerComponentConfig(BaseModel):
    """
    Default configuration for individual visualizer components.
    Defines algorithms, detector data and appearance.
    """

    algorithms: List[str]
    canvas: dict[str, List[str]]
    appearance: AppearanceConfig


# registry for visualizer components
COMP_REGISTRY = ModelsRegistry(VisualizerComponentConfig)
visualizer_comp_config = COMP_REGISTRY.register


# ================================================
#                 CUSTOM CONFIGS
# ================================================


@visualizer_comp_config("kinematics")
class KinematicsConfig(BaseModel):
    algorithms: List[str]
    canvas: dict[str, List[str]]
    joints: Dict[str, List[str]]
