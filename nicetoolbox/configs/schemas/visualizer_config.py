from copy import deepcopy
from pathlib import Path
from typing import Any, List

from pydantic import BaseModel, NonNegativeInt, model_validator

from ..models.models_registry import ModelsRegistry


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


class MediaVisualizeConfig(BaseModel):
    """
    Configuration for media visualization settings.
    Defines which components to visualize and video settings.
    """

    components: List[str]
    camera_position: bool
    start_frame: NonNegativeInt
    end_frame: int  # -1 means until the end
    visualize_interval: NonNegativeInt


# registry for visualizer components
_COMP_REGISTRY = ModelsRegistry(VisualizerComponentConfig)
visualizer_comp_config = _COMP_REGISTRY.register


class MediaConfig(BaseModel):
    """
    Configuration for media to be visualized.
    Includes dataset info, video info, and visualizer settings.
    Contains a collection of visualizer component configurations.
    """

    dataset_name: str
    video_name: str
    multi_view: bool
    visualize: MediaVisualizeConfig

    # visualizer components collection
    components: dict[str, BaseModel]

    # TODO: very nasty temporary hack
    # we treat any unknown field as a component config
    # parse, validate and store them into input dict
    # please move components into its own toml field
    @model_validator(mode="before")
    @classmethod
    def parse_components(cls, values):
        # because we will modify the input dict
        # it is safer to deepcopy it first
        values = deepcopy(values)
        # get all known fields except 'components'
        known = set(MediaConfig.__annotations__) - {"components"}
        comps = {k: v for k, v in values.items() if k not in known}
        # parse them and store into input dict
        values["components"] = _COMP_REGISTRY.parse_dict(comps)
        # remove original component entries from input dict
        for k in comps:
            del values[k]
        return values


class VisualizerIO(BaseModel):
    """
    Input/output folder configuration for the visualizer.
    Defines paths for input and output data storage.
    """

    dataset_folder: Path
    nice_tool_input_folder: Path
    nice_tool_output_folder: Path
    experiment_folder: Path
    experiment_video_folder: Path
    experiment_video_component: Path


# Top-level config in visualizer_config.toml
class VisualizerConfig(BaseModel):
    """
    Configuration schema for the visualizer settings.
    """

    io: VisualizerIO
    media: MediaConfig
