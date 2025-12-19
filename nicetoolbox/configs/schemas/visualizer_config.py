from copy import deepcopy
from pathlib import Path
from typing import List

from pydantic import BaseModel, NonNegativeInt, model_serializer, model_validator

from .visualizer_comp_configs import COMP_REGISTRY


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


class MediaConfig(BaseModel):
    """
    Configuration for media to be visualized.
    Includes dataset info, video info, and visualizer settings.
    Contains a collection of visualizer component configurations.
    """

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
        values["components"] = COMP_REGISTRY.parse_dict(comps)
        # remove original component entries from input dict
        for k in comps:
            del values[k]
        return values

    # now we do the opposite
    @model_serializer(mode="wrap")
    def serialize_components(self, serializer):
        # get the default serialized data
        data = serializer(self)
        # extract components dict
        components = data.pop("components", {})
        # promote each component to top-level
        for comp_name, comp_config in components.items():
            data[comp_name] = comp_config
        return data


class VisualizerIO(BaseModel):
    """
    Input/output folder configuration for the visualizer.
    Defines paths for input and output data storage.
    """

    dataset_folder: Path
    dataset_name: str
    video_name: str
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
