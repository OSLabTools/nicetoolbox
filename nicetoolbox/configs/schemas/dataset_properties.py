from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, PrivateAttr

from ..models.dict_model import DictModel


class DatasetConfigEvaluation(BaseModel):
    """
    Optional evaluation configuration for a dataset.
    Specifies annotation components and metric types to be used.
    """

    annotation_components: List[str]
    metric_types: List[str]


class DatasetConfig(BaseModel):
    """
    Configuration schema for a single dataset.
    Contains metadata and paths required for processing and evaluation.
    """

    session_IDs: List[str]
    sequence_IDs: List[str]

    cam_front: str = ""
    cam_top: str = ""
    cam_face1: str = ""
    cam_face2: str = ""

    subjects_descr: List[str]
    cam_sees_subjects: Optional[dict] = Field(default_factory=dict)

    data_input_folder: Path
    path_to_calibrations: Optional[Path] = None

    start_frame_index: NonNegativeInt
    fps: PositiveInt

    path_to_annotations: Path
    evaluation: List[DatasetConfigEvaluation]
    synonyms: Optional[dict] = Field(default_factory=dict)

    # Runtime fields
    _dataset_name: str = PrivateAttr()


# Top-level config in dataset_properties.toml
class DatasetProperties(DictModel[str, DatasetConfig]):
    """
    Dictionary of dataset configurations, keyed by unique dataset name.
    Users can define any custom datasets.
    """

    def model_post_init(self, _):
        # injecting key into each DatasetConfig
        for name, ds in self.root.items():
            ds._dataset_name = name
