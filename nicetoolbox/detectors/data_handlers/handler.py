"""
Abstract base class for modality-specific data handlers.

Each handler is responsible for preparing data of a specific modality
(video/frames, audio) from source files into the nicetoolbox_input folder.
"""

from abc import ABC, abstractmethod
from typing import Any

from ...configs.video_runtime_config import SequenceRuntimeConfig
from ..in_out import SequenceIO


class BaseModalityHandler(ABC):
    """
    Abstract base class for modality-specific data handlers.

    Handlers are responsible for:
    - Detecting if source data exists for their modality
    - Extracting/copying data to nicetoolbox_input folder
    - Providing a recipe for downstream loaders
    """

    def __init__(self, io: SequenceIO, sequence_context: SequenceRuntimeConfig):
        self.io = io
        self.sequence_context = sequence_context

        # References for child classes:
        self.nice_input_folder = self.io.nice_input_folder
        self.session_id = self.sequence_context.session_id
        self.sequence_id = self.sequence_context.sequence_id
        self.subjects_descr = self.sequence_context.subjects_descr
        self.all_camera_names = self.sequence_context.all_camera_names
        self.dataset_properties = self.sequence_context.dataset_properties

        self._available = False

    @abstractmethod
    def prepare(self) -> None:
        """
        Prepare data for this modality.
        """
        pass

    @abstractmethod
    def get_recipe(self) -> Any:
        """
        Get the input recipe for this modality's data loader.

        Returns:
            Recipe dictionary for constructing a data loader, or None if unavailable.
        """
        pass

    @property
    @abstractmethod
    def modality_name(self) -> str:  # TODO: Do we need this?? Currently no
        """Return the name of this modality (e.g., 'video', 'audio')."""
        pass

    @property
    def is_available(self) -> bool:
        """Whether data for this modality was successfully prepared."""
        return self._available
