"""
Base class for all detectors (method and feature).
Provides common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..configs.schemas.detectors_instances_configs import BaseAlgorithmConfig
from .data import SequenceData
from .detector_inputs import BaseDetectorInput, ResolvedInput, load_detector_inputs
from .detector_outputs import BaseDetectorOutput
from .in_out import SequenceIO
from .subsequence_context import SubsequenceContext


class BaseDetector(ABC):
    """
    Abstract base class for ALL detectors.

    Defines the common interface that both method and feature detectors implement.
    This enables a unified detector loop in main.py.
    """

    # Each detector should have an unique algorithm type
    algorithm_type: str
    # User-defined instance name from TOML key
    algorithm_instance: str

    # Instance attributes set during initialization
    data: SequenceData
    io: SequenceIO
    subsequence_context: SubsequenceContext
    detector_config: BaseAlgorithmConfig

    # Class attributes to be defined by subclasses
    inference_config: Any
    # TODO: drop components, derive from outputs
    components: List[str]

    # Declarative upstream inputs
    inputs: List[BaseDetectorInput] = []
    # Loaded upstream inputs during init
    loaded_inputs: Dict[str, ResolvedInput]

    # Declarative outputs this detector produces
    outputs: List[BaseDetectorOutput] = []
    # Resolved outputs during init
    declared_outputs: List[BaseDetectorOutput]

    # Additional attributes
    visualize: bool

    def __init__(
        self,
        io: SequenceIO,
        data: SequenceData,
        subsequence_context: SubsequenceContext,
        algorithm_instance: str,
    ) -> None:
        """
        Initialize base detector with references.

        Subclasses should call super().__init__() and set inference_config.
        """
        self.io = io
        self.data = data
        self.subsequence_context = subsequence_context
        self.algorithm_instance = algorithm_instance
        self.detector_config = subsequence_context.get_detector_config(algorithm_instance)
        self.visualize = getattr(self.detector_config, "visualize", False)
        # Allow detector configs (e.g. MMPose 2D) to declare components per-instance.
        # Falls back to the subclass's class-level `components` attribute.
        config_components = getattr(self.detector_config, "components", None)
        if config_components:
            self.components = list(config_components)

        # Input tracks validation
        self._check_declared_tracks_available()

        # Resolve declared upstream other detectors inputs into loaded, validated arrays.
        self.inputs = self.resolve_inputs()
        self.loaded_inputs = load_detector_inputs(
            declared=self.inputs,
            inputs_cfg=self.detector_config.inputs,
            io=self.io,
            subsequence_context=self.subsequence_context,
        )

        # Resolve declared outputs (validated against what compute() produces, in run()).
        self.declared_outputs = self.resolve_outputs()

    def _check_declared_tracks_available(self) -> None:
        """
        Fail fast when a detector's declared camera/track filter resolves to an empty set
        for this sequence. The filter has already been intersected against the sequence's
        available tracks in the config handler, so an empty list here means none of the
        requested tracks exist for this sequence.
        """
        if hasattr(self.detector_config, "camera_names") and not self.detector_config.camera_names:
            raise ValueError(
                f"Detector '{self.algorithm_instance}': none of the requested cameras are available "
                f"for this sequence. Available cameras: {self.subsequence_context.all_camera_names}."
            )
        if hasattr(self.detector_config, "track_names") and not self.detector_config.track_names:
            raise ValueError(
                f"Detector '{self.algorithm_instance}': none of the requested audio tracks are "
                f"available for this sequence."
            )

    def resolve_inputs(self) -> List[BaseDetectorInput]:
        """Return the inputs to resolve for this detector.

        Defaults to the class-level `inputs`. Override to vary inputs by config
        (e.g. add an optional input only when a flag is set).
        """
        return list(self.inputs)

    def resolve_outputs(self) -> List[BaseDetectorOutput]:
        """Return the outputs this detector declares it will produce.

        Defaults to the class-level `outputs`. Override to vary outputs by config
        (e.g. pick the npz_key/schema based on a per-instance dimension).
        """
        return list(self.outputs)

    @abstractmethod
    def run(self) -> Optional[Any]:
        """
        Execute the detector's main computation.

        For method detectors: runs subprocess inference + post_inference()
        For feature detectors: runs compute()

        Returns:
            Optional data for visualization (feature detectors currently return computed data)
        """
        pass

    @abstractmethod
    def visualization(self, data: Any) -> None:
        """
        Visualize detector output.

        Args:
            data: Output from run() or external data source
        """
        pass

    # -------------------------------------------------------------------------
    # Shared Helper Methods
    # -------------------------------------------------------------------------

    @property
    def predictions_mapping(self):
        """Access predictions mapping from runtime config."""
        return self.subsequence_context.predictions_mapping

    def compute_result_folders(self) -> Dict[str, str]:
        """Compute result folders for all components."""
        return {
            comp: str(self.io.get_detector_output_folder(comp, self.algorithm_instance, "result"))
            for comp in self.components
        }

    def compute_output_folders(self, requires_out_folder: bool) -> Dict[str, str]:
        """Compute extra output folders for all components."""
        if requires_out_folder:
            return {
                comp: str(self.io.get_detector_output_folder(comp, self.algorithm_instance, "output"))
                for comp in self.components
            }
        return {}

    def compute_viz_folders(self, visualize: bool) -> Dict[str, str]:
        """Compute visualization folders for all components."""
        if visualize:
            return {
                comp: str(self.io.get_detector_output_folder(comp, self.algorithm_instance, "visualization"))
                for comp in self.components
            }
        return {}

    def __str__(self) -> str:
        config_name = type(self.inference_config).__name__
        return f"Detector: {self.algorithm_instance}\n  Components: {self.components}\n  Config: {config_name}\n"
