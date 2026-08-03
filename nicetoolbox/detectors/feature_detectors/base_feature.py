"""
Base class for Feature Detectors.
Feature detectors run computations in-process using method detector outputs.
"""

import logging
import os
from abc import abstractmethod
from typing import Any, Dict, final

from ...configs.schemas.detectors_instances_configs import FeatureDetectorRuntime
from ...utils.base_detectors import flatten_inference_config
from ...utils.config import save_config
from ..base_detector import BaseDetector
from ..detector_outputs import DetectorOutput


class BaseFeature(BaseDetector):
    """
    Abstract base class for feature detectors.

    Feature detectors run in-process, computing derived features from
    method detector outputs.
    """

    requires_out_folder: bool = False

    @final
    def __init__(self, io, data, sequence_context, algorithm_instance: str):
        super().__init__(io, data, sequence_context, algorithm_instance)
        logging.info(
            f"Initializing feature detector {self.__class__.__name__} for instance '{self.algorithm_instance}' "
            f"and components {self.components}."
        )

        # Some common fields
        self.subjects_descr = self.data.subjects_descr
        self.viz_folders = self.compute_viz_folders(self.visualize)
        self.out_folders = self.compute_output_folders(self.requires_out_folder)
        self.result_folders = self.compute_result_folders()

        # This hook is used to allow detector initialize custom fields
        self._initialize_detector()

        # Prepare infernce config
        self.runtime = self._build_runtime()
        self.inference_config = flatten_inference_config(self.detector_config, self.runtime)

        # Pre-map legacy single component out_folder and viz_folder for backward compatibility
        if len(self.components) == 1:
            comp = self.components[0]
            self.out_folder = self.out_folders.get(comp)
            self.viz_folder = self.viz_folders.get(comp)

        # Save config for reproducibility
        for comp in self.components:
            folder = self.io.get_detector_output_folder(comp, self.algorithm_instance, "run_config")
            config_path = os.path.join(str(folder), "run_config.toml")
            save_config(self.inference_config, config_path)

        logging.info(
            f"Feature detector for component {self.components} and instance {self.algorithm_instance} initialized.\n"
        )

    def _build_runtime(self) -> FeatureDetectorRuntime:
        """
        Create standard feature detector runtime configuration.

        Subclasses MUST override this if they have a specific RuntimeConfig
        that requires additional extension fields. Currently, this used purely for logging.
        """
        return FeatureDetectorRuntime(
            result_folders=self.result_folders,
            out_folders=self.out_folders,
            viz_folders=self.viz_folders,
            algorithm=self.algorithm_instance,
            visualize=self.visualize,
            subjects_descr=self.subjects_descr,
        )

    def _build_inference_config(self) -> Dict[str, Any]:
        """
        Build flattened config dictionary (Static + Runtime).
        """
        config = self.detector_config.model_dump(by_alias=True)
        config.pop("RuntimeConfig", None)
        # Runtime fields take precedence
        config.update(self.runtime.model_dump())
        return config

    # -------------------------------------------------------------------------
    # BaseDetector Interface Implementation
    # -------------------------------------------------------------------------

    def _initialize_detector(self) -> None:
        pass

    def run(self) -> DetectorOutput:
        """
        Execute feature detector: compute(), then validate + save declared outputs.
        """
        # do the actual computation
        out = self.compute()
        # validate all outputs against defined outputs and schemas
        out.validate(self.declared_outputs)
        # validate that all npz have valid axes
        subjects, cameras, frames = self.data.canonical_axes
        out.validate_canonical_axes(subjects, cameras, frames)
        # dump results on drive
        out.save(self.io, self.algorithm_instance)
        return out

    @abstractmethod
    def compute(self) -> DetectorOutput:
        """
        Compute the feature from method detector outputs.

        Returns:
            Computed feature data (passed to visualization and post_compute)
        """
        pass
