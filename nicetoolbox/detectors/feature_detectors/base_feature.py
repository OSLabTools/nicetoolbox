"""
Base class for Feature Detectors.
Feature detectors run computations in-process using method detector outputs.
"""

import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple, final

from ...configs.schemas.detectors_instances_configs import FeatureDetectorRuntime
from ...utils.base_detectors import flatten_inference_config
from ...utils.config import save_config
from ..base_detector import BaseDetector


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
        # Legacy input paths for not-yet-migrated detectors (still on `input_detector_names`).
        # Migrated detectors read `self.loaded_inputs` (resolved in BaseDetector) instead.
        self.input_map = self._legacy_input_map()
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

    def _legacy_input_map(self) -> Dict[Tuple[str, str], Path]:
        """Build the legacy {(component, algorithm): npz_path} map from `input_detector_names`.

        Kept only for detectors not yet migrated to declarative `inputs`. Empty for migrated
        detectors (which declare `inputs` and use `self.loaded_inputs`).
        """
        input_map: Dict[Tuple[str, str], Path] = {}
        for component, algorithm in getattr(self.detector_config, "input_detector_names", None) or []:
            input_path = self.io.get_detector_output_folder(component, algorithm, "result")
            input_map[(component, algorithm)] = input_path / f"{algorithm}.npz"
        return input_map

    def get_input_file(self, component: str, algorithm: str) -> Path:
        """Legacy accessor for a not-yet-migrated detector's upstream NPZ path."""
        return self.input_map[(component, algorithm)]

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

    def run(self) -> Any:
        """
        Execute feature detector: compute(), then validate + save declared outputs.

        For detectors that declare `outputs`, compute() returns a DetectorOutput which is
        validated against the declaration and saved here (the detector no longer saves itself).
        Detectors that declare no outputs keep the legacy path: compute() saves its own NPZ and
        its return value is passed straight to visualization().

        Returns the computed data (a DetectorOutput for migrated detectors), for visualization.
        """
        out = self.compute()
        if not self.declared_outputs:
            return out  # legacy: compute() already saved itself
        out.validate(self.declared_outputs)
        out.validate_canonical_axes(*self._canonical_axes())
        out.save(self.io, self.algorithm_instance)
        return out

    def _canonical_axes(self) -> tuple[list[str], list[str], list[str]]:
        """The sequence's canonical (subjects, cameras, frames) axis labels.

        Cameras are sorted to match `resolve_filter`'s ordering (see config_handler), so every
        detector agrees on camera order. Frames are the zero-padded subsequence frame indices.
        """
        subjects = list(self.subsequence_context.subjects_descr)
        cameras = sorted(self.subsequence_context.all_camera_names)
        # TODO: hardcoded assumptions about the frame names, move somewhere else
        frames = [str(i).zfill(9) for i in range(self.data.video_length_frames)]
        return subjects, cameras, frames

    @abstractmethod
    def compute(self) -> Any:
        """
        Compute the feature from method detector outputs.

        Returns:
            Computed feature data (passed to visualization and post_compute)
        """
        pass
