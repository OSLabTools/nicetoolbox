import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nicetoolbox.configs.schemas.detectors_instances_configs import DetectorInputConfig
from nicetoolbox.detectors.in_out import SequenceIO
from nicetoolbox.detectors.subsequence_context import SubsequenceContext
from nicetoolbox_core.data.array_schema import BaseArraySchema
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes, load_array_from_path

from ..configs.schemas.detectors_instances_configs import BaseAlgorithmConfig


@dataclass(frozen=True)
class BaseDetectorInput:
    """A declared detector input, keyed by its upstream component and a local handle."""

    component: str
    name: str  # local handle used in compute(), e.g. "pose"
    optional: bool = False


@dataclass(frozen=True)
class NpzDetectorInput(BaseDetectorInput):
    """A declared NPZ-array input, validated against schema when loaded."""

    schema: BaseArraySchema | None = None


@dataclass
class ResolvedInput:
    """A loaded, validated detector input ready for use in compute()."""

    name: str
    component: str
    algorithm: str
    npz_key: str | None
    upstream_config: BaseAlgorithmConfig
    array: NpzArray | None
    file_path: Path

    @property
    def data(self) -> np.ndarray:
        """Shortcut to the underlying numpy array (raises if this is a non-array input)."""
        if self.array is None:
            raise ValueError(f"Input '{self.name}' has no array (optional-absent or non-npz).")
        return self.array.data

    @property
    def axes(self) -> NpzArrayAxes:
        """Shortcut to the array's named axes (raises if this is a non-array input)."""
        if self.array is None:
            raise ValueError(f"Input '{self.name}' has no array (optional-absent or non-npz).")
        return self.array.axes


# =========================================================================
# Helper functions
# =========================================================================


def _available_npz_keys(npz_path) -> list[str]:
    """Best-effort list of array keys declared in an NPZ's data_description (for error messages)."""
    try:
        with np.load(npz_path, allow_pickle=True) as f:
            if "data_description" in f.files:
                return list(f["data_description"].item().keys())
            return [name for name in f.files if name != "data_description"]
    except Exception:
        return []


def _npz_path(cfg: DetectorInputConfig, io: SequenceIO) -> Path:
    """Locate the file an upstream detector wrote for one configured input."""
    result_folder = io.get_detector_output_folder(cfg.component, cfg.algorithm, "result")
    return result_folder / f"{cfg.algorithm}.npz"


def _load_npz_input(spec: NpzDetectorInput, cfg: DetectorInputConfig, npz_path: Path):
    """Load and schema-validate one NPZ input.

    `optional` governs only whether the input *block* may be omitted from the config (handled in
    load_detector_inputs). Once a block is configured, it must resolve fully: the NPZ file and the
    requested npz_key must exist, so a missing file or key is always an error here.
    """
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Input '{spec.name}' (component '{cfg.component}', algorithm '{cfg.algorithm}') "
            f"expects NPZ at '{npz_path}', but it does not exist."
        )

    # Returns None when npz_key is absent from data_description.
    array = load_array_from_path(npz_path, cfg.npz_key)
    if array is None:
        available = _available_npz_keys(npz_path)
        raise ValueError(
            f"Input '{spec.name}' key '{cfg.npz_key}' is absent from '{npz_path}' Available keys: {available}."
        )

    if spec.schema is not None:
        errors = spec.schema.validate(array)
        if errors:
            details = "\n".join(errors)
            raise ValueError(
                f"Input '{spec.name}' from '{npz_path}' (key '{cfg.npz_key}') failed schema validation:\n{details}"
            )

    return array


def load_detector_inputs(
    declared: list[BaseDetectorInput],
    inputs_cfg: dict[str, DetectorInputConfig],
    io: SequenceIO,
    subsequence_context: SubsequenceContext,
) -> dict[str, ResolvedInput]:
    """Resolve a detector's declared inputs into loaded, validated arrays.

    Detectors consume whole upstream arrays (no per-input axis filtering), so the full
    array for the requested npz_key is loaded and validated as-is.

    Args:
        declared: The detector's declared inputs (from resolve_inputs()).
        inputs_cfg: The detector config's inputs table, keyed by local handle.
        io: SequenceIO, used to locate upstream detector result folders.
        subsequence_context: Provides the upstream detector config per algorithm.

    Returns:
        Mapping of input handle -> ResolvedInput, one entry per configured input. An optional
        input that is not present in inputs_cfg is omitted from the map; every entry that IS
        present has a fully loaded array (never None).

    Raises:
        KeyError: If a required (non-optional) input has no matching entry in inputs_cfg.
        FileNotFoundError: If a configured input's NPZ file is missing.
        ValueError: If a configured input's npz_key is absent, or it fails schema validation.
    """
    resolved: dict[str, ResolvedInput] = {}
    for spec in declared:
        input_cfg = inputs_cfg.get(spec.name)
        if input_cfg is None:
            if spec.optional:
                # Optional input the user chose not to wire: leave it out of the resolved map.
                logging.info(f"Optional input '{spec.name}' has no config entry; skipping.")
                continue
            raise KeyError(
                f"Detector input '{spec.name}' (component '{spec.component}') is declared in code "
                f"but missing from the config's [inputs] table. Available: {list(inputs_cfg)}."
            )

        array, file_path = None, None
        if isinstance(spec, NpzDetectorInput):
            file_path = _npz_path(input_cfg, io)
            array = _load_npz_input(spec, input_cfg, file_path)
        # TODO: load parsed json/other comp type here? as dict? as pydantic?
        # TODO: at least pass json path

        upstream_config = subsequence_context.get_detector_config(input_cfg.algorithm)
        resolved[spec.name] = ResolvedInput(
            name=spec.name,
            component=input_cfg.component,
            algorithm=input_cfg.algorithm,
            npz_key=input_cfg.npz_key,
            upstream_config=upstream_config,
            array=array,
            file_path=file_path,
        )

    return resolved
