import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pydantic import BaseModel

from nicetoolbox.configs.utils import model_to_dict
from nicetoolbox.detectors.in_out import SequenceIO
from nicetoolbox_core.data.array_schema import BaseArraySchema
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes, save_arrays


@dataclass(frozen=True)
class BaseDetectorOutput:
    """A declared detector output, identified by the component it belongs to."""

    component: str


@dataclass(frozen=True)
class NpzDetectorOutput(BaseDetectorOutput):
    """A declared NPZ-array output, validated against schema on save."""

    npz_key: str
    schema: BaseArraySchema | None = None


@dataclass(frozen=True)
class JsonDetectorOutput(BaseDetectorOutput):
    """A declared JSON output, for components whose results are not dense arrays."""

    schema: type[BaseModel] | None = None


class DetectorOutput:
    """A builder for a detector's produced arrays and JSONs.

    Detectors populate this in compute() via add()/add_json(), then BaseFeature.run() validates
    it against the declared outputs and saves it — grouping each component's arrays into that
    component's single NPZ (matching the toolbox data_description layout), and writing each
    JSON component to its own file.
    """

    def __init__(self) -> None:
        # (component, npz_key) -> NpzArray
        self._arrays: dict[tuple[str, str], NpzArray] = {}
        # component -> pydantic model (one JSON per component)
        self._jsons: dict[str, BaseModel] = {}

    def add(self, component: str, npz_key: str, *, data: np.ndarray, axes: NpzArrayAxes) -> "DetectorOutput":
        """Add one produced array under (component, npz_key); returns self so calls can chain."""
        return self.add_array(component, npz_key, NpzArray(data=data, axes=axes))

    def add_array(self, component: str, npz_key: str, array: NpzArray) -> "DetectorOutput":
        """Add a ready-made NpzArray under (component, npz_key); returns self so calls can chain."""
        key = (component, npz_key)
        if key in self._arrays:
            raise ValueError(f"Output '{npz_key}' for component '{component}' added more than once.")
        self._arrays[key] = array
        return self

    def add_json(self, component: str, model: BaseModel) -> "DetectorOutput":
        """Add a component's JSON output as a pydantic model; returns self so calls can chain.

        The model is kept as-is and only dumped to primitives at save time, so the declared
        schema stays available for validation (see validate) rather than being flattened away.
        """
        if component in self._jsons:
            raise ValueError(f"JSON for component '{component}' added more than once.")
        self._jsons[component] = model
        return self

    def get(self, component: str, npz_key: str) -> NpzArray:
        """Return the produced array for (component, npz_key), raising if absent."""
        key = (component, npz_key)
        if key not in self._arrays:
            raise KeyError(f"No output '{npz_key}' for component '{component}'. Produced: {list(self._arrays)}.")
        return self._arrays[key]

    def get_json(self, component: str) -> BaseModel:
        """Return the produced JSON model for a component, raising if absent."""
        if component not in self._jsons:
            raise KeyError(f"No JSON for component '{component}'. Produced: {list(self._jsons)}.")
        return self._jsons[component]

    def items(self, component: str) -> Iterable[tuple[str, NpzArray]]:
        """Iterate (npz_key, NpzArray) for a single component (e.g. for visualization)."""
        for (comp, npz_key), array in self._arrays.items():
            if comp == component:
                yield npz_key, array

    def validate(self, declared: list[BaseDetectorOutput]) -> None:
        """Check the produced outputs exactly match the declared ones.

        Arrays are keyed by (component, npz_key) and JSONs by component, so the two kinds are
        matched against their own declarations.

        Raises ValueError if a declared output is missing, an undeclared output was produced, or a
        produced array fails its NpzDetectorOutput.schema. JSON outputs need no schema check here:
        they are pydantic models, so they were validated when constructed.
        """
        declared_arrays = {(o.component, o.npz_key) for o in declared if isinstance(o, NpzDetectorOutput)}
        declared_jsons = {o.component for o in declared if isinstance(o, JsonDetectorOutput)}

        missing = sorted(declared_arrays - set(self._arrays)) + sorted(declared_jsons - set(self._jsons))
        extra = sorted(set(self._arrays) - declared_arrays) + sorted(set(self._jsons) - declared_jsons)
        if missing or extra:
            raise ValueError(
                f"Detector output mismatch. "
                f"Missing declared outputs: {missing}. "
                f"Undeclared produced outputs: {extra}."
            )

        for spec in declared:
            if isinstance(spec, NpzDetectorOutput) and spec.schema is not None:
                errors = spec.schema.validate(self._arrays[(spec.component, spec.npz_key)])
                if errors:
                    details = "\n".join(errors)
                    raise ValueError(
                        f"Output '{spec.npz_key}' (component '{spec.component}') "
                        f"failed schema validation:\n{details}"
                    )

    def validate_canonical_axes(self, subjects: list[str], cameras: list[str], frames: list[str]) -> None:
        """Check every produced array's structural axes match the sequence's canonical form.

        axis0 (subjects) and axis2 (frames) must equal the canonical lists exactly. axis1
        (cameras) must be an ordered subset of the canonical cameras — a detector may cover a
        subset (e.g. inference on 2 of 4 cameras), but never reorder or invent cameras — with
        the fused 3D pseudo-camera ["3d"] allowed as a special case. axis3/axis4 are
        detector-specific labels and not checked.

        Raises ValueError on the first mismatch — a detector that reorders or drops a subjects/
        frames axis, or emits cameras out of canonical order, is a bug.
        """
        for (component, npz_key), array in self._arrays.items():
            axes = array.axes
            where = f"output '{npz_key}' (component '{component}')"

            if axes.subjects != subjects:
                raise ValueError(f"{where}: subjects axis {axes.subjects} != canonical {subjects}.")
            if axes.frames != frames:
                raise ValueError(f"{where}: frames axis (len {len(axes.frames)}) != canonical (len {len(frames)}).")
            # Cameras: an ordered subset of canonical (keeps canonical order), or the 3D slot.
            canonical_subset = [cam for cam in cameras if cam in set(axes.cameras)]
            if axes.cameras != canonical_subset and axes.cameras != ["3d"]:
                raise ValueError(
                    f"{where}: cameras axis {axes.cameras} is not an ordered subset of canonical "
                    f"{cameras} (nor ['3d'])."
                )

    def save(self, io: SequenceIO, algorithm_instance: str) -> None:
        """Save produced outputs under each component's result folder.

        Array outputs are grouped into '<result_folder>/<algorithm_instance>.npz' (one NPZ per
        component). JSON outputs are written to '<result_folder>/<algorithm_instance>.json' —
        saved as-is, so downstream consumers (ELAN connector, evaluation transcript loaders)
        keep reading the same on-disk format.
        """
        by_component: dict[str, dict[str, NpzArray]] = defaultdict(dict)
        for (component, npz_key), array in self._arrays.items():
            by_component[component][npz_key] = array

        for component, arrays in by_component.items():
            result_folder = io.get_detector_output_folder(component, algorithm_instance, "result")
            npz_path = result_folder / f"{algorithm_instance}.npz"
            save_arrays(arrays, npz_path)
            logging.info(f"Saved output for component '{component}' ({list(arrays)}) to '{npz_path}'.")

        for component, model in self._jsons.items():
            result_folder = io.get_detector_output_folder(component, algorithm_instance, "result")
            json_path = result_folder / f"{algorithm_instance}.json"
            with open(json_path, "w") as f:
                json.dump(model_to_dict(model), f, indent=4)
            logging.info(f"Saved JSON output for component '{component}' to '{json_path}'.")
