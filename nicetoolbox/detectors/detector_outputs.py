import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from nicetoolbox.detectors.in_out import SequenceIO
from nicetoolbox_core.data.array_schema import BaseArraySchema
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes, save_arrays


@dataclass(frozen=True)
class BaseDetectorOutput:
    """A declared detector output, keyed by the component it belongs to and its npz_key."""

    component: str
    npz_key: str


@dataclass(frozen=True)
class NpzDetectorOutput(BaseDetectorOutput):
    """A declared NPZ-array output, validated against schema on save."""

    schema: BaseArraySchema | None = None


class DetectorOutput:
    """A builder for a detector's produced arrays.

    Detectors populate this in compute() via add(), then BaseFeature.run() validates it
    against the declared outputs and saves it — grouping each component's arrays into that
    component's single NPZ, matching the toolbox data_description layout.
    """

    def __init__(self) -> None:
        # (component, npz_key) -> NpzArray
        self._arrays: dict[tuple[str, str], NpzArray] = {}

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

    def get(self, component: str, npz_key: str) -> NpzArray:
        """Return the produced array for (component, npz_key), raising if absent."""
        key = (component, npz_key)
        if key not in self._arrays:
            raise KeyError(f"No output '{npz_key}' for component '{component}'. Produced: {list(self._arrays)}.")
        return self._arrays[key]

    def items(self, component: str) -> Iterable[tuple[str, NpzArray]]:
        """Iterate (npz_key, NpzArray) for a single component (e.g. for visualization)."""
        for (comp, npz_key), array in self._arrays.items():
            if comp == component:
                yield npz_key, array

    def validate(self, declared: list[BaseDetectorOutput]) -> None:
        """Check the produced arrays exactly match the declared outputs.

        Raises ValueError if a declared output is missing, an undeclared output was produced,
        or a produced array fails its NpzDetectorOutput.schema.
        """
        declared_keys = {(o.component, o.npz_key) for o in declared}
        produced_keys = set(self._arrays)

        missing = declared_keys - produced_keys
        extra = produced_keys - declared_keys
        if missing or extra:
            raise ValueError(
                f"Detector output mismatch. "
                f"Missing declared outputs: {sorted(missing)}. "
                f"Undeclared produced outputs: {sorted(extra)}."
            )

        for spec in declared:
            if isinstance(spec, NpzDetectorOutput) and spec.schema is not None:
                array = self._arrays[(spec.component, spec.npz_key)]
                errors = spec.schema.validate(array)
                if errors:
                    details = "\n".join(errors)
                    raise ValueError(
                        f"Output '{spec.npz_key}' (component '{spec.component}') failed schema validation:\n{details}"
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
        """Save produced arrays, one NPZ per component under its result folder.

        Each component's arrays are grouped into '<result_folder>/<algorithm_instance>.npz',
        matching the existing per-detector output layout.
        """
        by_component: dict[str, dict[str, NpzArray]] = defaultdict(dict)
        for (component, npz_key), array in self._arrays.items():
            by_component[component][npz_key] = array

        for component, arrays in by_component.items():
            result_folder = io.get_detector_output_folder(component, algorithm_instance, "result")
            npz_path = result_folder / f"{algorithm_instance}.npz"
            save_arrays(arrays, npz_path)
            logging.info(f"Saved output for component '{component}' ({list(arrays)}) to '{npz_path}'.")
