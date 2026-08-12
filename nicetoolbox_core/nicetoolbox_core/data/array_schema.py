from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .loaded_array import NpzArray, NpzArrayAxes


class BaseArraySchema(ABC):
    """Base validation for NpzArray."""

    @abstractmethod
    def validate(self, array: NpzArray) -> list[str]:
        """Return a list of validation errors. Empty when the array matches the schema."""
        ...


@dataclass(frozen=True)
class ArraySchema(BaseArraySchema):
    """Most common array schema, validates axis names and dtype."""

    labels_columns: tuple[str, ...] = field(default_factory=tuple)  # expected axis3 names (labels)
    data_columns: tuple[str, ...] = field(default_factory=tuple)  # expected axis4 names (data)
    dtype: type = np.floating

    def labels_list(self) -> list[str]:
        """The axis3 (labels) column names as a list, for NpzArrayAxes construction."""
        return list(self.labels_columns)

    def data(self) -> list[str]:
        """The axis4 (data) column names as a list, for NpzArrayAxes construction."""
        return list(self.data_columns)

    def make_axes(
        self,
        subjects: list[str],
        cameras: list[str],
        frames: list[str],
        labels: list[str] | None = None,
        data: list[str] | None = None,
    ) -> NpzArrayAxes:
        """Build an NpzArrayAxes from the structural axes, filling labels/data from this schema.

        The schema owns the axis3 (labels) and axis4 (data) column names. Pass `labels`/`data`
        only for the axes the schema leaves unspecified; passing one the schema already defines
        raises, so the label names never live in two places.
        """
        if labels is not None and self.labels_columns:
            raise ValueError("labels are defined by the schema; do not also pass them.")
        if data is not None and self.data_columns:
            raise ValueError("data columns are defined by the schema; do not also pass them.")

        final_labels = self.labels_list() if self.labels_columns else (labels or [])
        final_data = self.data() if self.data_columns else (data or [])
        # axis3 (labels) is mandatory: without it the array is (S, C, F, 0[, D]) — a zero-length axis.
        if not final_labels:
            raise ValueError("labels are required; an array must have a non-empty labels (axis3) axis.")

        return NpzArrayAxes(subjects=subjects, cameras=cameras, frames=frames, labels=final_labels, data=final_data)

    def validate(self, array: NpzArray) -> list[str]:
        errors: list[str] = []

        if self.labels_columns and tuple(array.axes.labels) != self.labels_columns:
            errors.append(f"labels axis must equal {list(self.labels_columns)}, got {array.axes.labels}")

        if self.data_columns and tuple(array.axes.data) != self.data_columns:
            errors.append(f"data axis must equal {list(self.data_columns)}, got {array.axes.data}")

        if not np.issubdtype(array.data.dtype, self.dtype):
            errors.append(f"data dtype must be a subtype of {self.dtype}, got {array.data.dtype}")

        return errors


@dataclass(frozen=True)
class AnyOf(BaseArraySchema):
    """Passes if the array matches ANY of the given schemas."""

    options: tuple[BaseArraySchema, ...] = field(default_factory=tuple)

    def __init__(self, *options: BaseArraySchema):
        object.__setattr__(self, "options", options)

    def validate(self, array: NpzArray) -> list[str]:
        option_errors: list[str] = []
        for i, option in enumerate(self.options, start=1):
            errors = option.validate(array)
            if not errors:
                return []
            option_errors.append(f"  option {i}: {'; '.join(errors)}")
        joined = "\n".join(option_errors)
        return [f"array matched none of the allowed schemas:\n{joined}"]


# =============================================================================
# Common Array Schemas
# =============================================================================

FLOAT = ArraySchema(
    # can have any dimensions and columns
    dtype=np.floating,
)

# 2d vectors
VECTOR_2D = ArraySchema(
    labels_columns=("coordinate_x", "coordinate_y"),
    dtype=np.floating,
)

VECTOR_2D_PER_LABEL = ArraySchema(
    data_columns=("coordinate_x", "coordinate_y"),
    dtype=np.floating,
)

# 2d vectors + confidence
VECTOR_2D_CONF = ArraySchema(
    labels_columns=("coordinate_x", "coordinate_y", "confidence_score"),
    dtype=np.floating,
)

VECTOR_2D_CONF_PER_LABEL = ArraySchema(
    data_columns=("coordinate_x", "coordinate_y", "confidence_score"),
    dtype=np.floating,
)

# 3d vectors
VECTOR_3D = ArraySchema(
    labels_columns=("coordinate_x", "coordinate_y", "coordinate_z"),
    dtype=np.floating,
)

VECTOR_3D_PER_LABEL = ArraySchema(
    data_columns=("coordinate_x", "coordinate_y", "coordinate_z"),
    dtype=np.floating,
)

# 3d vectors + confidence
VECTOR_3D_CONF = ArraySchema(
    labels_columns=("coordinate_x", "coordinate_y", "coordinate_z", "confidence_score"),
    dtype=np.floating,
)

VECTOR_3D_CONF_PER_LABEL = ArraySchema(
    data_columns=("coordinate_x", "coordinate_y", "coordinate_z", "confidence_score"),
    dtype=np.floating,
)

# 2d bounding box + confidence
BBOX_2D_CONF = ArraySchema(
    labels_columns=("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y", "confidence_score"),
    dtype=np.floating,
)

# =============================================================================
# Custom Array Schemas
# =============================================================================


@dataclass(frozen=True)
class BooleanSchema(ArraySchema):
    """Boolean-encoded-as-float array: every value must be 0, 1, or NaN."""

    def validate(self, array: NpzArray) -> list[str]:
        errors = super().validate(array)

        finite = array.data[~np.isnan(array.data)]
        invalid = finite[~np.isin(finite, (0.0, 1.0))]
        if invalid.size:
            errors.append(f"values must be 0, 1, or NaN; found {np.unique(invalid)[:5]}")

        return errors


BOOLEAN_NAN = BooleanSchema()
