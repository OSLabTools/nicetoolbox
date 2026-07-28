import numpy as np
import pytest

from nicetoolbox_core.data.array_schema import (
    BOOLEAN_NAN,
    FLOAT,
    VECTOR_2D_CONF_PER_LABEL,
    VECTOR_2D_PER_LABEL,
    VECTOR_3D_CONF_PER_LABEL,
    VECTOR_3D_PER_LABEL,
    AnyOf,
    ArraySchema,
    BaseArraySchema,
)
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _array(
    data: np.ndarray,
    *,
    subjects=("s1",),
    cameras=("c1",),
    frames=("f0",),
    labels=("j1",),
    data_labels=(),
) -> NpzArray:
    axes = NpzArrayAxes(
        subjects=list(subjects),
        cameras=list(cameras),
        frames=list(frames),
        labels=list(labels),
        data=list(data_labels),
    )
    return NpzArray(data=data, axes=axes)


# ---------------------------------------------------------------------------
# Base ArraySchema
# ---------------------------------------------------------------------------


class TestArraySchemaValueLabels:
    def test_matching_value_labels_pass(self):
        schema = ArraySchema(labels_columns=("nose", "neck"))
        arr = _array(np.zeros((1, 1, 1, 2)), labels=("nose", "neck"))
        assert schema.validate(arr) == []

    def test_wrong_value_labels_fail(self):
        schema = ArraySchema(labels_columns=("nose", "neck"))
        arr = _array(np.zeros((1, 1, 1, 2)), labels=("nose", "wrist"))
        errors = schema.validate(arr)
        assert any("labels axis" in e for e in errors)

    def test_order_matters(self):
        schema = ArraySchema(labels_columns=("nose", "neck"))
        arr = _array(np.zeros((1, 1, 1, 2)), labels=("neck", "nose"))
        assert any("labels axis" in e for e in schema.validate(arr))

    def test_empty_value_labels_skips_check(self):
        # value_labels not set -> label axis is not constrained
        schema = ArraySchema()
        arr = _array(np.zeros((1, 1, 1, 3)), labels=("a", "b", "c"))
        assert schema.validate(arr) == []


class TestArraySchemaDataLabels:
    def test_matching_data_labels_pass(self):
        arr = _array(
            np.zeros((1, 1, 1, 1, 2)),
            data_labels=("coordinate_x", "coordinate_y"),
        )
        assert VECTOR_2D_PER_LABEL.validate(arr) == []

    def test_wrong_data_labels_fail(self):
        arr = _array(np.zeros((1, 1, 1, 1, 2)), data_labels=("x", "y"))
        errors = VECTOR_2D_PER_LABEL.validate(arr)
        assert any("data axis" in e for e in errors)

    def test_missing_data_axis_fails_when_required(self):
        # schema expects a data axis but the array has none
        arr = _array(np.zeros((1, 1, 1, 1)), data_labels=())
        assert any("data axis" in e for e in VECTOR_2D_PER_LABEL.validate(arr))


class TestArraySchemaDtype:
    def test_float_passes(self):
        arr = _array(np.zeros((1, 1, 1, 1), dtype=np.float64))
        assert ArraySchema().validate(arr) == []

    def test_int_fails(self):
        arr = _array(np.zeros((1, 1, 1, 1), dtype=np.int32))
        errors = ArraySchema().validate(arr)
        assert any("dtype" in e for e in errors)

    def test_custom_dtype(self):
        schema = ArraySchema(dtype=np.integer)
        assert schema.validate(_array(np.zeros((1, 1, 1, 1), dtype=np.int64))) == []
        assert any("dtype" in e for e in schema.validate(_array(np.zeros((1, 1, 1, 1), dtype=np.float32))))

    def test_multiple_errors_accumulate(self):
        # wrong labels AND wrong dtype -> two errors
        schema = ArraySchema(labels_columns=("nose",), dtype=np.floating)
        arr = _array(np.zeros((1, 1, 1, 1), dtype=np.int32), labels=("wrist",))
        errors = schema.validate(arr)
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# Predefined vector schemas
# ---------------------------------------------------------------------------


class TestVectorSchemas:
    @pytest.mark.parametrize(
        "schema,labels",
        [
            (VECTOR_2D_PER_LABEL, ("coordinate_x", "coordinate_y")),
            (VECTOR_2D_CONF_PER_LABEL, ("coordinate_x", "coordinate_y", "confidence_score")),
            (VECTOR_3D_PER_LABEL, ("coordinate_x", "coordinate_y", "coordinate_z")),
            (VECTOR_3D_CONF_PER_LABEL, ("coordinate_x", "coordinate_y", "coordinate_z", "confidence_score")),
        ],
    )
    def test_valid(self, schema, labels):
        arr = _array(np.zeros((1, 1, 1, 1, len(labels))), data_labels=labels)
        assert schema.validate(arr) == []

    def test_vector_3d_rejects_confidence_labels(self):
        # VECTOR_3D has no confidence; a 4-label array must fail
        arr = _array(
            np.zeros((1, 1, 1, 1, 4)),
            data_labels=("coordinate_x", "coordinate_y", "coordinate_z", "confidence_score"),
        )
        assert any("data axis" in e for e in VECTOR_3D_PER_LABEL.validate(arr))


# ---------------------------------------------------------------------------
# AnyOf
# ---------------------------------------------------------------------------


class TestAnyOf:
    def test_accepts_first_option(self):
        schema = AnyOf(VECTOR_3D_PER_LABEL, VECTOR_3D_CONF_PER_LABEL)
        arr = _array(
            np.zeros((1, 1, 1, 1, 3)),
            data_labels=("coordinate_x", "coordinate_y", "coordinate_z"),
        )
        assert schema.validate(arr) == []

    def test_accepts_second_option(self):
        schema = AnyOf(VECTOR_3D_PER_LABEL, VECTOR_3D_CONF_PER_LABEL)
        arr = _array(
            np.zeros((1, 1, 1, 1, 4)),
            data_labels=("coordinate_x", "coordinate_y", "coordinate_z", "confidence_score"),
        )
        assert schema.validate(arr) == []

    def test_rejects_when_no_option_matches(self):
        schema = AnyOf(VECTOR_3D_PER_LABEL, VECTOR_3D_CONF_PER_LABEL)
        arr = _array(np.zeros((1, 1, 1, 1, 2)), data_labels=("coordinate_x", "coordinate_y"))
        errors = schema.validate(arr)
        assert any("matched none" in e for e in errors)

    def test_reports_underlying_errors(self):
        # a non-matching array should surface why each option failed
        schema = AnyOf(VECTOR_3D_PER_LABEL, VECTOR_3D_CONF_PER_LABEL)
        arr = _array(np.zeros((1, 1, 1, 1, 2)), data_labels=("coordinate_x", "coordinate_y"))
        (msg,) = schema.validate(arr)
        assert "data axis" in msg  # each VECTOR_3D* option complains about the data axis

    def test_first_match_wins_no_errors(self):
        schema = AnyOf(VECTOR_2D_PER_LABEL, VECTOR_3D_PER_LABEL)
        arr = _array(np.zeros((1, 1, 1, 1, 2)), data_labels=("coordinate_x", "coordinate_y"))
        assert schema.validate(arr) == []

    def test_is_base_array_schema(self):
        schema = AnyOf(VECTOR_3D_PER_LABEL, VECTOR_3D_CONF_PER_LABEL)
        assert isinstance(schema, BaseArraySchema)
        assert isinstance(schema, AnyOf)
        assert not isinstance(schema, ArraySchema)


# ---------------------------------------------------------------------------
# FLOAT
# ---------------------------------------------------------------------------


class TestFloat:
    def test_float_valid(self):
        arr = _array(np.zeros((1, 1, 1, 2)), labels=("a", "b"))
        assert FLOAT.validate(arr) == []

    def test_int_dtype_fails(self):
        arr = _array(np.zeros((1, 1, 1, 2), dtype=np.int32), labels=("a", "b"))
        assert any("dtype" in e for e in FLOAT.validate(arr))

    def test_any_labels_allowed(self):
        arr = _array(np.zeros((1, 1, 1, 3)), labels=("foo", "bar", "baz"))
        assert FLOAT.validate(arr) == []

    def test_data_axis_allowed(self):
        # FLOAT does not constrain shape — a vector array is fine
        arr = _array(np.zeros((1, 1, 1, 1, 3)), data_labels=("x", "y", "z"))
        assert FLOAT.validate(arr) == []


# ---------------------------------------------------------------------------
# BooleanSchema
# ---------------------------------------------------------------------------


class TestBooleanSchema:
    def test_zeros_ones_nan_valid(self):
        data = np.array([[[[0.0, 1.0, np.nan]]]])  # (1, 1, 1, 3)
        assert BOOLEAN_NAN.validate(_array(data, labels=("a", "b", "c"))) == []

    def test_non_binary_value_fails(self):
        data = np.array([[[[0.0, 0.5]]]])  # 0.5 invalid
        errors = BOOLEAN_NAN.validate(_array(data, labels=("a", "b")))
        assert any("must be 0, 1, or NaN" in e for e in errors)

    def test_all_nan_valid(self):
        data = np.full((1, 1, 1, 2), np.nan)
        assert BOOLEAN_NAN.validate(_array(data, labels=("a", "b"))) == []

    def test_int_dtype_fails_base_check(self):
        # inherits ArraySchema dtype=np.floating
        data = np.zeros((1, 1, 1, 2), dtype=np.int32)
        assert any("dtype" in e for e in BOOLEAN_NAN.validate(_array(data, labels=("a", "b"))))
