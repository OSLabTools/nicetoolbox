import numpy as np
import pytest

from nicetoolbox_core.data.loaded_array import NpzArrayWithMeta, NpzAxisFilters, load_and_filter_array, load_array
from nicetoolbox_core.data.npz_meta import PathMeta
from tests.unit.data.conftest import make_npz


def _meta(path, key="landmarks") -> PathMeta:
    return PathMeta(npz_path=path, npz_key=key)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_returns_loaded_array(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz")
        result = load_array(_meta(npz))

        assert isinstance(result, NpzArrayWithMeta)

    def test_meta_passed_through(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz")
        meta = _meta(npz)
        result = load_array(meta)

        assert result.meta is meta

    def test_axes_populated(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz")
        result = load_array(_meta(npz))

        assert result.axes.subjects == ["s1", "s2"]
        assert result.axes.cameras == ["cam1"]
        assert result.axes.frames == ["f0", "f1", "f2"]
        assert result.axes.labels == ["x", "y", "z"]
        assert result.axes.data == []

    def test_data_shape(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz")
        result = load_array(_meta(npz))

        assert result.data.shape == (2, 1, 3, 3)

    def test_data_values(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz")
        result = load_array(_meta(npz))

        expected = np.arange(2 * 1 * 3 * 3, dtype=float).reshape(2, 1, 3, 3)
        np.testing.assert_array_equal(result.data, expected)

    def test_with_data_axis(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz", data=("u", "v"))
        result = load_array(_meta(npz))

        assert result.axes.data == ["u", "v"]
        assert result.data.shape == (2, 1, 3, 3, 2)

    def test_without_data_axis(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz")
        result = load_array(_meta(npz))

        assert result.axes.data == []
        assert result.data.ndim == 4


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    """load_and_filter_array = load_array + filter_array in one step."""

    def test_subject_filter(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz")
        filters = NpzAxisFilters(subject="s1", camera="*", label="*", data="*")
        result = load_and_filter_array(_meta(npz), filters)

        assert result.axes.subjects == ["s1"]
        np.testing.assert_array_equal(result.data, result.data[0:1])

    def test_label_filter(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz")
        filters = NpzAxisFilters(subject="*", camera="*", label=["x", "z"], data="*")
        result = load_and_filter_array(_meta(npz), filters)

        assert result.axes.labels == ["x", "z"]
        assert result.data.shape == (2, 1, 3, 2)

    def test_filter_empties_axis_returns_none(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz")
        filters = NpzAxisFilters(subject="nonexistent", camera="*", label="*", data="*")
        result = load_and_filter_array(_meta(npz), filters)

        assert result is None

    def test_data_filter(self, tmp_path):
        npz = make_npz(tmp_path / "result.npz", data=("u", "v", "w"))
        filters = NpzAxisFilters(subject="*", camera="*", label="*", data="v")
        result = load_and_filter_array(_meta(npz), filters)

        assert result.axes.data == ["v"]
        assert result.data.shape == (2, 1, 3, 3, 1)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_data_description_raises(self, tmp_path):
        npz = tmp_path / "bad.npz"
        np.savez(npz, landmarks=np.zeros((2, 1, 3, 3)))

        with pytest.raises(KeyError, match="data_description"):
            load_array(_meta(npz))

    def test_npz_key_missing_from_data_description_returns_none(self, tmp_path, caplog):
        import logging

        npz = make_npz(tmp_path / "result.npz", key="landmarks")

        with caplog.at_level(logging.WARNING):
            result = load_array(_meta(npz, key="wrong_key"))

        assert result is None
        assert "wrong_key" in caplog.text

    def test_npz_key_in_description_but_missing_from_data_raises(self, tmp_path):
        """data_description references a key that has no corresponding array."""
        npz = tmp_path / "bad.npz"
        descr = {"landmarks": {"axis0": ["s1"], "axis1": ["c1"], "axis2": ["f0"], "axis3": ["x"]}}
        # Save description but omit the actual data array
        np.savez(npz, data_description=np.array(descr, dtype=object))

        with pytest.raises(KeyError, match="landmarks"):
            load_array(_meta(npz))

    @pytest.mark.parametrize(
        "axis,descr_override,shape",
        [
            ("axis0", {"axis0": ["s1", "s2", "s3"]}, (2, 1, 1, 1)),  # 3 labels, 2 in data
            ("axis1", {"axis1": ["c1", "c2"]}, (1, 1, 1, 1)),  # 2 labels, 1 in data
            ("axis2", {"axis2": ["f0", "f1", "f2"]}, (1, 1, 1, 1)),  # 3 labels, 1 in data
            ("axis3", {"axis3": ["x", "y", "z"]}, (1, 1, 1, 1)),  # 3 labels, 1 in data
        ],
    )
    def test_shape_mismatch_raises(self, tmp_path, axis, descr_override, shape):
        """data.shape[dim] disagrees with the corresponding axis label count."""
        npz = tmp_path / "bad.npz"
        base = {"axis0": ["s1"], "axis1": ["c1"], "axis2": ["f0"], "axis3": ["x"]}
        descr = {"landmarks": {**base, **descr_override}}
        np.savez(npz, data_description=np.array(descr, dtype=object), landmarks=np.zeros(shape))

        with pytest.raises(ValueError, match=axis):
            load_array(_meta(npz))

    def test_too_few_dimensions_raises(self, tmp_path):
        """Data array with fewer than 4 dims is rejected before any filtering."""
        npz = tmp_path / "bad.npz"
        descr = {"landmarks": {"axis0": ["s1"], "axis1": ["c1"], "axis2": ["f0"], "axis3": ["x"]}}
        np.savez(npz, data_description=np.array(descr, dtype=object), landmarks=np.zeros((1, 1, 1)))

        with pytest.raises(ValueError, match="dimensions"):
            load_array(_meta(npz))

    def test_axis4_in_description_but_data_is_4d_raises(self, tmp_path):
        """axis4 declared in description but data only has 4 dimensions."""
        npz = tmp_path / "bad.npz"
        descr = {"landmarks": {"axis0": ["s1"], "axis1": ["c1"], "axis2": ["f0"], "axis3": ["x"], "axis4": ["u", "v"]}}
        np.savez(npz, data_description=np.array(descr, dtype=object), landmarks=np.zeros((1, 1, 1, 1)))

        with pytest.raises(ValueError, match="axis4"):
            load_array(_meta(npz))

    def test_axis4_shape_mismatch_raises(self, tmp_path):
        """axis4 label count disagrees with data.shape[4]."""
        npz = tmp_path / "bad.npz"
        descr = {
            "landmarks": {"axis0": ["s1"], "axis1": ["c1"], "axis2": ["f0"], "axis3": ["x"], "axis4": ["u", "v", "w"]}
        }
        # 2 elements on dim 4 but 3 labels in description
        np.savez(npz, data_description=np.array(descr, dtype=object), landmarks=np.zeros((1, 1, 1, 1, 2)))

        with pytest.raises(ValueError, match="axis4"):
            load_array(_meta(npz))

    def test_required_axis_key_missing_from_description_raises(self, tmp_path):
        """axis2 omitted from data_description — required axes must all be present."""
        npz = tmp_path / "bad.npz"
        descr = {"landmarks": {"axis0": ["s1"], "axis1": ["c1"], "axis3": ["x"]}}  # axis2 missing
        np.savez(npz, data_description=np.array(descr, dtype=object), landmarks=np.zeros((1, 1, 1, 1)))

        with pytest.raises(KeyError, match="axis2"):
            load_array(_meta(npz))
