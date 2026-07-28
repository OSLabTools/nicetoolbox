from pathlib import Path

import numpy as np

from nicetoolbox_core.data.loaded_array import _apply_filter

# Dummy path — only used in warning messages, never accessed on disk.
_PATH = Path("dummy.npz")


def _data(shape=(3, 1, 2, 4)) -> np.ndarray:
    """Return a sequential array so we can verify correct slices were kept."""
    return np.arange(np.prod(shape), dtype=float).reshape(shape)


# ---------------------------------------------------------------------------
# Wildcard
# ---------------------------------------------------------------------------


class TestWildcard:
    def test_wildcard_returns_all_labels(self):
        data = _data()
        labels = ["s1", "s2", "s3"]

        _, result_labels = _apply_filter(data, labels, "*", axis=0, npz_path=_PATH)

        assert result_labels == labels

    def test_wildcard_returns_same_data(self):
        data = _data()
        result_data, _ = _apply_filter(data, ["s1", "s2", "s3"], "*", axis=0, npz_path=_PATH)

        # result should be the identical object — no copy
        assert result_data is data


# ---------------------------------------------------------------------------
# Single string filter
# ---------------------------------------------------------------------------


class TestSingleFilter:
    def test_single_match_label_returned(self):
        data = _data()
        _, labels = _apply_filter(data, ["s1", "s2", "s3"], "s2", axis=0, npz_path=_PATH)

        assert labels == ["s2"]

    def test_single_match_reduces_axis(self):
        data = _data(shape=(3, 1, 2, 4))
        result_data, _ = _apply_filter(data, ["s1", "s2", "s3"], "s2", axis=0, npz_path=_PATH)

        np.testing.assert_array_equal(result_data, data[1:2])

    def test_single_match_correct_slice(self):
        data = _data(shape=(3, 1, 2, 4))
        result_data, _ = _apply_filter(data, ["s1", "s2", "s3"], "s2", axis=0, npz_path=_PATH)

        assert result_data.shape == (1, 1, 2, 4)
        np.testing.assert_array_equal(result_data, data[1:2])

    def test_no_match_returns_empty_labels(self):
        data = _data()
        _, labels = _apply_filter(data, ["s1", "s2"], "missing", axis=0, npz_path=_PATH)

        assert labels == []

    def test_no_match_logs_warning(self, caplog):
        import logging

        data = _data()
        with caplog.at_level(logging.WARNING):
            _apply_filter(data, ["s1", "s2"], "missing", axis=0, npz_path=_PATH)

        assert "missing" in caplog.text


# ---------------------------------------------------------------------------
# List filter
# ---------------------------------------------------------------------------


class TestListFilter:
    def test_all_match(self):
        data = _data(shape=(3, 1, 2, 4))
        result_data, labels = _apply_filter(data, ["s1", "s2", "s3"], ["s1", "s3"], axis=0, npz_path=_PATH)

        assert labels == ["s1", "s3"]
        assert result_data.shape == (2, 1, 2, 4)
        np.testing.assert_array_equal(result_data[0], data[0])  # s1 is index 0
        np.testing.assert_array_equal(result_data[1], data[2])  # s3 is index 2

    def test_partial_match_keeps_intersection(self):
        data = _data(shape=(3, 1, 2, 4))
        result_data, labels = _apply_filter(data, ["s1", "s2", "s3"], ["s1", "missing"], axis=0, npz_path=_PATH)

        assert labels == ["s1"]
        np.testing.assert_array_equal(result_data, data[0:1])  # s1 is index 0

    def test_partial_match_logs_warning(self, caplog):
        import logging

        data = _data()
        with caplog.at_level(logging.WARNING):
            _apply_filter(data, ["s1", "s2"], ["s1", "missing"], axis=0, npz_path=_PATH)

        assert "missing" in caplog.text

    def test_none_match_returns_empty(self):
        data = _data()
        _, labels = _apply_filter(data, ["s1", "s2"], ["x", "y"], axis=0, npz_path=_PATH)

        assert labels == []

    def test_order_follows_wanted_not_source(self):
        """Result order must match the requested order, not the original label order."""
        data = _data(shape=(3, 1, 2, 4))
        result_data, labels = _apply_filter(data, ["s1", "s2", "s3"], ["s3", "s1"], axis=0, npz_path=_PATH)

        assert labels == ["s3", "s1"]
        assert result_data.shape == (2, 1, 2, 4)
        np.testing.assert_array_equal(result_data[0], data[2])  # s3 is index 2
        np.testing.assert_array_equal(result_data[1], data[0])  # s1 is index 0


# ---------------------------------------------------------------------------
# Axis selection
# ---------------------------------------------------------------------------


class TestAxisSelection:
    def test_filter_axis_3_reduces_last_dim(self):
        data = _data(shape=(3, 1, 2, 4))
        result_data, labels = _apply_filter(data, ["x", "y", "z", "w"], ["x", "z"], axis=3, npz_path=_PATH)

        assert labels == ["x", "z"]
        np.testing.assert_array_equal(result_data, data[:, :, :, [0, 2]])  # x=0, z=2


# ---------------------------------------------------------------------------
# Label coercion
# ---------------------------------------------------------------------------


class TestLabelCoercion:
    def test_integer_labels_coerced_to_strings(self):
        """axis labels stored as integers must still match string filter values."""
        data = _data(shape=(3, 1, 2, 4))
        _, labels = _apply_filter(data, [0, 1, 2], "1", axis=0, npz_path=_PATH)

        assert labels == ["1"]
