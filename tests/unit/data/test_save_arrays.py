import numpy as np
import pytest

from nicetoolbox_core.data.loaded_array import (
    NpzArray,
    NpzArrayAxes,
    NpzAxisFilters,
    filter_array,
    load_array,
    load_static_array,
    save_arrays,
    select_array,
)
from nicetoolbox_core.data.npz_meta import PathMeta


def _axes(subjects, cameras, frames, labels, data=()) -> NpzArrayAxes:
    return NpzArrayAxes(
        subjects=list(subjects),
        cameras=list(cameras),
        frames=list(frames),
        labels=list(labels),
        data=list(data),
    )


def _meta(path, key) -> PathMeta:
    return PathMeta(npz_path=path, npz_key=key)


class TestSaveArrays:
    def test_creates_file(self, tmp_path):
        arr = NpzArray(np.zeros((1, 1, 2, 3)), _axes(("s1",), ("c1",), ("f0", "f1"), ("x", "y", "z")))
        out = tmp_path / "out.npz"
        save_arrays({"landmarks": arr}, out)

        assert out.exists()

    def test_writes_data_description(self, tmp_path):
        arr = NpzArray(np.zeros((2, 1, 2, 3)), _axes(("s1", "s2"), ("c1",), ("f0", "f1"), ("x", "y", "z")))
        out = tmp_path / "out.npz"
        save_arrays({"landmarks": arr}, out)

        with np.load(out, allow_pickle=True) as f:
            assert "data_description" in f.files
            descr = f["data_description"].item()["landmarks"]
            assert descr["axis0"] == ["s1", "s2"]
            assert descr["axis1"] == ["c1"]
            assert descr["axis2"] == ["f0", "f1"]
            assert descr["axis3"] == ["x", "y", "z"]
            assert "axis4" not in descr  # no data axis -> axis4 omitted

    def test_axis4_written_when_data_present(self, tmp_path):
        arr = NpzArray(
            np.zeros((1, 1, 1, 2, 2)),
            _axes(("s1",), ("c1",), ("f0",), ("x", "y"), ("u", "v")),
        )
        out = tmp_path / "out.npz"
        save_arrays({"vec": arr}, out)

        with np.load(out, allow_pickle=True) as f:
            descr = f["data_description"].item()["vec"]
            assert descr["axis4"] == ["u", "v"]

    def test_multiple_arrays_in_one_file(self, tmp_path):
        a = NpzArray(np.zeros((1, 1, 1, 2)), _axes(("s1",), ("c1",), ("f0",), ("x", "y")))
        b = NpzArray(np.ones((1, 1, 1, 3)), _axes(("s1",), ("c1",), ("f0",), ("p", "q", "r")))
        out = tmp_path / "out.npz"
        save_arrays({"a": a, "b": b}, out)

        with np.load(out, allow_pickle=True) as f:
            assert set(f.files) == {"a", "b", "data_description"}
            descr = f["data_description"].item()
            assert descr["a"]["axis3"] == ["x", "y"]
            assert descr["b"]["axis3"] == ["p", "q", "r"]

    def test_round_trip_through_load_array(self, tmp_path):
        # save_arrays output must be readable by load_array unchanged
        values = np.arange(2 * 1 * 3 * 2, dtype=float).reshape(2, 1, 3, 2)
        arr = NpzArray(values, _axes(("s1", "s2"), ("c1",), ("f0", "f1", "f2"), ("x", "y")))
        out = tmp_path / "out.npz"
        save_arrays({"landmarks": arr}, out)

        loaded = load_array(_meta(out, "landmarks"))
        assert loaded is not None
        np.testing.assert_array_equal(loaded.data, values)
        assert loaded.axes.subjects == ["s1", "s2"]
        assert loaded.axes.labels == ["x", "y"]
        assert loaded.axes.data == []


class TestExtra:
    def _array(self, extra=None) -> NpzArray:
        return NpzArray(np.zeros((1, 1, 2, 3)), _axes(("s1",), ("c1",), ("f0", "f1"), ("x", "y", "z")), extra or {})

    def test_extra_written_next_to_axes(self, tmp_path):
        out = tmp_path / "out.npz"
        save_arrays({"vertices": self._array({"csv_skip": True, "space": "camera"})}, out)

        with np.load(out, allow_pickle=True) as f:
            descr = f["data_description"].item()["vertices"]
            assert descr["csv_skip"] is True
            assert descr["space"] == "camera"
            assert descr["axis3"] == ["x", "y", "z"]

    def test_extra_round_trip(self, tmp_path):
        out = tmp_path / "out.npz"
        save_arrays({"a": self._array({"csv_skip": True}), "b": self._array()}, out)

        assert load_array(_meta(out, "a")).array.extra == {"csv_skip": True}
        assert load_array(_meta(out, "b")).array.extra == {}

    def test_extra_kept_by_select_and_filter(self):
        arr = self._array({"csv_skip": True})
        assert select_array(arr, labels=["x"]).extra == {"csv_skip": True}
        filters = NpzAxisFilters(subject="*", camera="*", label=["y"], data="*")
        assert filter_array(arr, filters).extra == {"csv_skip": True}


class TestStaticArrays:
    def _canonical(self) -> NpzArray:
        return NpzArray(np.zeros((1, 1, 2, 3)), _axes(("s1",), ("c1",), ("f0", "f1"), ("v0", "v1", "v2")))

    def test_static_array_stored_without_description(self, tmp_path):
        faces = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)
        out = tmp_path / "out.npz"
        save_arrays({"vertices": self._canonical()}, out, static_arrays={"faces": faces})

        with np.load(out, allow_pickle=True) as f:
            assert set(f.files) == {"vertices", "faces", "data_description"}
            assert set(f["data_description"].item()) == {"vertices"}
        # not described, so it is invisible to the canonical loader
        assert load_array(_meta(out, "faces")) is None

    def test_static_round_trip(self, tmp_path):
        faces = np.arange(6, dtype=np.int32).reshape(2, 3)
        out = tmp_path / "out.npz"
        save_arrays({}, out, static_arrays={"faces": faces})

        np.testing.assert_array_equal(load_static_array(out, "faces"), faces)
        assert load_static_array(out, "missing") is None

    def test_rejects_key_both_canonical_and_static(self, tmp_path):
        out = tmp_path / "out.npz"
        with pytest.raises(ValueError, match="both"):
            save_arrays({"a": self._canonical()}, out, static_arrays={"a": np.zeros(3)})
