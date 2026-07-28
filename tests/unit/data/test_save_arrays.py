import numpy as np

from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes, load_array, save_arrays
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
