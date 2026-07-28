"""load_input schema validation: when a schema is passed, each loaded array is
validated and a mismatch raises."""

import numpy as np
import pytest

from nicetoolbox.evaluation.data import input_loader
from nicetoolbox_core.data.array_schema import BOOLEAN_NAN
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes, NpzArrayWithMeta
from nicetoolbox_core.data.npz_meta import PathMeta


def _loaded(values: np.ndarray) -> NpzArrayWithMeta:
    axes = NpzArrayAxes(["s1"], ["c1"], ["f0"], list("ab"[: values.shape[-1]]))
    meta = PathMeta(npz_path="dummy.npz", npz_key="k")
    return NpzArrayWithMeta(array=NpzArray(values, axes), npz_meta=meta)


class _Block:
    """Minimal stand-in for a BaseInputBlock (only what load_input touches)."""

    def axis_filters(self):
        return None


@pytest.fixture
def patched(monkeypatch):
    """Stub file resolution so load_input runs on an in-memory array."""
    holder = {"array": None}

    monkeypatch.setattr(input_loader, "get_npz_files", lambda *_: [holder["array"].meta])
    monkeypatch.setattr(input_loader, "load_and_filter_array", lambda *_: holder["array"])
    return holder


def test_valid_array_passes_schema(patched):
    patched["array"] = _loaded(np.array([[[[0.0, 1.0]]]]))  # 0/1 -> valid BOOLEAN
    result = input_loader.load_input(_Block(), schema=BOOLEAN_NAN)
    assert result == [patched["array"]]


def test_invalid_array_raises(patched):
    patched["array"] = _loaded(np.array([[[[0.0, 0.5]]]]))  # 0.5 -> not boolean
    with pytest.raises(ValueError, match="failed schema validation"):
        input_loader.load_input(_Block(), schema=BOOLEAN_NAN)


def test_no_schema_skips_validation(patched):
    patched["array"] = _loaded(np.array([[[[0.0, 0.5]]]]))  # would fail BOOLEAN, but no schema passed
    result = input_loader.load_input(_Block())
    assert result == [patched["array"]]
