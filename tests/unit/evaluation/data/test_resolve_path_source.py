from pathlib import Path

import pytest

from nicetoolbox.configs.schemas.evaluation_input_block import PathInput
from nicetoolbox.evaluation.data.input_loader import _resolve_path_source
from nicetoolbox_core.data.npz_meta import PathMeta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input_block(path: Path | list[Path], npz_key: str = "landmarks") -> PathInput:
    """Create a path-source InputBlock pointing at the given path(s)."""
    return PathInput(path=path, npz_key=npz_key)


def _touch_npz(path: Path) -> Path:
    """Create an empty .npz file at the given path, ensuring parents exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestSingleFile:
    def test_single_npz(self, tmp_path):
        npz = _touch_npz(tmp_path / "result.npz")
        block = _make_input_block(npz)

        result = _resolve_path_source(block)

        assert len(result) == 1
        assert result[0].npz_path == npz
        assert result[0].npz_key == "landmarks"

    def test_returns_path_meta(self, tmp_path):
        npz = _touch_npz(tmp_path / "result.npz")
        block = _make_input_block(npz)

        result = _resolve_path_source(block)

        assert isinstance(result[0], PathMeta)

    def test_key_propagated(self, tmp_path):
        npz = _touch_npz(tmp_path / "result.npz")
        block = _make_input_block(npz, npz_key="distances")

        result = _resolve_path_source(block)

        assert result[0].npz_key == "distances"


# ---------------------------------------------------------------------------
# Glob expansion
# ---------------------------------------------------------------------------


class TestGlobExpansion:
    def test_wildcard_matches_multiple(self, tmp_path):
        _touch_npz(tmp_path / "a.npz")
        _touch_npz(tmp_path / "b.npz")
        _touch_npz(tmp_path / "c.npz")
        block = _make_input_block(tmp_path / "*.npz")

        result = _resolve_path_source(block)

        assert len(result) == 3

    def test_wildcard_skips_non_npz(self, tmp_path):
        _touch_npz(tmp_path / "a.npz")
        _touch_npz(tmp_path / "b.txt")
        block = _make_input_block(tmp_path / "*.npz")

        result = _resolve_path_source(block)

        assert len(result) == 1
        assert result[0].npz_path == tmp_path / "a.npz"

    def test_recursive_glob(self, tmp_path):
        _touch_npz(tmp_path / "sub1" / "a.npz")
        _touch_npz(tmp_path / "sub2" / "b.npz")
        block = _make_input_block(tmp_path / "**" / "*.npz")

        result = _resolve_path_source(block)

        assert len(result) == 2

    def test_no_matches_raises(self, tmp_path):
        block = _make_input_block(tmp_path / "*.npz")

        with pytest.raises(FileNotFoundError, match="No files matched"):
            _resolve_path_source(block)


# ---------------------------------------------------------------------------
# Multiple paths
# ---------------------------------------------------------------------------


class TestMultiplePaths:
    def test_two_explicit_paths(self, tmp_path):
        npz_a = _touch_npz(tmp_path / "a.npz")
        npz_b = _touch_npz(tmp_path / "b.npz")
        block = _make_input_block([npz_a, npz_b])

        result = _resolve_path_source(block)

        assert len(result) == 2
        returned_paths = {m.npz_path for m in result}
        assert returned_paths == {npz_a, npz_b}

    def test_one_pattern_no_match_raises(self, tmp_path):
        npz = _touch_npz(tmp_path / "a.npz")
        block = _make_input_block([npz, tmp_path / "missing" / "*.npz"])

        with pytest.raises(FileNotFoundError, match="No files matched"):
            _resolve_path_source(block)

    def test_mixed_explicit_and_glob(self, tmp_path):
        npz_explicit = _touch_npz(tmp_path / "explicit.npz")
        _touch_npz(tmp_path / "glob_dir" / "x.npz")
        _touch_npz(tmp_path / "glob_dir" / "y.npz")
        block = _make_input_block([npz_explicit, tmp_path / "glob_dir" / "*.npz"])

        result = _resolve_path_source(block)

        assert len(result) == 3


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_overlapping_globs_deduplicated(self, tmp_path):
        """Overlapping patterns matching the same file produce a single entry."""
        npz = _touch_npz(tmp_path / "result.npz")
        block = _make_input_block([tmp_path / "*.npz", npz])

        result = _resolve_path_source(block)

        assert len(result) == 1
        assert result[0].npz_path == npz
