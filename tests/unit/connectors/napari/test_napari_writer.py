"""Tests for the napari writer and its round-trip with the import parser."""

import numpy as np
import pytest
from pydantic import ValidationError

from nicetoolbox.connectors.napari.napari_configs import (
    NapariExportBodyJointsConfig,
    NapariExportSequenceConfig,
    NapariWindowConfig,
)
from nicetoolbox.connectors.napari.parser import load_napari_dataframe
from nicetoolbox.connectors.napari.writer import H5_FILENAME, SCORER, body_joints_npz_to_napari, load_body_joints_array
from nicetoolbox_core.data.loaded_array import NpzArray, NpzArrayAxes, save_arrays

SUBJECTS = ["person_left", "person_right"]
CAMERAS = ["view_top", "view_side"]
FRAMES = ["000000000", "000000001", "000000002"]
JOINTS = ["nose", "left_ear"]
DATA = ["coordinate_x", "coordinate_y", "confidence_score"]


def _write_npz(tmp_path, data=None, data_axis=DATA, key="2d", frames=FRAMES):
    shape = (len(SUBJECTS), len(CAMERAS), len(frames), len(JOINTS), len(data_axis))
    if data is None:
        data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    axes = NpzArrayAxes(subjects=SUBJECTS, cameras=CAMERAS, frames=list(frames), labels=JOINTS, data=list(data_axis))
    path = tmp_path / "body_joints.npz"
    save_arrays({key: NpzArray(data, axes)}, path)
    return tmp_path / "body_joints.npz"


def _config(**kwargs):
    return NapariExportBodyJointsConfig(
        log_level="INFO",
        log_file_path="unused.log",
        npz_key="2d",
        run={},
        **kwargs,
    )


def _write_frames(tmp_path, cameras=CAMERAS, frames=FRAMES):
    """Create the nicetoolbox_input layout: "<root>/<camera>/frames/<frame>.png"."""
    root = tmp_path / "nicetoolbox_input"
    for camera in cameras:
        frames_dir = root / camera / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            (frames_dir / f"{frame}.png").write_bytes(f"{camera}:{frame}".encode())
    return root


def _sequence(tmp_path, **kwargs):
    fields = {
        "input": tmp_path / "body_joints.npz",
        "output": tmp_path / "project",
        "cameras": "*",
        "frames_folder": tmp_path / "nicetoolbox_input",
    }
    return NapariExportSequenceConfig(**{**fields, **kwargs})


def _camera_dir(tmp_path, camera):
    """Where the writer puts one camera's .h5 and its frames."""
    return tmp_path / "project" / camera


@pytest.fixture(autouse=True)
def _frames(tmp_path):
    """Every export test needs the source frames present; copy-specific tests override this.

    Writes a generous frame range so window tests, which use longer sequences, are covered.
    """
    return _write_frames(tmp_path, frames=[f"{i:09d}" for i in range(20)])


# --- schema validation ---


def test_load_rejects_missing_key(tmp_path):
    npz = _write_npz(tmp_path)
    with pytest.raises(ValueError, match="not found in data_description"):
        load_body_joints_array(npz, "3d")


def test_load_rejects_non_body_joints_array(tmp_path):
    """A 3d pose array has the wrong coords axis and must be refused."""
    npz = _write_npz(tmp_path, data_axis=["coordinate_x", "coordinate_y", "coordinate_z", "confidence_score"])
    with pytest.raises(ValueError, match="not a valid 2d body_joints array"):
        load_body_joints_array(npz, "2d")


# --- multi-camera export ---


def test_exports_one_file_per_camera(tmp_path):
    _write_npz(tmp_path)
    written = body_joints_npz_to_napari(_sequence(tmp_path), _config())

    assert written == [_camera_dir(tmp_path, c) / H5_FILENAME for c in CAMERAS]
    assert all(p.exists() for p in written)


def test_h5_sits_in_the_camera_image_folder(tmp_path):
    """CollectedData_<scorer>.h5 sits alongside the frames it annotates."""
    _write_npz(tmp_path)
    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())

    assert written == [_camera_dir(tmp_path, "view_top") / H5_FILENAME]
    assert written[0].name == "CollectedData_NICEToolbox.h5"
    # the frames it references are siblings of the .h5
    assert (written[0].parent / f"{FRAMES[0]}.png").is_file()


def test_rejects_unknown_requested_camera(tmp_path):
    _write_npz(tmp_path)
    with pytest.raises(ValueError, match="not available"):
        body_joints_npz_to_napari(_sequence(tmp_path, cameras=["nope"]), _config())


def test_rejects_empty_camera_list(tmp_path):
    """Empty is a config mistake; exporting all cameras must be an explicit "*"."""
    _write_npz(tmp_path)
    with pytest.raises(ValueError, match='Use "\\*" to select all'):
        body_joints_npz_to_napari(_sequence(tmp_path, cameras=[]), _config())


@pytest.mark.parametrize("wildcard", ["*", ["*"]])
def test_wildcard_exports_every_camera(tmp_path, wildcard):
    _write_npz(tmp_path)
    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras=wildcard), _config())

    assert written == [_camera_dir(tmp_path, c) / H5_FILENAME for c in CAMERAS]


def test_single_camera_as_bare_string(tmp_path):
    _write_npz(tmp_path)
    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras="view_side"), _config())

    assert written == [_camera_dir(tmp_path, "view_side") / H5_FILENAME]


def test_each_camera_gets_its_own_slice(tmp_path):
    """Camera files must differ — guards against the axis-order bug."""
    npz = _write_npz(tmp_path)
    body_joints_npz_to_napari(_sequence(tmp_path), _config())

    top = load_napari_dataframe(_camera_dir(tmp_path, "view_top") / H5_FILENAME)
    side = load_napari_dataframe(_camera_dir(tmp_path, "view_side") / H5_FILENAME)

    expected = np.load(npz, allow_pickle=True)["2d"]
    np.testing.assert_allclose(
        top.to_numpy().reshape(len(FRAMES), len(SUBJECTS), len(JOINTS), 3), expected[:, 0].transpose(1, 0, 2, 3)
    )
    assert not np.allclose(top.to_numpy(), side.to_numpy())


# --- NaN and naming ---


def test_nan_joints_are_preserved(tmp_path):
    """All-NaN bodyparts must survive; pivot_table would have dropped them."""
    shape = (len(SUBJECTS), len(CAMERAS), len(FRAMES), len(JOINTS), len(DATA))
    data = np.full(shape, np.nan)
    _write_npz(tmp_path, data=data)

    body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())
    df = load_napari_dataframe(_camera_dir(tmp_path, "view_top") / H5_FILENAME)

    assert list(df.columns.get_level_values("bodyparts").unique()) == JOINTS
    assert df.isna().all().all()


def test_scorer_is_fixed_and_survives_round_trip(tmp_path):
    """The scorer level is hardcoded and must round-trip through HDF5 unchanged."""
    _write_npz(tmp_path)
    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras="view_top"), _config())

    df = load_napari_dataframe(written[0])
    assert list(df.columns.get_level_values("scorer").unique()) == [SCORER]
    assert SCORER == "NICEToolbox"


def test_exports_toolbox_names_unchanged(tmp_path):
    """Names are written exactly as stored in the NPZ — the export never renames."""
    _write_npz(tmp_path)
    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())

    df = load_napari_dataframe(written[0])
    assert list(df.columns.get_level_values("individuals").unique()) == SUBJECTS
    assert list(df.columns.get_level_values("bodyparts").unique()) == JOINTS
    assert list(dict.fromkeys(df.index.get_level_values(1))) == ["view_top"]


# --- frame image copying ---


def test_copies_frames_next_to_the_h5(tmp_path):
    _write_npz(tmp_path)
    body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())

    dest = _camera_dir(tmp_path, "view_top")
    assert sorted(p.name for p in dest.glob("*.png")) == [f"{f}.png" for f in FRAMES]
    # content copied verbatim from the matching source camera
    assert (dest / f"{FRAMES[0]}.png").read_bytes() == f"view_top:{FRAMES[0]}".encode()


def test_copied_paths_match_the_annotation_index(tmp_path):
    """Row-index paths are relative to the project root's parent and must resolve to real files."""
    _write_npz(tmp_path)
    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())

    df = load_napari_dataframe(written[0])
    assert list(dict.fromkeys(df.index.get_level_values(0))) == ["project"]
    for root, camera, name in df.index:
        assert (tmp_path / root / camera / name).is_file()


def test_copies_frames_per_camera(tmp_path):
    _write_npz(tmp_path)
    body_joints_npz_to_napari(_sequence(tmp_path), _config())

    for camera in CAMERAS:
        dest = _camera_dir(tmp_path, camera)
        assert sorted(p.name for p in dest.glob("*.png")) == [f"{f}.png" for f in FRAMES]
        assert (dest / f"{FRAMES[0]}.png").read_bytes() == f"{camera}:{FRAMES[0]}".encode()


def test_raises_when_frames_folder_missing(tmp_path):
    _write_npz(tmp_path)
    seq = _sequence(tmp_path, cameras=["view_top"], frames_folder=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="Frames folder for camera"):
        body_joints_npz_to_napari(seq, _config())


def test_raises_when_a_referenced_frame_is_missing(tmp_path):
    """A frame in the NPZ with no image on disk must fail, not silently skip."""
    _write_npz(tmp_path)
    (tmp_path / "nicetoolbox_input" / "view_top" / "frames" / f"{FRAMES[1]}.png").unlink()

    with pytest.raises(FileNotFoundError, match="missing from"):
        body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())


# --- re-export clears the previous run ---


def test_stale_frames_are_removed_on_re_export(tmp_path):
    """A narrower second export must not leave the first run's extra frames behind."""
    _write_npz(tmp_path)
    body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())

    dest = _camera_dir(tmp_path, "view_top")
    stale = dest / "000000999.png"
    stale.write_bytes(b"stale")
    assert stale.is_file()

    body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())

    assert not stale.exists()
    assert sorted(p.name for p in dest.glob("*.png")) == [f"{f}.png" for f in FRAMES]
    assert (dest / H5_FILENAME).is_file()


def test_re_export_is_idempotent(tmp_path):
    _write_npz(tmp_path)
    body_joints_npz_to_napari(_sequence(tmp_path), _config())
    first = {p.name for p in _camera_dir(tmp_path, "view_top").iterdir()}

    body_joints_npz_to_napari(_sequence(tmp_path), _config())
    second = {p.name for p in _camera_dir(tmp_path, "view_top").iterdir()}

    assert first == second


def test_existing_camera_folder_is_replaced(tmp_path):
    """Whatever was in the camera folder before is gone after a re-export."""
    _write_npz(tmp_path)
    dest = _camera_dir(tmp_path, "view_top")
    dest.mkdir(parents=True)
    (dest / "leftover.txt").write_text("from an earlier run")

    body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())

    assert not (dest / "leftover.txt").exists()
    assert sorted(p.name for p in dest.glob("*.png")) == [f"{f}.png" for f in FRAMES]
    assert (dest / H5_FILENAME).is_file()


def test_only_the_exported_cameras_are_cleared(tmp_path):
    """Cameras not in this run keep their folders untouched."""
    _write_npz(tmp_path)
    body_joints_npz_to_napari(_sequence(tmp_path), _config())

    body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())

    # view_side was exported by the first run and not re-exported by the second
    assert (_camera_dir(tmp_path, "view_side") / H5_FILENAME).is_file()


# --- sliding-window sampling ---


def _window(size, stride):
    return _config(window=NapariWindowConfig(size=size, stride=stride))


def test_window_keeps_chunks_at_each_stride(tmp_path):
    frames = [f"{i:09d}" for i in range(10)]
    _write_npz(tmp_path, frames=frames)

    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _window(size=2, stride=5))

    df = load_napari_dataframe(written[0])
    assert [name.split(".")[0] for _, _, name in df.index] == ["000000000", "000000001", "000000005", "000000006"]


def test_window_copies_only_sampled_frames(tmp_path):
    frames = [f"{i:09d}" for i in range(10)]
    _write_npz(tmp_path, frames=frames)
    _write_frames(tmp_path, frames=frames)

    body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _window(size=2, stride=5))

    dest = _camera_dir(tmp_path, "view_top")
    assert sorted(p.stem for p in dest.glob("*.png")) == ["000000000", "000000001", "000000005", "000000006"]


def test_trailing_window_is_truncated_not_dropped(tmp_path):
    """The final partial window still exports, so the last frames stay reachable."""
    frames = [f"{i:09d}" for i in range(7)]
    _write_npz(tmp_path, frames=frames)

    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _window(size=3, stride=5))

    df = load_napari_dataframe(written[0])
    kept = [name.split(".")[0] for _, _, name in df.index]
    assert kept == ["000000000", "000000001", "000000002", "000000005", "000000006"]


def test_window_values_follow_their_frames(tmp_path):
    """Sampling must carry each frame's own data, not shift rows."""
    frames = [f"{i:09d}" for i in range(10)]
    npz = _write_npz(tmp_path, frames=frames)
    body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _window(size=2, stride=5))

    full = np.load(npz, allow_pickle=True)["2d"]
    df = load_napari_dataframe(_camera_dir(tmp_path, "view_top") / H5_FILENAME)
    got = df.to_numpy().reshape(4, len(SUBJECTS), len(JOINTS), 3)

    expected = full[:, 0][:, [0, 1, 5, 6]].transpose(1, 0, 2, 3)
    np.testing.assert_allclose(got, expected)


def test_stride_equal_to_size_is_contiguous(tmp_path):
    frames = [f"{i:09d}" for i in range(6)]
    _write_npz(tmp_path, frames=frames)

    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _window(size=3, stride=3))

    df = load_napari_dataframe(written[0])
    assert len(df) == 6  # every frame kept


def test_one_window_applies_to_every_sequence(tmp_path):
    """The window lives on the run config, so all algorithms sample the same frames."""
    frames = [f"{i:09d}" for i in range(10)]
    _write_npz(tmp_path, frames=frames)
    cfg = _window(size=2, stride=5)

    first = body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), cfg)
    second = body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"], output=tmp_path / "other"), cfg)

    kept = lambda p: [n.split(".")[0] for _, _, n in load_napari_dataframe(p).index]  # noqa: E731
    assert kept(first[0]) == kept(second[0]) == ["000000000", "000000001", "000000005", "000000006"]


def test_overlapping_window_is_rejected():
    with pytest.raises(ValidationError, match="may not overlap"):
        NapariWindowConfig(size=10, stride=5)


def test_no_window_exports_every_frame(tmp_path):
    _write_npz(tmp_path)
    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())

    assert len(load_napari_dataframe(written[0])) == len(FRAMES)


# --- round-trip with the import parser ---


def test_round_trip_through_import_parser(tmp_path):
    """The exported file must be readable by the import side's loader."""
    _write_npz(tmp_path)
    written = body_joints_npz_to_napari(_sequence(tmp_path, cameras=["view_top"]), _config())

    df = load_napari_dataframe(written[0])

    assert df.columns.names == ["scorer", "individuals", "bodyparts", "coords"]
    assert list(df.columns.get_level_values("coords").unique()) == ["x", "y", "likelihood"]
    assert list(df.columns.get_level_values("bodyparts").unique()) == JOINTS
    # 3-level row index, as parser.load_napari_dataframe expects
    assert df.index.nlevels == 3
    assert list(dict.fromkeys(df.index.get_level_values(1))) == ["view_top"]
    assert [name.split(".")[0] for _, _, name in df.index] == FRAMES
