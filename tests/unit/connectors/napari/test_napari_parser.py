"""Tests for the napari import parser, focused on multi-camera merging."""

import numpy as np
import pandas as pd
import pytest

from nicetoolbox.connectors.napari.napari_configs import NapariImportBodyJointsConfig, NapariSequenceConfig
from nicetoolbox.connectors.napari.parser import napari_to_body_joints_npz

INDIVIDUALS = ["pL", "pR"]
JOINTS = ["nose", "left_ear"]
COORDS = ["x", "y", "likelihood"]
FPS = 25


def _write_camera(
    root, camera, frames, individuals=INDIVIDUALS, joints=JOINTS, fill=0.0, name="CollectedData_X.h5", coords=COORDS
):
    """Write one camera's annotation file into "<root>/<camera>/<name>"."""
    return _write_h5(root / camera / name, frames, individuals=individuals, joints=joints, fill=fill, coords=coords)


def _write_h5(path, frames, individuals=INDIVIDUALS, joints=JOINTS, fill=0.0, coords=COORDS):
    """Write a napari HDF5 file with a 3-level row index and 4-level column index."""
    columns = pd.MultiIndex.from_tuples(
        [("X", ind, joint, coord) for ind in individuals for joint in joints for coord in coords],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    index = pd.MultiIndex.from_tuples([("labeled-data", "cam", f"{f:09d}.png") for f in frames])
    data = np.full((len(frames), len(columns)), fill, dtype=np.float64)
    # make each row identifiable: value encodes the frame number
    for i, f in enumerate(frames):
        data[i, :] = fill + f
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, index=index, columns=columns).to_hdf(path, key="df_with_missing", mode="w", format="fixed")
    return path


def _sequence(tmp_path, input_root, **kwargs):
    fields = {
        "input": input_root,
        "output": tmp_path / "out.npz",
        "fps": FPS,
        "start": 0,
        "end": -1,
        "reset_frames": False,
    }
    return NapariSequenceConfig(**{**fields, **kwargs})


def _config(**kwargs):
    return NapariImportBodyJointsConfig(
        log_level="INFO",
        log_file_path="unused.log",
        export_csv=False,
        run={},
        **kwargs,
    )


def _load(path):
    raw = np.load(path, allow_pickle=True)
    return raw["2d"], raw["data_description"].item()["2d"]


# --- multi-camera merging ---


def test_merges_all_subfolders_as_cameras(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1, 2], fill=100.0)
    _write_camera(root, "cam4", [0, 1, 2], fill=200.0)

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    arr, desc = _load(tmp_path / "out.npz")
    assert desc["axis1"] == ["cam3", "cam4"]
    assert arr.shape == (len(INDIVIDUALS), 2, 3, len(JOINTS), 3)
    # each camera keeps its own values
    assert arr[0, 0, 0, 0, 0] == 100.0
    assert arr[0, 1, 0, 0, 0] == 200.0


def test_cameras_are_discovered_in_sorted_order(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam9", [0, 1], fill=10.0)
    _write_camera(root, "cam1", [0, 1], fill=20.0)

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    _, desc = _load(tmp_path / "out.npz")
    assert desc["axis1"] == ["cam1", "cam9"]


def test_annotation_filename_is_not_read(tmp_path):
    """A napari project names the file after its own scorer; discovery must not assume ours."""
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1], name="CollectedData_SomeoneElse.h5")

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    _, desc = _load(tmp_path / "out.npz")
    assert desc["axis1"] == ["cam3"]


def test_single_camera_still_works(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1, 2], fill=5.0)

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    arr, desc = _load(tmp_path / "out.npz")
    assert desc["axis1"] == ["cam3"]
    assert arr.shape[1] == 1


def test_camera_names_go_through_the_mapping(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1], fill=1.0)

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config(cameras={"cam3": "view_top"}))

    _, desc = _load(tmp_path / "out.npz")
    assert desc["axis1"] == ["view_top"]


# --- differing frame coverage ---


def test_frame_range_spans_all_cameras(tmp_path):
    """A camera annotated over a shorter range is NaN-padded, not truncated."""
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1, 2, 3], fill=100.0)
    _write_camera(root, "cam4", [0, 1], fill=200.0)

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    arr, desc = _load(tmp_path / "out.npz")
    assert desc["axis2"] == ["000000000", "000000001", "000000002", "000000003"]
    # cam4 stops at frame 1, so its later frames are NaN while cam3 still has data
    assert np.isnan(arr[0, 1, 2, 0, 0])
    assert not np.isnan(arr[0, 0, 2, 0, 0])
    assert not np.isnan(arr[0, 0, 3, 0, 0])


def test_fps_resolves_timestamp_bounds(tmp_path):
    """fps is only consulted for "hh:mm:ss" bounds: at 25fps, 00:00:02 is frame 50."""
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", list(range(80)))

    napari_to_body_joints_npz(_sequence(tmp_path, root, fps=25, start="00:00:02", end="00:00:03"), _config())

    _, desc = _load(tmp_path / "out.npz")
    assert desc["axis2"][0] == "000000050"
    assert len(desc["axis2"]) == 25


def test_integer_bounds_ignore_fps(tmp_path):
    """Frame-index bounds are used as-is, so fps cannot shift them."""
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", list(range(10)))

    napari_to_body_joints_npz(_sequence(tmp_path, root, fps=99, start=2, end=5), _config())

    _, desc = _load(tmp_path / "out.npz")
    assert desc["axis2"] == ["000000002", "000000003", "000000004"]


# --- likelihood column ---


def test_missing_likelihood_gets_manual_confidence(tmp_path):
    """napari drops the likelihood column when a human saves; placed points get confidence 1.0."""
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1], coords=["x", "y"])

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    arr, desc = _load(tmp_path / "out.npz")
    assert desc["axis4"] == ["coordinate_x", "coordinate_y", "confidence_score"]
    conf = arr[..., 2]
    assert set(np.unique(conf[~np.isnan(conf)])) == {1.0}


def test_unplaced_keypoints_keep_nan_confidence(tmp_path):
    """A keypoint the annotator never placed is NaN in x/y and must stay NaN in confidence."""
    root = tmp_path / "annotations"
    path = _write_camera(root, "cam3", [0, 1], coords=["x", "y"])
    df = pd.read_hdf(path)
    df.iloc[0, 0:2] = np.nan  # first individual/joint on frame 0 unplaced
    df.to_hdf(path, key="df_with_missing", mode="w", format="fixed")

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    arr, _ = _load(tmp_path / "out.npz")
    assert np.isnan(arr[0, 0, 0, 0, 0])  # x
    assert np.isnan(arr[0, 0, 0, 0, 2])  # confidence, not 1.0


def test_likelihood_is_used_when_present(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1], fill=7.0)

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    arr, _ = _load(tmp_path / "out.npz")
    # _write_h5 fills every cell with fill+frame, so confidence follows the file, not 1.0
    assert arr[0, 0, 0, 0, 2] == 7.0


def test_missing_x_or_y_is_still_rejected(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1], coords=["x", "likelihood"])

    with pytest.raises(ValueError, match=r"expected coords \['y'\]"):
        napari_to_body_joints_npz(_sequence(tmp_path, root), _config())


# --- sparse frame axis ---


def test_unannotated_frames_are_not_emitted(tmp_path):
    """A windowed export leaves gaps; they must be absent from axis2, not NaN-padded.

    Evaluation aligns on frame labels, so a padded gap would be scored as a failed
    prediction instead of being skipped.
    """
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1, 30, 31, 60])

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    arr, desc = _load(tmp_path / "out.npz")
    assert desc["axis2"] == ["000000000", "000000001", "000000030", "000000031", "000000060"]
    assert arr.shape[2] == 5
    assert not np.isnan(arr[..., 0]).any()


def test_sparse_frames_keep_their_own_values(tmp_path):
    """Row order must follow the sorted frame numbers, not the file's row order."""
    root = tmp_path / "annotations"
    # _write_h5 encodes the frame number into every cell as fill + frame
    _write_camera(root, "cam3", [60, 0, 30], fill=0.0)

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    arr, desc = _load(tmp_path / "out.npz")
    assert desc["axis2"] == ["000000000", "000000030", "000000060"]
    assert list(arr[0, 0, :, 0, 0]) == [0.0, 30.0, 60.0]


def test_cameras_with_disjoint_frames_are_unioned(tmp_path):
    """Frames annotated for only one camera are kept, NaN for the camera that lacks them."""
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 10])
    _write_camera(root, "cam4", [10, 20])

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    arr, desc = _load(tmp_path / "out.npz")
    assert desc["axis2"] == ["000000000", "000000010", "000000020"]
    assert np.isnan(arr[0, 1, 0, 0, 0])  # cam4 has no frame 0
    assert np.isnan(arr[0, 0, 2, 0, 0])  # cam3 has no frame 20
    assert not np.isnan(arr[0, 0, 1, 0, 0])  # both have frame 10


def test_reset_frames_renumbers_sparse_frames(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 30, 60])

    napari_to_body_joints_npz(_sequence(tmp_path, root, reset_frames=True), _config())

    _, desc = _load(tmp_path / "out.npz")
    assert desc["axis2"] == ["000000000", "000000001", "000000002"]


def test_empty_frame_range_is_rejected(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1, 2])

    with pytest.raises(ValueError, match="No annotated frames in range"):
        napari_to_body_joints_npz(_sequence(tmp_path, root, start=100, end=200), _config())


# --- frame range bounds ---


def test_end_minus_one_keeps_the_last_annotated_frame(tmp_path):
    """ "Until the end" must include the final frame, not drop it to a half-open bound."""
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1, 2])

    napari_to_body_joints_npz(_sequence(tmp_path, root, end=-1), _config())

    arr, desc = _load(tmp_path / "out.npz")
    assert desc["axis2"] == ["000000000", "000000001", "000000002"]
    assert not np.isnan(arr[0, 0, 2, 0, 0])  # last frame carries data


def test_explicit_end_stays_half_open(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1, 2, 3])

    napari_to_body_joints_npz(_sequence(tmp_path, root, start=0, end=3), _config())

    _, desc = _load(tmp_path / "out.npz")
    assert desc["axis2"] == ["000000000", "000000001", "000000002"]


# --- validation ---


def test_rejects_cameras_with_different_bodyparts(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1])
    _write_camera(root, "cam4", [0, 1], joints=["nose", "right_ear"])

    with pytest.raises(ValueError, match="bodyparts differ"):
        napari_to_body_joints_npz(_sequence(tmp_path, root), _config())


def test_rejects_cameras_with_different_individuals(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1])
    _write_camera(root, "cam4", [0, 1], individuals=["pL", "pX"])

    with pytest.raises(ValueError, match="individuals .* differ"):
        napari_to_body_joints_npz(_sequence(tmp_path, root), _config())


def test_rejects_unknown_camera_in_mapping(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1])

    with pytest.raises(ValueError, match="not in \\[cameras\\] mapping"):
        napari_to_body_joints_npz(_sequence(tmp_path, root), _config(cameras={"cam4": "view_top"}))


# --- discovery failures ---


def test_rejects_missing_input_root(tmp_path):
    with pytest.raises(NotADirectoryError, match="not a directory"):
        napari_to_body_joints_npz(_sequence(tmp_path, tmp_path / "nope"), _config())


def test_rejects_root_with_no_camera_subfolders(tmp_path):
    root = tmp_path / "annotations"
    root.mkdir()
    (root / "CollectedData_X.h5").write_bytes(b"stray file, no camera folders")

    with pytest.raises(FileNotFoundError, match="No camera subfolders"):
        napari_to_body_joints_npz(_sequence(tmp_path, root), _config())


def test_csv_annotation_files_are_not_discovered(tmp_path):
    """CSV support was dropped; a camera folder holding only a .csv is not usable."""
    root = tmp_path / "annotations"
    (root / "cam3").mkdir(parents=True)
    (root / "cam3" / "CollectedData_X.csv").write_text("scorer,individuals\n")

    with pytest.raises(FileNotFoundError, match="No annotation file"):
        napari_to_body_joints_npz(_sequence(tmp_path, root), _config())


def test_rejects_camera_folder_without_annotation_file(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1])
    (root / "cam4").mkdir()
    (root / "cam4" / "000000000.png").write_bytes(b"frame only")

    with pytest.raises(FileNotFoundError, match="No annotation file"):
        napari_to_body_joints_npz(_sequence(tmp_path, root), _config())


def test_rejects_camera_folder_with_several_annotation_files(tmp_path):
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1])
    _write_h5(root / "cam3" / "CollectedData_Other.h5", [0, 1])

    with pytest.raises(ValueError, match="Expected one annotation file"):
        napari_to_body_joints_npz(_sequence(tmp_path, root), _config())


def test_ignores_frame_images_beside_the_annotation_file(tmp_path):
    """Our own export leaves .png frames next to the .h5; they must not confuse discovery."""
    root = tmp_path / "annotations"
    _write_camera(root, "cam3", [0, 1])
    (root / "cam3" / "000000000.png").write_bytes(b"frame")
    (root / "cam3" / "000000001.png").write_bytes(b"frame")

    napari_to_body_joints_npz(_sequence(tmp_path, root), _config())

    _, desc = _load(tmp_path / "out.npz")
    assert desc["axis1"] == ["cam3"]
