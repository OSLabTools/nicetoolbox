# ruff: noqa: ARG001
import pytest

from nicetoolbox.detectors.data_handlers.video_handler import VideoDataHandler
from tests.unit.detectors.data_handlers.video.conftest import (
    assert_handler_output,
    make_flat_handler,
    make_frames_cache,
    make_io,
    make_sequence_context,
)

# ---------------------------------------------------------------------------
# Direct paths
# ---------------------------------------------------------------------------


def test_single_camera_direct_path(tmp_path, default_vid_patches):
    handler, _ctx, _io = make_flat_handler(tmp_path, ["cam_front"])
    handler.prepare()

    assert handler.camera_video_paths == {"cam_front": tmp_path / "data_source" / "cam_front.mp4"}
    assert_handler_output(handler, ["cam_front"])
    default_vid_patches["split_into_frames"].assert_called_once()


def test_two_cameras_direct_path(tmp_path, default_vid_patches):
    cameras = ["cam_front", "cam_top"]
    handler, _ctx, _io = make_flat_handler(tmp_path, cameras)
    handler.prepare()

    for cam in cameras:
        assert handler.camera_video_paths[cam] == tmp_path / "data_source" / f"{cam}.mp4"
    assert_handler_output(handler, cameras)
    assert default_vid_patches["split_into_frames"].call_count == 2


def test_direct_path_reused_from_cache(tmp_path, default_vid_patches):
    handler, ctx, io = make_flat_handler(tmp_path, ["cam_front"])
    make_frames_cache(io, ctx)
    handler.prepare()

    assert_handler_output(handler, ["cam_front"])
    default_vid_patches["split_into_frames"].assert_not_called()


def test_missing_file_raises(tmp_path, default_vid_patches):
    tracks = {"cam_front": tmp_path / "does_not_exist.mp4"}
    ctx = make_sequence_context(tracks)
    handler = VideoDataHandler(io=make_io(tmp_path), subsequence_context=ctx)

    with pytest.raises(FileNotFoundError, match="file not found"):
        handler.prepare()


# ---------------------------------------------------------------------------
# Glob wildcards
# ---------------------------------------------------------------------------


def test_glob_resolves_single_match(tmp_path, default_vid_patches):
    subfolder = tmp_path / "Cam1"
    subfolder.mkdir()
    expected = subfolder / "PIS_ID_00_2_Cam1_20200811_043527.036.mp4"
    expected.touch()

    tracks = {"Cam1": subfolder / "*.mp4"}
    ctx = make_sequence_context(tracks)
    handler = VideoDataHandler(io=make_io(tmp_path), subsequence_context=ctx)
    handler.prepare()

    assert handler.camera_video_paths == {"Cam1": expected}


def test_glob_zero_matches_raises(tmp_path, default_vid_patches):
    subfolder = tmp_path / "Cam1"
    subfolder.mkdir()

    tracks = {"Cam1": subfolder / "*.mp4"}
    ctx = make_sequence_context(tracks)
    handler = VideoDataHandler(io=make_io(tmp_path), subsequence_context=ctx)

    with pytest.raises(ValueError, match="no files match"):
        handler.prepare()


def test_glob_multiple_matches_raises(tmp_path, default_vid_patches):
    subfolder = tmp_path / "Cam1"
    subfolder.mkdir()
    (subfolder / "a.mp4").touch()
    (subfolder / "b.mp4").touch()

    tracks = {"Cam1": subfolder / "*.mp4"}
    ctx = make_sequence_context(tracks)
    handler = VideoDataHandler(io=make_io(tmp_path), subsequence_context=ctx)

    with pytest.raises(ValueError, match="matches 2 files"):
        handler.prepare()
