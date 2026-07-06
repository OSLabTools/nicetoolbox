# ruff: noqa: ARG001
from pathlib import Path

import pytest

from nicetoolbox.detectors.data_handlers.video_handler import VideoDataHandler
from tests.unit.detectors.data_handlers.video.conftest import (
    VIDEO_FPS,
    VIDEO_FRAMES,
    FakeVideoInfo,
    make_flat_handler,
    make_frames_cache,
    make_io,
    make_sequence_context,
)


def test_fps_detected_from_probe(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    handler, ctx, io = make_flat_handler(tmp_path, cameras)
    make_frames_cache(io, ctx)

    handler.prepare()

    assert handler.fps == int(VIDEO_FPS)


def test_video_start_beyond_length_raises(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    # VIDEO_FRAMES=300, start at 500 → no frames available
    handler, _, _ = make_flat_handler(tmp_path, cameras, video_start=500, video_length=-1)

    with pytest.raises(ValueError, match="beyond the end"):
        handler.prepare()


def test_auto_length_detected_from_frame_count(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    handler, ctx, io = make_flat_handler(tmp_path, cameras, video_start=0, video_length=-1)
    make_frames_cache(io, ctx)

    default_vid_patches["json_to_video_info"].return_value = FakeVideoInfo(video_path=Path("fake.mp4"), frames=300)
    handler.prepare()

    assert handler.length_frames == VIDEO_FRAMES


def test_inconsistent_fps_across_cameras_raises(tmp_path, default_vid_patches):
    cameras = ["cam_front", "cam_top"]
    handler, _, _ = make_flat_handler(tmp_path, cameras, video_length=100)

    default_vid_patches["json_to_video_info"].side_effect = [
        FakeVideoInfo(video_path=Path("cam_front.mp4"), fps=30.0),
        FakeVideoInfo(video_path=Path("cam_top.mp4"), fps=25.0),
    ]

    with pytest.raises(ValueError, match="FPS"):
        handler.prepare()


def test_inconsistent_frame_counts_across_cameras_raises(tmp_path, default_vid_patches):
    cameras = ["cam_front", "cam_top"]
    handler, _, _ = make_flat_handler(tmp_path, cameras, video_length=-1)

    default_vid_patches["json_to_video_info"].side_effect = [
        FakeVideoInfo(video_path=Path("cam_front.mp4"), frames=300),
        FakeVideoInfo(video_path=Path("cam_top.mp4"), frames=250),
    ]

    with pytest.raises(ValueError, match="frame counts"):
        handler.prepare()


def test_timestamp_video_start(tmp_path, default_vid_patches):
    # "00-00-02" = 2 seconds at 30fps = frame 60; VIDEO_FRAMES=300 → length=240
    cameras = ["cam_front"]
    handler, _, io = make_flat_handler(tmp_path, cameras, video_start="00-00-02", video_length=-1)

    # make_frames_cache can't handle string video_start — create frames manually
    from nicetoolbox.detectors.data_handlers.video_handler import FILENAME_TEMPLATE

    for cam in cameras:
        frames_dir = io.nice_input_folder / cam / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(60, VIDEO_FRAMES):
            (frames_dir / FILENAME_TEMPLATE.format(idx=idx)).write_bytes(b"\x89PNG")

    handler.prepare()

    assert handler.start_frame == 60
    assert handler.length_frames == VIDEO_FRAMES - 60


def test_empty_camera_list_raises(tmp_path, default_vid_patches):
    ctx = make_sequence_context({}, video_length=100)
    handler = VideoDataHandler(io=make_io(tmp_path), subsequence_context=ctx)

    with pytest.raises(ValueError, match="No camera names"):
        handler.prepare()
