# ruff: noqa: ARG001
import numpy as np
import pytest

from nicetoolbox.detectors.data_handlers.video_handler import VideoDataHandler
from tests.unit.detectors.data_handlers.video.conftest import (
    assert_handler_output,
    make_flat_handler,
    make_frames_cache,
    make_io,
    make_sequence_context,
)


def test_no_calibration_file(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    handler, ctx, io = make_flat_handler(tmp_path, cameras)
    make_frames_cache(io, ctx)
    handler.prepare()

    assert_handler_output(handler, cameras, calibration_is_none=True)


def test_valid_calibration_filters_to_active_cameras(tmp_path, default_vid_patches):
    # calib file has cam_front, cam_top, cam_unknown — only active cameras are kept
    cameras = ["cam_front", "cam_top"]
    handler, ctx, io = make_flat_handler(tmp_path, cameras)
    make_frames_cache(io, ctx)

    calib_path = tmp_path / "calibration.npz"
    calib_data = {
        "cam_front": {"intrinsics": np.eye(3)},
        "cam_top": {"intrinsics": np.eye(3)},
        "cam_unknown": {"intrinsics": np.eye(3)},
    }
    np.savez(calib_path, **{"seq_01": calib_data})
    io.get_calibration_file.return_value = str(calib_path)

    handler.prepare()

    assert_handler_output(handler, cameras, calibration_keys={"cam_front", "cam_top"})


def test_missing_calibration_key_raises(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    handler, ctx, io = make_flat_handler(tmp_path, cameras)
    make_frames_cache(io, ctx)

    calib_path = tmp_path / "calibration.npz"
    np.savez(calib_path, **{"wrong_key": {"data": 1}})
    io.get_calibration_file.return_value = str(calib_path)

    with pytest.raises(KeyError):
        handler.prepare()


def test_sequence_id_only_calibration_key(tmp_path, default_vid_patches):
    # Calibration is keyed purely by sequence_id.
    cameras = ["cam_front"]
    video_path = tmp_path / "cam_front.mp4"
    video_path.touch()

    ctx = make_sequence_context({"cam_front": video_path}, sequence_id="seq_01")
    io = make_io(tmp_path)
    make_frames_cache(io, ctx)

    calib_path = tmp_path / "calibration.npz"
    calib_data = {"cam_front": {"intrinsics": np.eye(3)}}
    np.savez(calib_path, **{"seq_01": calib_data})
    io.get_calibration_file.return_value = str(calib_path)

    handler = VideoDataHandler(io=io, subsequence_context=ctx)
    handler.prepare()

    assert_handler_output(handler, cameras, calibration_keys={"cam_front"})
