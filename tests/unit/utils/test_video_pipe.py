import json
import os
import stat
import tempfile
from unittest import mock

import numpy as np

from nicetoolbox.utils import ffmpeg
from nicetoolbox.utils.ffmpeg import get_ffmpeg_path, get_ffprobe_path, initialize_ffmpeg
from nicetoolbox.utils.system import detect_arch, detect_os_type
from nicetoolbox.utils.video import FFmpegPipeWriter, probe_video, render_subtitled_track


def _make_mock_binary(directory: str, name: str) -> str:
    """Create a minimal executable shell stub to satisfy path + access checks."""
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write("#!/bin/sh\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _setup_mock_assets(tmp_root: str) -> str:
    """
    Create the expected binary layout under tmp_root using stub scripts.

    Use for pure unit tests that mock subprocess.run (e.g. probe_video, path resolution).
    Layout: <tmp_root>/ffmpeg/<os>/<arch>/ffmpeg[.exe] and ffprobe[.exe]
    """
    os_type = detect_os_type()
    arch = detect_arch()
    binary_dir = os.path.join(tmp_root, "ffmpeg", os_type, arch, "bin")
    os.makedirs(binary_dir, exist_ok=True)

    suffix = ".exe" if os_type == "windows" else ""
    _make_mock_binary(binary_dir, f"ffmpeg{suffix}")
    _make_mock_binary(binary_dir, f"ffprobe{suffix}")

    return tmp_root


def _setup_real_assets(tmp_root: str) -> str:
    """
    Create the expected binary layout under tmp_root by symlinking system ffmpeg/ffprobe.

    Use for integration tests that actually encode video (e.g. FFmpegPipeWriter streaming).
    Skips the test if system ffmpeg/ffprobe are not found.
    """
    import shutil

    import pytest

    os_type = detect_os_type()
    arch = detect_arch()
    binary_dir = os.path.join(tmp_root, "ffmpeg", os_type, arch, "bin")
    os.makedirs(binary_dir, exist_ok=True)

    suffix = ".exe" if os_type == "windows" else ""
    for name in (f"ffmpeg{suffix}", f"ffprobe{suffix}"):
        system_bin = shutil.which(name.removesuffix(".exe") if suffix else name)
        if system_bin is None:
            pytest.skip(f"System {name} not found; skipping integration test")
        dest = os.path.join(binary_dir, name)
        try:
            os.symlink(system_bin, dest)
        except OSError:
            shutil.copy2(system_bin, dest)

    return tmp_root


def test_get_ffmpeg_path():
    """get_ffmpeg_path() returns a non-empty string after initialize_ffmpeg() is called."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assets_root = _setup_mock_assets(tmpdir)
        initialize_ffmpeg(assets_root)

        path = get_ffmpeg_path()
        assert path is not None
        assert isinstance(path, str)
        assert len(path) > 0
        assert os.path.isfile(path)


def test_get_ffprobe_path():
    """get_ffprobe_path() returns a non-empty string after initialize_ffmpeg() is called."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assets_root = _setup_mock_assets(tmpdir)
        initialize_ffmpeg(assets_root)

        path = get_ffprobe_path()
        assert path is not None
        assert isinstance(path, str)
        assert len(path) > 0
        assert os.path.isfile(path)


def test_get_ffmpeg_path_raises_without_init():
    """get_ffmpeg_path() raises RuntimeError when initialize_ffmpeg() has not been called."""
    with mock.patch.object(ffmpeg, "_ffmpeg_path", None):
        try:
            get_ffmpeg_path()
            assert not True, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "ffmpeg not initialized" in str(exc)


def test_get_ffprobe_path_raises_without_init():
    """get_ffprobe_path() raises RuntimeError when initialize_ffmpeg() has not been called."""
    with mock.patch.object(ffmpeg, "_ffprobe_path", None):
        try:
            get_ffprobe_path()
            assert not True, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "ffprobe not initialized" in str(exc)


# ---------------------------------------------------------------------------
# Sample ffprobe JSON output used by probe_video tests
# ---------------------------------------------------------------------------
SAMPLE_FFPROBE_JSON = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30/1",
            "nb_frames": "300",
            "duration": "10.000000",
        }
    ],
    "format": {
        "filename": "/tmp/sample.mp4",
        "duration": "10.000000",
    },
}


def test_probe_video_uses_ffprobe():
    """probe_video() calls ffprobe and returns the parsed JSON dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assets_root = _setup_mock_assets(tmpdir)
        initialize_ffmpeg(assets_root)

        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(SAMPLE_FFPROBE_JSON).encode()
        mock_result.stderr = b""

        with mock.patch("nicetoolbox.utils.video.subprocess.run", return_value=mock_result) as mock_run:
            result = probe_video("/tmp/sample.mp4")

        # subprocess.run should have been called with ffprobe args
        assert mock_run.called
        call_args = mock_run.call_args[0][0]  # first positional arg = command list
        assert any("ffprobe" in str(a) for a in call_args)
        assert "-show_streams" in call_args
        assert "-show_format" in call_args
        assert "/tmp/sample.mp4" in call_args

        # Result should mirror the ffprobe JSON directly
        assert result["format"]["filename"] == "/tmp/sample.mp4"
        video_stream = next(s for s in result["streams"] if s["codec_type"] == "video")
        assert video_stream["codec_name"] == "h264"
        assert video_stream["width"] == 1920
        assert video_stream["height"] == 1080


def test_probe_video_raises_on_nonzero_exit():
    """probe_video() raises RuntimeError when ffprobe returns a non-zero exit code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assets_root = _setup_mock_assets(tmpdir)
        initialize_ffmpeg(assets_root)

        mock_result = mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b""
        mock_result.stderr = b"No such file or directory"

        with mock.patch("nicetoolbox.utils.video.subprocess.run", return_value=mock_result):
            try:
                probe_video("/nonexistent/video.mp4")
                assert not True, "Expected RuntimeError"
            except RuntimeError as exc:
                assert "ffprobe failed" in str(exc)


def test_ffmpeg_pipe_writer_streams_video():
    with tempfile.TemporaryDirectory() as tmpdir:
        assets_root = _setup_real_assets(tmpdir)
        initialize_ffmpeg(assets_root)

        out_video = os.path.join(tmpdir, "test_stream.mp4")
        width, height, fps, n_frames = 320, 240, 30, 15

        with FFmpegPipeWriter(out_video, size=(width, height), fps=fps) as writer:
            for i in range(n_frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :, 0] = (i * 10) % 256
                writer.write_frame(frame)

        assert os.path.isfile(out_video)
        assert os.path.getsize(out_video) > 0


def test_ffmpeg_pipe_writer_render_subtitled_track_skips_missing_srt():
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_srt = os.path.join(tmpdir, "missing.srt")
        out_video = os.path.join(tmpdir, "sub.mp4")
        audio_file = os.path.join(tmpdir, "audio.wav")

        result = render_subtitled_track(
            srt_path=missing_srt,
            audio_path=audio_file,
            output_path=out_video,
            fps=30.0,
        )
        assert result is False
        assert not os.path.exists(out_video)


def test_ffmpeg_pipe_writer_save_frames_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        assets_root = _setup_real_assets(tmpdir)
        initialize_ffmpeg(assets_root)

        out_video = os.path.join(tmpdir, "test_save.mp4")
        frames_dir = os.path.join(tmpdir, "images", "cam01")
        width, height, fps, n_frames = 160, 120, 30, 5

        with FFmpegPipeWriter(out_video, size=(width, height), fps=fps, save_frames_dir=frames_dir) as writer:
            for i in range(n_frames):
                frame = np.full((height, width, 3), i * 20, dtype=np.uint8)
                writer.write_frame(frame, frame_idx=i)

        assert os.path.isfile(out_video)
        assert os.path.getsize(out_video) > 0
        assert os.path.isdir(frames_dir)
        saved_images = os.listdir(frames_dir)
        assert len(saved_images) == n_frames
        assert "000000000.jpg" in saved_images
