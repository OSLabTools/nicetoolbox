# ruff: noqa: ARG001
import pytest

from nicetoolbox.configs.schemas.detectors_run_file import FRAME_EXTRACTION_FILENAME
from nicetoolbox.detectors.data_handlers.video_handler import VideoDataHandler
from tests.unit.detectors.data_handlers.video.conftest import (
    make_flat_handler,
    make_frames_cache,
    make_io,
    make_sequence_context,
)


def test_recipe_frames_folders_default_layout(tmp_path, default_vid_patches):
    cameras = ["cam_front"]
    handler, ctx, io = make_flat_handler(tmp_path, cameras)
    make_frames_cache(io, ctx)
    handler.prepare()

    recipe = handler.get_recipe()
    assert recipe.frames_folders == {"cam_front": str(tmp_path / "nice_input" / "cam_front" / "frames")}
    assert recipe.frame_path("cam_front", 7) == tmp_path / "nice_input" / "cam_front" / "frames" / "000000007.png"


def test_recipe_frames_folders_per_camera_roots(tmp_path, default_vid_patches):
    # each camera on its own "drive"
    cameras = ["cam_front", "cam_top"]
    handler, _, io = make_flat_handler(tmp_path, cameras)
    io.camera_input_folders["cam_front"] = tmp_path / "drive_d" / "cam_front"
    io.camera_input_folders["cam_top"] = tmp_path / "drive_e" / "cam_top"
    handler.prepare()

    # frames extracted into the per-camera folders
    extracted_to = {call.args[1] for call in default_vid_patches["split_into_frames"].call_args_list}
    for cam, drive in [("cam_front", "drive_d"), ("cam_top", "drive_e")]:
        frames_dir = tmp_path / drive / cam / "frames"
        assert frames_dir.is_dir()
        assert (tmp_path / drive / cam / "meta.json").is_file()
        assert (tmp_path / drive / cam / FRAME_EXTRACTION_FILENAME).is_file()
        assert str(frames_dir) + "/" in extracted_to

    recipe = handler.get_recipe()
    assert recipe.frames_dir("cam_front") == tmp_path / "drive_d" / "cam_front" / "frames"
    assert recipe.frames_dir("cam_top") == tmp_path / "drive_e" / "cam_top" / "frames"
    assert recipe.frame_path("cam_top", 3) == tmp_path / "drive_e" / "cam_top" / "frames" / "000000003.png"


def test_recipe_unsorted_cameras_are_sorted(tmp_path, default_vid_patches):
    cameras = ["cam_top", "cam_front"]  # intentionally unsorted
    handler, ctx, io = make_flat_handler(tmp_path, cameras)
    make_frames_cache(io, ctx)
    handler.prepare()

    recipe = handler.get_recipe()
    assert recipe.camera_names == ["cam_front", "cam_top"]


def test_recipe_before_prepare_raises(tmp_path):
    ctx = make_sequence_context({"cam_front": tmp_path / "cam_front.mp4"})
    handler = VideoDataHandler(io=make_io(tmp_path, ctx), subsequence_context=ctx)

    with pytest.raises((AttributeError, TypeError)):
        handler.get_recipe()
