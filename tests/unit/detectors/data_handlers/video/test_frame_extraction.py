# ruff: noqa: ARG001
import pytest
from pydantic import TypeAdapter, ValidationError

from nicetoolbox.configs.schemas.detectors_run_file import (
    FRAME_EXTRACTION_FILENAME,
    FrameExtractionConfig,
    JpegFrameExtraction,
    PngFrameExtraction,
    load_frame_extraction,
)
from tests.unit.detectors.data_handlers.video.conftest import FRAME_EXTRACTION, make_flat_handler, make_frames_cache

ADAPTER = TypeAdapter(FrameExtractionConfig)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, expected_type",
    [
        ({"frame_extension": ".png", "compression_level": 9}, PngFrameExtraction),
        ({"frame_extension": ".jpg", "compression_quality": 2}, JpegFrameExtraction),
        ({"frame_extension": ".jpeg", "compression_quality": 31}, JpegFrameExtraction),
    ],
)
def test_schema_selects_format(data, expected_type):
    assert type(ADAPTER.validate_python(data)) is expected_type


@pytest.mark.parametrize(
    "data",
    [
        {"frame_extension": ".jpg", "compression_level": 2},  # key of another format
        {"frame_extension": ".png", "compression_quality": 2},
        {"frame_extension": ".png", "compression_level": 10},  # out of range
        {"frame_extension": ".jpg", "compression_quality": 0},
        {"frame_extension": "png", "compression_level": 2},  # missing dot
        {"frame_extension": ".bmp"},
    ],
)
def test_schema_rejects_invalid(data):
    with pytest.raises(ValidationError):
        ADAPTER.validate_python(data)


# ---------------------------------------------------------------------------
# Cache validation in VideoDataHandler
# ---------------------------------------------------------------------------


def test_settings_saved_next_to_frames(tmp_path, default_vid_patches):
    handler, _ctx, io = make_flat_handler(tmp_path, ["cam_front"])
    handler.prepare()

    assert (io.nice_input_folder / "cam_front" / FRAME_EXTRACTION_FILENAME).is_file()
    assert load_frame_extraction(io.nice_input_folder / "cam_front") == FRAME_EXTRACTION
    split_kwargs = default_vid_patches["split_into_frames"].call_args.kwargs
    assert split_kwargs["frame_extraction"] == FRAME_EXTRACTION
    assert split_kwargs["hwaccel"] == "cuda"


def test_changed_settings_reextract_and_drop_old_frames(tmp_path, default_vid_patches):
    handler, ctx, io = make_flat_handler(tmp_path, ["cam_front"])
    old = JpegFrameExtraction(frame_extension=".jpg", compression_quality=2)
    make_frames_cache(io, ctx, frame_extraction=old)

    handler.prepare()

    default_vid_patches["split_into_frames"].assert_called_once()
    assert not list((io.nice_input_folder / "cam_front" / "frames").glob("*.jpg"))
    assert load_frame_extraction(io.nice_input_folder / "cam_front") == FRAME_EXTRACTION


def test_missing_settings_reextract(tmp_path, default_vid_patches):
    handler, ctx, io = make_flat_handler(tmp_path, ["cam_front"])
    make_frames_cache(io, ctx)
    (io.nice_input_folder / "cam_front" / FRAME_EXTRACTION_FILENAME).unlink()

    handler.prepare()

    default_vid_patches["split_into_frames"].assert_called_once()


def test_only_stale_camera_reextracted(tmp_path, default_vid_patches):
    handler, ctx, io = make_flat_handler(tmp_path, ["cam_front", "cam_top"])
    make_frames_cache(io, ctx)
    stale = PngFrameExtraction(frame_extension=".png", compression_level=9)
    (io.nice_input_folder / "cam_top" / FRAME_EXTRACTION_FILENAME).write_text(stale.model_dump_json())

    handler.prepare()

    default_vid_patches["split_into_frames"].assert_called_once()
    assert default_vid_patches["split_into_frames"].call_args.args[0].endswith("cam_top.mp4")
