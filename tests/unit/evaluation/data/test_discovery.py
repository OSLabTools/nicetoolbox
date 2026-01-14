from unittest.mock import MagicMock

import pytest

from nicetoolbox.configs.schemas.detectors_run_file import RunConfigVideo
from nicetoolbox.evaluation.data.discovery import ChunkWorkItem, DiscoveryEngine

# --- Tests for _parse_gt_key (Pure Function) ---


@pytest.mark.parametrize(
    "session,sequence,gt_dim_key,expected",
    [
        ("S1", "Seq1", "3d", "S1__Seq1__3d"),
        ("S2", "SeqA", "2d", "S2__SeqA__2d"),
        ("SessionX", "SequenceY", "depth", "SessionX__SequenceY__depth"),
        ("S1", "", "3d", "S1__3d"),
        ("S1", None, "3d", "S1__3d"),
    ],
)
def test_parse_gt_key(session, sequence, gt_dim_key, expected):
    """
    Parameterized tests for _parse_gt_key to cover multiple combinations
    without duplicating boilerplate.
    """

    result = DiscoveryEngine._parse_gt_key(session=session, sequence=sequence, gt_dim_key=gt_dim_key)
    assert result == expected


# --- Tests for _create_reconciliation_maps (Pure Function) ---


@pytest.mark.parametrize(
    "pred_desc, gt_desc, expected_pred_map, expected_gt_map",
    [
        (
            # Case 1: Identical axis3 and axis4
            {"axis3": ["a", "b"], "axis4": ["x", "y"]},
            {"axis3": ["a", "b"], "axis4": ["x", "y"]},
            {},  # No reconciliation needed
            {},  # No reconciliation needed
        ),
        (
            # Case 2: Axis3 overlap, identical axis4
            {"axis3": ["a", "b"], "axis4": ["x", "y"]},
            {"axis3": ["b", "a"], "axis4": ["x", "y"]},
            {"axis3": (0, 1)},
            {"axis3": (1, 0)},
        ),
        (
            # Case 3: Axis3 overlap, Different axis4 with partial overlap
            {"axis3": ["a", "b"], "axis4": ["x", "y", "z"]},
            {"axis3": ["b", "a"], "axis4": ["y", "x"]},
            {"axis3": (0, 1), "axis4": (0, 1)},
            {"axis3": (1, 0), "axis4": (1, 0)},
        ),
        (
            # Case 4: Both axes partially overlap
            {"axis3": ["a", "b", "c", "d"], "axis4": ["x", "y", "z", "conf"]},
            {"axis3": ["c", "b", "a"], "axis4": ["x", "y", "conf"]},
            {"axis3": (0, 1, 2), "axis4": (0, 1, 3)},
            {"axis3": (2, 1, 0), "axis4": (0, 1, 2)},
        ),
        # Case 5: NO overlaps (should raise ValueError) - see next test.
    ],
)
def test_create_reconciliation_maps_success_cases(pred_desc, gt_desc, expected_pred_map, expected_gt_map):
    """
    Tests the core reconciliation logic for scenarios that should succeed.
    """
    pred_map, gt_map = DiscoveryEngine._create_reconciliation_maps(pred_desc, gt_desc)
    assert pred_map == expected_pred_map
    assert gt_map == expected_gt_map


def test_create_reconciliation_maps_raises_error_on_no_overlap():
    """
    Tests that a ValueError is raised when prediction and GT have no
    common labels in axis3 or axis4.
    """
    # Arrange
    pred_desc = {"axis3": ["a", "b"], "axis4": ["x", "y"]}
    gt_desc = {"axis3": ["c", "d"], "axis4": ["n", "m"]}  # No common labels

    # Act & Assert
    # Test successful if a ValueError is raised
    with pytest.raises(ValueError) as excinfo:
        DiscoveryEngine._create_reconciliation_maps(pred_desc, gt_desc)

    # Assert that the error massage is actually helpful
    assert "no common labels" in str(excinfo.value)
    assert "('a', 'b')" in str(excinfo.value)
    assert "('c', 'd')" in str(excinfo.value)
    assert "('x', 'y')" in str(excinfo.value)
    assert "('n', 'm')" in str(excinfo.value)


# --- Tests for _load_and_process_annot_descriptions ---


def test_load_and_process_annot_descriptions_with_synonyms(discovery_engine_factory):
    """
    Tests that synonyms from dataset properties are correctly applied to the
    loaded annotation descriptions.
    """
    # Arrange: Use the factory to create a scenario with specific synonyms
    mock_annot_desc = {"S1__Seq1__3d": {"axis3": ["original_name_1", "name_2"]}}
    synonyms = {"axis3": {"original_name_1": "new_synonym_name"}}

    engine = discovery_engine_factory(dataset_props_overrides={"synonyms": synonyms}, mock_annot_desc=mock_annot_desc)

    # Act: The method is called during the engine's __init__
    processed_desc = engine.annot_descriptions

    # Assert
    assert "S1__Seq1__3d" in processed_desc
    final_axis3 = processed_desc["S1__Seq1__3d"]["axis3"]
    assert final_axis3 == ["new_synonym_name", "name_2"]


def test_load_and_process_returns_none_if_gt_not_needed(discovery_engine_factory):
    """Tests that the method short-circuits if no metric requires ground truth."""
    # Arrange: Use the factory to create a scenario where gt is not required
    engine = discovery_engine_factory(gt_needed=False)

    # Act
    processed_desc = engine.annot_descriptions

    # Assert
    assert processed_desc is None


# --- Tests for _create_chunks_from_description ---


def test_create_chunks_filters_frames_correctly(discovery_engine_factory):
    """
    Tests that only frames within the video_start and video_length
    of the run_config are included in the final ChunkWorkItem.
    """
    # Arrange
    pred_desc = {
        "3d": {
            "axis0": ["p1"],
            "axis1": ["c1"],
            "axis2": [str(i) for i in range(20)],  # Frames 0-19
        }
    }
    # We define the run to only process frames from 5 to 14 (length 10)
    run_config_overrides = {
        "videos": [
            RunConfigVideo(  # Use the real Pydantic model here
                video_start=5, video_length=10, session_ID="S1", sequence_ID="Seq1"
            )
        ]
    }

    engine = discovery_engine_factory(
        run_config_overrides=run_config_overrides,
        mock_pred_desc=pred_desc,
        gt_needed=False,  # Simplify test by not requiring GT
    )

    # Act
    # We call the "private" method directly for this unit test
    chunks = engine._create_chunks_from_description(
        pred_descriptions=pred_desc,
        pred_path=MagicMock(),
        metric_cfg=MagicMock(),
        start_frame=5,
        end_frame=15,
        session="S1",
        sequence="Seq1",
        component="body_joints",
        algorithm="vitpose",
        metric_type="point_cloud_metrics",
    )

    # Assert
    assert len(chunks) == 1
    work_item = chunks[0]
    assert isinstance(work_item, ChunkWorkItem)

    # The crucial check: we should have 10 frames, from 5 to 14
    assert len(work_item.frames) == 10
    assert work_item.frames[0].frame == 5
    assert work_item.frames[-1].frame == 14


def test_create_chunks_handles_missing_gt_for_frame(discovery_engine_factory):
    """
    Tests that if a frame exists in predictions but not in annotations,
    it is still included but with its annot_slicing_indices set to None.
    """
    # Arrange
    pred_desc = {
        "3d": {
            "axis0": ["p1"],
            "axis1": ["c1"],
            "axis2": ["0", "1"],
            "axis3": ["nose", "eye"],
            "axis4": ["x", "y"],
        }
    }
    # Annotation is missing frame '1' but has the same axes
    annot_desc = {
        "S1__Seq1__3d": {
            "axis0": ["p1"],
            "axis1": ["c1"],
            "axis2": ["0"],
            "axis3": ["nose", "eye"],
            "axis4": ["x", "y"],
        }
    }
    engine = discovery_engine_factory(mock_pred_desc=pred_desc, mock_annot_desc=annot_desc)

    # Act
    chunks = engine._create_chunks_from_description(
        pred_descriptions=pred_desc,
        pred_path=MagicMock(),
        metric_cfg=engine.eval_config.metric_types["point_cloud_metrics"],
        start_frame=0,
        end_frame=2,
        session="S1",
        sequence="Seq1",
        component="body_joints",
        algorithm="vitpose",
        metric_type="point_cloud_metrics",
    )

    # Assert
    assert len(chunks) == 1
    frames_info = chunks[0].frames
    assert len(frames_info) == 2

    # Frame 0 exists in both, so it should have slicing indices
    assert frames_info[0].frame == 0
    assert frames_info[0].annot_slicing_indices is not None

    # Frame 1 is missing from GT, so its slicing indices must be None
    assert frames_info[1].frame == 1
    assert frames_info[1].annot_slicing_indices is None
