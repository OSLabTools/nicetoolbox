from unittest.mock import create_autospec, patch

import pytest
import torch

from nicetoolbox.configs.schemas.evaluation_config import EvaluationMetricType
from nicetoolbox.evaluation.data.discovery import ChunkWorkItem, FrameInfo
from nicetoolbox.evaluation.metrics.keypoint import (
    BoneLength,
    JumpDetection,
    KeypointMetric,
)


@pytest.fixture
def mock_keypoints_mapping():
    """Provides a standard keypoints mapping dictionary for all tests."""
    return {
        "human_pose": {
            "bone_dict": {
                "l_arm": ["l_shoulder", "l_elbow"],
                "r_arm": ["r_shoulder", "r_elbow"],
            },
            "joint_diameter_size": {
                "l_shoulder": 0.15,  # 15cm threshold in meters
                "l_elbow": 0.1,  # 10cm threshold in meters
            },
        }
    }


@pytest.fixture
def mock_keypoint_metadata():
    """Provides standard metadata for keypoint metrics."""
    mock_chunk = create_autospec(ChunkWorkItem, instance=True)
    mock_chunk.component = "body_pose"
    mock_chunk.algorithm = "mmpose"
    mock_chunk.pred_data_description_axis3 = ["l_shoulder", "l_elbow"]
    mock_chunk.pred_data_key = "3d"

    # FrameInfo needs a 'person' attribute for summary tests
    frame1 = create_autospec(FrameInfo, instance=True)
    frame1.person = "P1"
    frame1.camera = "cam1"

    frame2 = create_autospec(FrameInfo, instance=True)
    frame2.person = "P1"
    frame2.camera = "cam1"

    return mock_chunk, [frame1, frame2]


# --- (1) Tests for the KeypointMetric Handler ---


@patch("toml.load")  # Mock the file loading
def test_keypoint_handler_creates_metrics(patched_toml_load, mock_keypoints_mapping):
    """
    Given: A config specifying 'bone_length' and 'jump_detection'.
    When:  The KeypointMetric handler is initialized.
    Then:  It should create instances of the correct metric classes.
    """
    # Arrange
    mock_config = create_autospec(EvaluationMetricType, instance=True)
    mock_config.metric_names = ["bone_length", "jump_detection"]
    mock_config.keypoint_mapping_file = "fake/path.toml"
    # This is the mocked return value when toml.load is called
    patched_toml_load.return_value = mock_keypoints_mapping

    # Act
    handler = KeypointMetric(cfg=mock_config, device="cpu")

    # Assert
    # Check that the TOML file was "loaded"
    patched_toml_load.assert_called_once_with("fake/path.toml")

    assert "bone_length" in handler.metrics
    assert isinstance(handler.metrics["bone_length"], BoneLength)
    assert "jump_detection" in handler.metrics
    assert isinstance(handler.metrics["jump_detection"], JumpDetection)


# --- (2) Tests for the BoneLength Metric ---


def test_bonelength_calculates_correctly(
    mock_keypoints_mapping, mock_keypoint_metadata
):
    """
    Given: Predictions with known 3D coordinates.
    When:  The BoneLength metric is updated.
    Then:  It calculates the correct Euclidean distance for the defined bones.
    """
    # Arrange
    metric = BoneLength(mock_keypoints_mapping)
    mock_chunk, mock_frames = mock_keypoint_metadata

    # Preds: 2 frames, 2 joints (l_shoulder, l_elbow), 3D coords
    # Frame 1: l_shoulder at (0,0,0), l_elbow at (3,4,0) -> distance should be 5.0
    # Frame 2: l_shoulder at (1,1,1), l_elbow at (1,1,1) -> distance should be 0.0
    preds = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.99], [3.0, 4.0, 0.0, 0.95]],
            [[1.0, 1.0, 1.0, 0.80], [1.0, 1.0, 1.0, 0.85]],
        ]
    )

    # Act
    metric.update(preds, None, mock_chunk, mock_frames)
    results = metric.compute()

    # Assert Frame-Based Results
    frame_result_key = ("body_pose", "mmpose", "bone_length")
    batch_result = results[frame_result_key][0]
    expected_lengths = torch.tensor(
        [[5.0, torch.nan], [0.0, torch.nan]]
    )  # bone r_arm is NaN since joints not predicted/given
    assert torch.allclose(batch_result.results_tensor, expected_lengths, equal_nan=True)

    # Assert Summary Results for 'l_arm' and person 'P1'
    mean_key = ("body_pose", "mmpose", "P1", "l_arm", "mean")
    std_key = ("body_pose", "mmpose", "P1", "l_arm", "std")
    assert mean_key in results and std_key in results
    assert results[mean_key].value == pytest.approx(2.5)  # (5.0 + 0.0) / 2
    assert results[std_key].value == pytest.approx(3.5355, rel=1e-2)  # sample std


def test_bonelength_ignores_non_3d_key(mock_keypoints_mapping, mock_keypoint_metadata):
    """
    Given: A chunk with a pred_data_key that is not '3d'.
    When:  The BoneLength metric is updated.
    Then:  The metric's storage should remain empty.
    """
    # Arrange
    metric = BoneLength(mock_keypoints_mapping)
    mock_chunk, mock_frames = mock_keypoint_metadata
    mock_chunk.pred_data_key = "2d_interpolated"  # Not '3d'
    preds = torch.ones(2, 2, 2)

    # Act
    metric.update(preds, None, mock_chunk, mock_frames)

    # Assert
    assert not metric.storage
    assert not metric.summary_storage


def test_bonelength_raises_err_2d_data(mock_keypoints_mapping, mock_keypoint_metadata):
    """
    Given: A chunk with a pred_data_key that is not '3d'.
    When:  The BoneLength metric is updated.
    Then:  The metric's storage should remain empty.
    """
    # Arrange
    metric = BoneLength(mock_keypoints_mapping)
    mock_chunk, mock_frames = mock_keypoint_metadata
    mock_chunk.pred_data_key = "3d"
    preds = torch.ones(2, 2, 2)  # Accidentally 2D data

    # Act / Assert
    with pytest.raises(ValueError):
        metric.update(preds, None, mock_chunk, mock_frames)


# --- (3) Tests for the JumpDetection Metric ---


def test_jumpdetection_detects_jumps_correctly(
    mock_keypoints_mapping, mock_keypoint_metadata
):
    """
    Given: A sequence of predictions with small and large movements.
    When:  The JumpDetection metric is updated.
    Then:  It correctly identifies the frame with the large jump.
    """
    # Arrange
    metric = JumpDetection(mock_keypoints_mapping)
    mock_chunk, mock_frames = mock_keypoint_metadata
    mock_frames = mock_frames * 3

    # Thresholds are l_shoulder: 150mm, l_elbow: 100mm
    preds = torch.tensor(
        [
            # Frame 0: Initial position
            [[0.0, 0.0, 0.0, 0.99], [1000.0, 0.0, 0.0, 0.95]],
            # Frame 1: Move l_shoulder by 160mm (> 150mm) -> JUMP
            [[160.0, 0.0, 0.0, 0.99], [1000.0, 0.0, 0.0, 0.95]],
            # Frame 2: Move l_elbow by 99mm (< 100mm) -> NO JUMP
            [[160.0, 0.0, 0.0, 0.99], [1000.0, 99.0, 0.0, 0.95]],
            # Frame 3: Move both, but under thresholds -> NO JUMP
            [[170.0, 0.0, 0.0, 0.99], [1000.0, 109.0, 0.0, 0.95]],
            # Frame 4: Move l_elbow by 101mm (> 100mm) -> JUMP
            [[170.0, 0.0, 0.0, 0.99], [1000.0, 210.0, 0.0, 0.95]],
            # Frame 5: No movement for both joints -> NO JUMP
            [[170.0, 0.0, 0.0, 0.99], [1000.0, 210.0, 0.0, 0.95]],
        ],
        dtype=torch.float32,
    )

    # Act
    # Note: Jump is calculated for frame N based on frame N-1.
    metric.update(preds, None, mock_chunk, mock_frames)
    results = metric.compute()

    # Assert Frame-Based Results
    frame_result_key = ("body_pose", "mmpose", "jump_detection")
    batch_result = results[frame_result_key][0].results_tensor

    # Expected jumps (6 frames yield 5 comparisons)
    expected_jumps = torch.tensor(
        [[True, False], [False, False], [False, False], [False, True], [False, False]],
        dtype=torch.bool,
    )
    assert torch.equal(batch_result, expected_jumps)

    # Assert Summary Results
    summary_key = ("body_pose", "mmpose", "P1", "cam1", "jump_count")
    assert results[summary_key].value == 2  # Two True values in the tensor


def test_jumpdetection_is_stateful_across_updates(
    mock_keypoints_mapping, mock_keypoint_metadata
):
    """
    Given: A sequence of predictions split across two batches.
    When:  The JumpDetection metric is updated twice.
    Then:  It correctly detects a jump between the last frame of batch 1
           and the first frame of batch 2.
    """
    # Arrange
    metric = JumpDetection(mock_keypoints_mapping)
    mock_chunk, mock_frames = mock_keypoint_metadata

    # Batch 1: Two frames
    preds_batch1 = torch.tensor(
        [
            [[100.0, 0.0, 0.0, 0.95], [0.0, 10.0, 0.0, 0.90]],  # Frame 0
            [[110.0, 0.0, 10.0, 0.92], [0.0, 0.0, 0.0, 0.88]],  # Frame 1 (small move)
        ]
    )

    # Batch 2: Two more frames. The first frame has a big jump from Frame 1
    preds_batch2 = torch.tensor(
        [
            [[310.0, 0.0, 0.0, 0.93], [0.0, 10.0, 0.0, 0.91]],  # Frame 2 (large move)
            [[320.0, 0.0, 0.0, 0.94], [0.0, 0.0, 10.0, 0.89]],  # Frame 3 (small move)
        ]
    )

    # Act
    metric.update(preds_batch1, None, mock_chunk, mock_frames)
    metric.update(preds_batch2, None, mock_chunk, mock_frames)
    results = metric.compute()

    # Assert frame-based results
    # We expect 3 comparisons in total.
    # F1-F0: no jump
    # F2-F1: JUMP on l_shoulder (movement 200.0 > 150.0)
    # F3-F2: no jump
    frame_result_key = ("body_pose", "mmpose", "jump_detection")
    batch_result = results[frame_result_key][1].results_tensor
    expected_jumps = torch.tensor(
        [[True, False], [False, False]], dtype=torch.bool
    )  # Only the first comparison in batch 2 is a jump
    assert torch.equal(batch_result, expected_jumps)

    # Assert summary results
    summary_key = ("body_pose", "mmpose", "P1", "cam1", "jump_count")
    assert results[summary_key].value == 1
