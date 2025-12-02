from unittest.mock import MagicMock, create_autospec

import pytest
import torch

from nicetoolbox.configs.schemas.evaluation_config import EvaluationMetricType
from nicetoolbox.evaluation.data.discovery import ChunkWorkItem, FrameInfo
from nicetoolbox.evaluation.metrics.point_cloud import Jpe, PointCloudMetric
from nicetoolbox.evaluation.metrics.results_schema import BatchResult


@pytest.fixture
def jpe_metric():
    """Provides a fresh instance of the Jpe metric for each test."""
    return Jpe()


@pytest.fixture
def mock_jpe_metadata():
    """
    Provides a standard set of mock metadata for a JPE calculation.
    Using a fixture means we don't have to redefine these mocks in every test.
    """
    # We use MagicMock to create fake objects that behave like real metadata objects.
    # 'spec=...' ensures that if we try to access an attribute that doesn't
    # exist on the real class, the mock will raise an error.
    mock_chunk = MagicMock(spec=ChunkWorkItem)
    mock_chunk.component = "body_joints"
    mock_chunk.algorithm = "vitpose"
    mock_chunk.pred_data_description_axis3 = ["nose", "left_eye", "right_eye"]
    # For this metadata, we assume no reconciliation has happened.
    mock_chunk.pred_reconciliation_map = {}

    mock_frames = [
        MagicMock(spec=FrameInfo),
        MagicMock(spec=FrameInfo),
    ]  # A batch of 2 frames

    return mock_chunk, mock_frames


# --- (1) Tests for the PointCloudMetric Handler ---


def test_point_cloud_handler_creates_jpe_metric_if_configured():
    """
    Given: A config specifying the 'jpe' metric.
    When:  The PointCloudMetric handler is initialized.
    Then:  It should create an instance of the Jpe class.
    """
    # Arrange
    mock_config = create_autospec(EvaluationMetricType, instance=True)
    mock_config.metric_names = ["jpe"]

    # Act
    handler = PointCloudMetric(cfg=mock_config, device="cpu")

    # Assert
    assert "jpe" in handler.metrics
    assert isinstance(handler.metrics["jpe"], Jpe)
    assert len(handler.metrics) == 1


def test_point_cloud_handler_ignores_unknown_metrics():
    """
    Given: A config specifying an unknown metric.
    When:  The PointCloudMetric handler is initialized.
    Then:  It should not create any metric instances.
    """
    # Arrange
    mock_config = create_autospec(EvaluationMetricType, instance=True)
    mock_config.metric_names = ["some_unknown_metric"]

    # Act
    handler = PointCloudMetric(cfg=mock_config, device="cpu")

    # Assert
    assert not handler.metrics  # The metrics dictionary should be empty


# --- (2) Tests for Jpe metric ---


def test_jpe_reset(jpe_metric, mock_jpe_metadata):
    """
    Given: A Jpe metric with some data in its internal storage.
    When:  The reset() method is called.
    Then:  The internal storage is cleared.
    """
    # Arrange: Add some dummy data to the metric's storage.
    mock_chunk, mock_frames = mock_jpe_metadata
    preds = torch.ones((2, 3, 3))
    gts = torch.ones((2, 3, 3))
    jpe_metric.update(preds, gts, mock_chunk, mock_frames)
    assert jpe_metric.storage  # Verify storage is not empty

    # Act
    jpe_metric.reset()

    # Assert
    assert not jpe_metric.storage  # Verify storage is now empty


@pytest.mark.parametrize(
    "pred_axis3, reconciliation_map, expected",
    [
        # No reconciliation, should return all keypoints
        (["nose", "left_eye", "right_eye"], {}, ["nose", "left_eye", "right_eye"]),
        # Reconciliation selects only 'nose' and 'right_eye'
        (["nose", "left_eye", "right_eye"], {"axis3": (0, 2)}, ["nose", "right_eye"]),
        # Reconciliation selects only 'left_eye'
        (["nose", "left_eye", "right_eye"], {"axis3": (1,)}, ["left_eye"]),
        # Reconciliation selects none (edge case)
        (["nose", "left_eye", "right_eye"], {"axis3": ()}, []),
    ],
)
def test_jpe_get_axis3(pred_axis3, reconciliation_map, expected):
    """
    Given: A chunk where keypoints have been reconciled (some were selected).
    When:  get_axis3() is called.
    Then:  It should return only the names of the selected keypoints.
    """
    # Arrange
    metric = Jpe()
    mock_chunk = MagicMock(spec=ChunkWorkItem)
    mock_chunk.pred_data_description_axis3 = pred_axis3
    mock_chunk.pred_reconciliation_map = reconciliation_map

    # Act
    axis3_description = metric.get_axis3(mock_chunk)

    # Assert
    assert axis3_description == expected


@pytest.mark.parametrize(
    "preds, gts, expected_errors",
    [
        (
            # Predictions (1 frame, 3 joints, 3D coords)
            torch.tensor([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [1.0, 0.0, 0.0]]]),
            # Ground Truth (1 frame, 3 joints, 3D coords)
            torch.tensor([[[3.0, 0.0, 4.0], [5.0, 5.0, 5.0], [2.0, 0.0, 0.0]]]),
            # Expected Errors (L2 norm per joint)
            torch.tensor([[5.0, 0.0, 1.0]]),
        ),
        (
            # Predictions (1 frame, 2 joints, 2D coords)
            torch.tensor([[[0.0, 0.0], [10.0, 10.0]]]),
            # Ground Truth (1 frame, 2 joints, 2D coords)
            torch.tensor([[[3.0, 4.0], [10.0, 5.0]]]),
            # Expected Errors (L2 norm per joint)
            torch.tensor([[5.0, 5.0]]),
        ),
    ],
)
def test_jpe_update_calculates_correct_error(
    jpe_metric, mock_jpe_metadata, preds, gts, expected_errors
):
    """
    Given: Prediction and Ground Truth tensors (both 2D and 3D).
    When:  The update() method is called.
    Then:  The calculated Euclidean error should be correct.
    """
    # Arrange
    mock_chunk, _ = mock_jpe_metadata
    # Adjust metadata to match the shape of the test data
    num_frames = preds.shape[0]
    mock_frames = [create_autospec(FrameInfo, instance=True) for _ in range(num_frames)]

    # Act
    jpe_metric.update(preds, gts, mock_chunk, mock_frames)

    # Assert
    result_key = ("body_joints", "vitpose", "jpe")
    assert result_key in jpe_metric.storage
    assert len(jpe_metric.storage[result_key]) == 1

    batch_result = jpe_metric.storage[result_key][0]
    assert isinstance(batch_result, BatchResult)
    assert torch.allclose(batch_result.results_tensor, expected_errors)
    assert batch_result.results_description == ["nose", "left_eye", "right_eye"]


def test_jpe_update_does_nothing_if_gt_is_none(jpe_metric, mock_jpe_metadata):
    """
    Given: A metric that relies on Ground Truth.
    When:  update() is called with gts=None.
    Then:  The metric should do nothing and its storage should remain empty.
    """
    # Arrange
    mock_chunk, mock_frames = mock_jpe_metadata
    preds = torch.ones((2, 3, 3))

    # Act
    jpe_metric.update(preds, gts=None, meta_chunk=mock_chunk, meta_frames=mock_frames)

    # Assert
    assert not jpe_metric.storage


def test_jpe_compute_returns_stored_results(jpe_metric, mock_jpe_metadata):
    """
    Given: A metric with data in its storage.
    When:  compute() is called.
    Then:  It should return the entire storage dictionary.
    """
    # Arrange
    mock_chunk, mock_frames = mock_jpe_metadata
    preds = torch.ones((2, 3, 3))
    gts = torch.zeros((2, 3, 3))
    jpe_metric.update(preds, gts, mock_chunk, mock_frames)

    # Act
    computed_results = jpe_metric.compute()

    # Assert
    assert computed_results is jpe_metric.storage  # Check they are the same object
    assert ("body_joints", "vitpose", "jpe") in computed_results


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA-enabled GPU")
def test_jpe_result_format_and_device(jpe_metric, mock_jpe_metadata):
    """
    Given: Input tensors on a GPU.
    When:  The metric is updated and computed.
    Then:  The final result object should have the correct format and its
           data tensor must be on the CPU.
    """
    # Arrange
    metric = jpe_metric
    mock_chunk, mock_frames = mock_jpe_metadata

    # Create tensors and move them to the GPU
    preds = torch.ones((2, 3, 3), device="cuda:0")
    gts = torch.zeros((2, 3, 3), device="cuda:0")

    # Act
    metric.update(preds, gts, mock_chunk, mock_frames)
    computed_results = metric.compute()

    # Assert
    result_key = ("body_joints", "vitpose", "jpe")
    result_list = computed_results[result_key]
    batch_result = result_list[0]
    assert isinstance(batch_result, BatchResult)
    assert isinstance(batch_result.results_tensor, torch.Tensor)
    device = batch_result.results_tensor.device.type
    assert device == "cpu", f"Result tensor was on {device}, but should be on CPU!"
