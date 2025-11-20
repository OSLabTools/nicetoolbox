from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import torch

from nicetoolbox.evaluation.data.dataset import EvaluationDataset
from nicetoolbox.evaluation.data.discovery import ChunkWorkItem

# --- Tests for _apply_reconciliation ---


# --- Tests for _apply_reconciliation ---


@pytest.mark.parametrize(
    "rec_map, expected_values",
    [
        (
            # Scenario 1: Select first and third rows (indices 0 and 2)
            {"axis3": (0, 2)},
            np.array(
                [
                    [0, 1, 2],  # This is the original row 0
                    [6, 7, 8],  # This is the original row 2
                ]
            ),
        ),
        (
            # Scenario 2: Select the second column (index 1)
            {"axis4": (1,)},
            np.array(
                [
                    [1],  # Original col 1 from row 0
                    [4],  # Original col 1 from row 1
                    [7],  # Original col 1 from row 2
                ]
            ),
        ),
        (
            # Scenario 3: Select second row (index 1) AND first/third col (indices 0, 2)
            {"axis3": (1,), "axis4": (0, 2)},
            np.array(
                [
                    [3, 5]  # This is row 1, sliced to keep only columns 0 and 2
                ]
            ),
        ),
        (
            # Scenario 4: No reconciliation (empty map)
            {},
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        ),
    ],
)
def test_apply_reconciliation_slices_correct_indices(rec_map, expected_values):
    """
    Tests that _apply_reconciliation correctly slices an array by checking
    that the CONTENT of the output array is correct, not just the shape.
    """
    # Arrange
    data = np.arange(9).reshape(3, 3)
    # data will be:
    # [[0, 1, 2],  <- keypoint 0
    #  [3, 4, 5],  <- keypoint 1
    #  [6, 7, 8]]  <- keypoint 2

    # Act
    reconciled_data = EvaluationDataset._apply_reconciliation(data, rec_map)

    # Assert
    assert reconciled_data.shape == expected_values.shape
    assert np.array_equal(reconciled_data, expected_values)


# --- Tests for __iter__ (now using fixtures and parametrized) ---


@pytest.mark.parametrize(
    "pred_rec_map, gt_rec_map, expected_pred_shape, expected_gt_shape",
    [
        # No reconciliation, should return full shapes
        ({}, {}, (3, 3), (4, 3)),
        # Reconcile axis3 only
        ({"axis3": (0, 2)}, {"axis3": (1, 3)}, (2, 3), (2, 3)),
        # Reconcile axis4 only
        ({"axis4": (0, 2)}, {"axis4": (0, 2)}, (3, 2), (4, 2)),
        # Reconcile both axes
        (
            {"axis3": (1,), "axis4": (1,)},
            {"axis3": (0, 2), "axis4": (1,)},
            (1, 1),
            (2, 1),
        ),
    ],
)
def test_dataset_iteration_flow(
    mock_pred_loader,
    mock_annot_loader,
    mock_chunk_factory,
    pred_rec_map,
    gt_rec_map,
    expected_pred_shape,
    expected_gt_shape,
):
    """
    Tests the main iteration logic with different reconciliation scenarios.
    """
    # Arrange
    raw_preds = np.arange(2 * 2 * 2 * 3 * 3).reshape(2, 2, 2, 3, 3)
    raw_gts = np.arange(2 * 2 * 2 * 4 * 3).reshape(2, 2, 2, 4, 3) * -1
    mock_pred_loader.load_full_array.return_value = raw_preds
    mock_annot_loader.load_full_array.return_value = raw_gts

    chunk = mock_chunk_factory(
        pred_reconciliation_map=pred_rec_map, gt_reconciliation_map=gt_rec_map
    )
    dataset = EvaluationDataset([chunk], mock_pred_loader, mock_annot_loader)

    # Act
    results = list(dataset)

    # Assert
    assert len(results) == 2  # Two frames in the mock chunk
    pred_data, gt_data, _, _ = results[0]

    assert pred_data.shape == expected_pred_shape
    assert gt_data.shape == expected_gt_shape


# --- Tests for collate_fn ---


def test_collate_fn_groups_items_into_homogeneous_batches(mock_chunk_factory):
    """
    Given: A batch of items with different metric_types and prediction shapes.
    When:  collate_fn is called.
    Then:  It should correctly group the items into separate, homogeneous sub-batches.
    """
    # Arrange: Create two distinct chunk types using our factory
    chunk_A = mock_chunk_factory(metric_type="type_A", component="comp1", session="S1")
    chunk_B = mock_chunk_factory(metric_type="type_B", component="comp2", session="S1")

    # Create a batch containing items from both chunk types
    batch = [
        # Three items for Group A
        (np.ones((3, 2)), np.ones((3, 2)), chunk_A, MagicMock()),
        (np.ones((3, 2)), np.ones((3, 2)), chunk_A, MagicMock()),
        (np.ones((3, 2)), np.ones((3, 2)), chunk_A, MagicMock()),
        # One item for Group B
        (np.zeros(5), np.zeros(5), chunk_B, MagicMock()),
    ]

    # Act
    collated_batch = EvaluationDataset.collate_fn(batch)

    # Assert
    # 1. Assert that exactly two groups were created
    assert len(collated_batch) == 2

    # 2. Define the expected keys for each group
    # Key = (metric_type, pred_shape, component, algorithm, pred_data_key)
    key_A = ("type_A", (3, 2), "S1", "comp1", "algo1", "3d")
    key_B = ("type_B", (5,), "S1", "comp2", "algo1", "3d")  # The shape is different

    assert key_A in collated_batch
    assert key_B in collated_batch

    # 3. Assert the content and shape of the first group
    group_A = collated_batch[key_A]
    assert isinstance(group_A["pred"], torch.Tensor)
    assert group_A["pred"].shape == (3, 3, 2)  # 3 items, each with shape (3, 2)
    assert group_A["gt"].shape == (3, 3, 2)
    assert group_A["chunk"] == chunk_A
    assert len(group_A["frames"]) == 3

    # 4. Assert the content and shape of the second group
    group_B = collated_batch[key_B]
    assert group_B["pred"].shape == (1, 5)  # 1 item, with shape (5,)
    assert group_B["gt"].shape == (1, 5)
    assert group_B["chunk"] == chunk_B
    assert len(group_B["frames"]) == 1


def test_collate_fn_handles_batches_without_ground_truth(mock_chunk_factory):
    """
    Given: A batch where all ground truth (gt) items are None.
    When:  collate_fn is called.
    Then:  It should correctly create a batch where the 'gt' tensor is also None.
    """
    # Arrange
    chunk = mock_chunk_factory()
    batch = [
        (np.ones(3), None, chunk, MagicMock()),
        (np.ones(3), None, chunk, MagicMock()),
    ]

    # Act
    collated_batch = EvaluationDataset.collate_fn(batch)

    # Assert
    assert len(collated_batch) == 1

    # Get the single group from the dictionary
    group = next(iter(collated_batch.values()))

    assert group["pred"] is not None
    assert group["pred"].shape == (2, 3)
    # The crucial check: the 'gt' key in the final batch should be None
    assert group["gt"] is None


def test_collate_fn_raises_error_for_inconsistent_chunks(mock_chunk_factory):
    """
    Given: A batch where items have the same grouping key but originate from
           different ChunkWorkItem objects.
    When:  collate_fn is called.
    Then:  It should raise a ValueError to prevent data corruption.
    """
    # Arrange
    # The key properties for the compound_key are identical
    key_properties = {
        "metric_type": "A",
        "session": "S1",
        "component": "c1",
        "algorithm": "algo1",
        "pred_data_key": "3d",
    }

    # Create two DISTINCT mock objects that share these properties
    chunk_obj1 = mock_chunk_factory(**key_properties)
    chunk_obj2 = create_autospec(ChunkWorkItem, instance=True, **key_properties)

    # We must ensure they are different objects, even if the factory might cache
    assert chunk_obj1 is not chunk_obj2

    # Create a batch where both items will fall into the same group
    batch = [
        (np.ones((3, 2)), None, chunk_obj1, MagicMock()),
        (np.ones((3, 2)), None, chunk_obj2, MagicMock()),  # Inconsistent chunk object
    ]

    # Act & Assert
    with pytest.raises(ValueError, match="Inconsistent chunks in collate_fn grouping"):
        EvaluationDataset.collate_fn(batch)
