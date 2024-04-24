import pytest
import oslab_utils.image_processing as im

def test_no_intersection():
    box1 = (1, 1, 3, 3)
    box2 = (4, 4, 6, 6)
    expected = 0.0
    assert im.calculate_iou(box1, box2) == expected, "Expected no intersection IoU to be 0.0"

def test_partial_intersection():
    box1 = (1, 1, 5, 5)
    box2 = (3, 3, 7, 7)
    expected = 4 / 28  # Intersection area = 4, Union area = 28
    assert im.calculate_iou(box1, box2) == pytest.approx(expected), f"Expected partial intersection IoU to be approximately {expected}"

def test_almost_full_overlap():
    box1 = (2, 2, 6, 6)
    box2 = (1, 1, 7, 7)
    expected = 16 / 36  # Intersection area = 16, Union area = 36
    assert im.calculate_iou(box1, box2) == pytest.approx(expected), f"Expected almost full overlap IoU to be approximately {expected}"

def test_full_overlap():
    box1 = (2, 2, 6, 6)
    box2 = (2, 2, 6, 6)
    expected = 1.0  # Intersection area = Union area = 16
    assert im.calculate_iou(box1, box2) == expected, "Expected full overlap IoU to be 1.0"
