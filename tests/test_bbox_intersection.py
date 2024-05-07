import pytest

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1 : array_like
        Bounding box, format: [x1, y1, x2, y2] where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate.
    box2 : array_like
        Bounding box, format: [x1, y1, x2, y2] where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate.

    Returns
    -------
    float
        The Intersection over Union (IoU) overlap between the two bounding boxes. This is 0 if there is no overlap.

    Notes
    -----
        IoU = Area of Overlap / Area of Union
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Check if there is no overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of each bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the Intersection over Union
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def test_no_intersection():
    box1 = (1, 1, 3, 3)
    box2 = (4, 4, 6, 6)
    expected = 0.0
    assert calculate_iou(box1, box2) == expected, "Expected no intersection IoU to be 0.0"

def test_partial_intersection():
    box1 = (1, 1, 5, 5)
    box2 = (3, 3, 7, 7)
    expected = 4 / 28  # Intersection area = 4, Union area = 28
    assert calculate_iou(box1, box2) == pytest.approx(expected), f"Expected partial intersection IoU to be approximately {expected}"

def test_almost_full_overlap():
    box1 = (2, 2, 6, 6)
    box2 = (1, 1, 7, 7)
    expected = 16 / 36  # Intersection area = 16, Union area = 36
    assert calculate_iou(box1, box2) == pytest.approx(expected), f"Expected almost full overlap IoU to be approximately {expected}"

def test_full_overlap():
    box1 = (2, 2, 6, 6)
    box2 = (2, 2, 6, 6)
    expected = 1.0  # Intersection area = Union area = 16
    assert calculate_iou(box1, box2) == expected, "Expected full overlap IoU to be 1.0"
