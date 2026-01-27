import numpy as np
import pytest

from nicetoolbox.detectors.method_detectors.body_joints.pose_utils import create_iou_all_pairs


def test_output_shape(mock_output_data):
    S, C, F, _, _ = mock_output_data.shape
    iou = create_iou_all_pairs(mock_output_data)
    assert iou.shape == (S, C, F, S)


def test_symmetry_over_subjects(mock_output_data):
    iou = create_iou_all_pairs(mock_output_data)
    _, C, F, _ = iou.shape

    # For each (C,F), matrix over subjects must be symmetric (i,j) == (j,i)
    for c in range(C):
        for f in range(F):
            sub = iou[:, c, f, :]  # (S, S)
            assert np.allclose(sub, sub.T, equal_nan=True)


def test_diagonal_rules(mock_output_data):
    """
    Diagonal rule:
    - For any present subject (valid bbox with positive area),
      IoU(subject, subject) must be 1.0.
    """
    S, C, F, _, _ = mock_output_data.shape
    data = create_iou_all_pairs(mock_output_data)
    # Check diagonal: (Subject with itself)
    for s in range(S):
        for c in range(C):
            for f in range(F):
                bbox = mock_output_data[s, c, f, 0, :]
                if np.isnan(bbox).any():
                    continue
                diag_val = data[s, c, f, s]
                print(diag_val)
                assert diag_val == pytest.approx(1.0)


def test_iou_no_overlap():
    """
    Generates a known data with
    2 subjects, 1 camera, 1 frame.

    subj0 box: [0, 0, 10, 10]
    subj1 box: [20, 20, 30, 30]

    Non-overlapping boxes -> IoU off-diagonal must be 0.0,
    """

    S, C, F = 2, 1, 1
    data = np.full((S, C, F, 1, 5), np.nan, dtype=np.float32)

    # subject 0
    data[0, 0, 0, 0, :] = [0.0, 0.0, 10.0, 10.0, 1.0]
    # subject 1 far away
    data[1, 0, 0, 0, :] = [20.0, 20.0, 30.0, 30.0, 1.0]

    iou = create_iou_all_pairs(data)

    # diag still 1.0 (two present)
    assert iou[0, 0, 0, 0] == pytest.approx(1.0)
    assert iou[1, 0, 0, 1] == pytest.approx(1.0)

    # off-diagonal IoU = 0.0 (valid pair, no overlap)
    assert iou[0, 0, 0, 1] == pytest.approx(0.0)
    assert iou[1, 0, 0, 0] == pytest.approx(0.0)


def test_iou_known_overlap():
    """
      Generates a known data with
      3 subjects, 1 camera, 1 frame.

    Boxes (all areas = 100):
      s0: [0,  0, 10, 10]
      s1: [5,  0, 15, 10]
      s2: [8,  0, 18, 10]

    Pairwise overlaps:
      s0 vs s1: inter = 5*10 = 50,  union = 100+100-50 = 150  -> IoU = 50/150 = 1/3
      s0 vs s2: inter = 2*10 = 20,  union = 100+100-20 = 180  -> IoU = 20/180 = 1/9
      s1 vs s2: inter = 7*10 = 70,  union = 100+100-70 = 130  -> IoU = 70/130 = 7/13

    Expectations:
      - Diagonal entries (self IoU) = 1.0.
      - Off-diagonals match the values above and are symmetric.
    """
    S, C, F = 3, 1, 1
    data = np.full((S, C, F, 1, 5), np.nan, dtype=np.float32)

    # add subject data
    data[0, 0, 0, 0, :] = [0.0, 0.0, 10.0, 10.0, 1.0]
    data[1, 0, 0, 0, :] = [5.0, 0.0, 15.0, 10.0, 1.0]
    data[2, 0, 0, 0, :] = [8.0, 0.0, 18.0, 10.0, 1.0]

    iou = create_iou_all_pairs(data)  # expected shape: (S, C, F, S)
    assert iou.shape == (3, 1, 1, 3)

    # Diagonal (self IoU = 1.0)
    for s in range(S):
        assert iou[s, 0, 0, s] == pytest.approx(1.0)

    # Off-diagonals
    assert iou[0, 0, 0, 1] == pytest.approx(1 / 3)
    assert iou[1, 0, 0, 0] == pytest.approx(1 / 3)

    assert iou[0, 0, 0, 2] == pytest.approx(1 / 9)
    assert iou[2, 0, 0, 0] == pytest.approx(1 / 9)

    assert iou[1, 0, 0, 2] == pytest.approx(7 / 13)
    assert iou[2, 0, 0, 1] == pytest.approx(7 / 13)
