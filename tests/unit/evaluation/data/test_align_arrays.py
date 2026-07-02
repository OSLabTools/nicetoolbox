import logging
from pathlib import Path

import numpy as np
import pytest

from nicetoolbox.evaluation.data.input_loader import (
    AnnotationMeta,
    ExperimentMeta,
    PathMeta,
    SubsequenceInfo,
    align_arrays,
)
from tests.unit.evaluation.data.conftest import make_loaded_array

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_PATH = Path("dummy.npz")
_DUMMY_KEY = "k"


def _exp(dataset="ds1", session="s01", sequence="seq01", component="body") -> ExperimentMeta:
    return ExperimentMeta(
        dataset=dataset,
        session=session,
        sequence=sequence,
        component=component,
        algorithm="algo",
        subsequence=SubsequenceInfo(0, 0, 100),
        npz_path=_DUMMY_PATH,
        npz_key=_DUMMY_KEY,
    )


def _ann(dataset="ds1", session="s01", sequence="seq01", component="body") -> AnnotationMeta:
    return AnnotationMeta(
        dataset=dataset,
        session=session,
        sequence=sequence,
        component=component,
        npz_path=_DUMMY_PATH,
        npz_key=_DUMMY_KEY,
    )


def _path() -> PathMeta:
    return PathMeta(npz_path=_DUMMY_PATH, npz_key=_DUMMY_KEY)


# ---------------------------------------------------------------------------
# Basic pairing
# ---------------------------------------------------------------------------


class TestBasicPairing:
    def test_matching_key_produces_one_pair(self):
        pred = make_loaded_array(_exp())
        gt = make_loaded_array(_ann())

        result = align_arrays([pred], [gt])

        assert len(result) == 1

    def test_returned_metas_are_original(self):
        pred = make_loaded_array(_exp())
        gt = make_loaded_array(_ann())

        ((aligned_pred, aligned_gt),) = align_arrays([pred], [gt])

        assert aligned_pred.meta is pred.meta
        assert aligned_gt.meta is gt.meta

    def test_key_mismatch_produces_no_pairs(self, caplog):
        pred = make_loaded_array(_exp(dataset="ds1"))
        gt = make_loaded_array(_ann(dataset="ds2"))

        with caplog.at_level(logging.WARNING):
            result = align_arrays([pred], [gt])

        assert result == []
        assert "No ground truth" in caplog.text

    def test_multiple_components_pair_independently(self):
        pred_a = make_loaded_array(_exp(component="body"))
        pred_b = make_loaded_array(_exp(component="hand"))
        gt_a = make_loaded_array(_ann(component="body"))
        gt_b = make_loaded_array(_ann(component="hand"))

        result = align_arrays([pred_a, pred_b], [gt_a, gt_b])

        assert len(result) == 2

    def test_multiple_algorithms_each_pair_with_gt(self):
        pred_a = make_loaded_array(
            ExperimentMeta(
                dataset="ds1",
                session="s01",
                sequence="seq01",
                component="body",
                algorithm="algo_a",
                subsequence=SubsequenceInfo(0, 0, 100),
                npz_path=_DUMMY_PATH,
                npz_key=_DUMMY_KEY,
            )
        )
        pred_b = make_loaded_array(
            ExperimentMeta(
                dataset="ds1",
                session="s01",
                sequence="seq01",
                component="body",
                algorithm="algo_b",
                subsequence=SubsequenceInfo(0, 0, 100),
                npz_path=_DUMMY_PATH,
                npz_key=_DUMMY_KEY,
            )
        )
        gt = make_loaded_array(_ann())

        result = align_arrays([pred_a, pred_b], [gt])

        assert len(result) == 2
        assert all(aligned_gt.meta is gt.meta for _, aligned_gt in result)

    def test_multiple_datasets_pair_independently(self):
        pred1 = make_loaded_array(_exp(dataset="ds1"))
        pred2 = make_loaded_array(_exp(dataset="ds2"))
        gt1 = make_loaded_array(_ann(dataset="ds1"))
        gt2 = make_loaded_array(_ann(dataset="ds2"))

        result = align_arrays([pred1, pred2], [gt1, gt2])

        assert len(result) == 2
        paired_gts = {id(aligned_gt.meta) for _, aligned_gt in result}
        assert id(gt1.meta) in paired_gts
        assert id(gt2.meta) in paired_gts

    def test_empty_predictions_returns_empty(self):
        gt = make_loaded_array(_ann())

        result = align_arrays([], [gt])

        assert result == []

    def test_empty_ground_truth_returns_empty(self, caplog):
        pred = make_loaded_array(_exp())

        with caplog.at_level(logging.WARNING):
            result = align_arrays([pred], [])

        assert result == []
        assert "No ground truth" in caplog.text

    def test_both_empty_returns_empty(self):
        assert align_arrays([], []) == []

    def test_partial_key_mismatch_produces_no_pairs(self, caplog):
        """All four key fields must match — same dataset but different session is not a pair."""
        pred = make_loaded_array(_exp(dataset="ds1", session="s01"))
        gt = make_loaded_array(_ann(dataset="ds1", session="s02"))

        with caplog.at_level(logging.WARNING):
            result = align_arrays([pred], [gt])

        assert result == []
        assert "No ground truth" in caplog.text

    def test_unmatched_gt_is_silently_ignored(self, caplog):
        """A GT with no matching pred produces no warning and no error."""
        pred = make_loaded_array(_exp(dataset="ds1"))
        gt_matched = make_loaded_array(_ann(dataset="ds1"))
        gt_unmatched = make_loaded_array(_ann(dataset="ds2"))

        with caplog.at_level(logging.WARNING):
            result = align_arrays([pred], [gt_matched, gt_unmatched])

        assert len(result) == 1
        assert "No ground truth" not in caplog.text


# ---------------------------------------------------------------------------
# Cross-meta pairing
# ---------------------------------------------------------------------------


class TestCrossMetaPairing:
    def test_annotation_pred_with_experiment_gt_pairs(self):
        """align_key is symmetric — AnnotationMeta pred can pair with ExperimentMeta GT."""
        pred = make_loaded_array(_ann())
        gt = make_loaded_array(_exp())

        result = align_arrays([pred], [gt])

        assert len(result) == 1

    def test_experiment_pred_with_annotation_gt_pairs(self):
        """Standard case: ExperimentMeta pred pairs with AnnotationMeta GT."""
        pred = make_loaded_array(_exp())
        gt = make_loaded_array(_ann())

        result = align_arrays([pred], [gt])

        assert len(result) == 1

    def test_path_pred_with_structured_gt_produces_no_pair(self, caplog):
        """PathMeta pred has no align_key — cannot match structured GT, no wildcard available."""
        pred = make_loaded_array(_path())
        gt = make_loaded_array(_ann())

        with caplog.at_level(logging.WARNING):
            result = align_arrays([pred], [gt])

        assert result == []

    def test_experiment_pred_with_path_gt_pairs_via_wildcard(self):
        """PathMeta GT acts as wildcard — pairs with any structured pred that has no better match."""
        pred = make_loaded_array(_exp())
        gt = make_loaded_array(_path())

        result = align_arrays([pred], [gt])

        assert len(result) == 1

    def test_structured_gt_takes_priority_over_wildcard(self):
        """When both structured and PathMeta GTs are present, structured match wins."""
        pred = make_loaded_array(_exp())
        structured_gt = make_loaded_array(_ann())
        wildcard_gt = make_loaded_array(_path())

        ((_, aligned_gt),) = align_arrays([pred], [structured_gt, wildcard_gt])

        assert aligned_gt.meta is structured_gt.meta

    def test_wildcard_used_when_no_structured_match(self):
        """Pred with no structured GT match falls back to PathMeta wildcard."""
        pred = make_loaded_array(_exp(dataset="ds_unknown"))
        structured_gt = make_loaded_array(_ann(dataset="ds_other"))
        wildcard_gt = make_loaded_array(_path())

        result = align_arrays([pred], [structured_gt, wildcard_gt])

        assert len(result) == 1
        _, aligned_gt = result[0]
        assert aligned_gt.meta is wildcard_gt.meta

    def test_path_pred_with_path_gt_pairs(self):
        pred = make_loaded_array(_path())
        gt = make_loaded_array(_path())

        result = align_arrays([pred], [gt])

        assert len(result) == 1

    def test_mixed_preds_both_pair_with_path_wildcard(self):
        """PathMeta pred and ExperimentMeta pred both fall back to the same PathMeta wildcard."""
        pred_path = make_loaded_array(_path())
        pred_exp = make_loaded_array(_exp())
        gt_wildcard = make_loaded_array(_path())

        result = align_arrays([pred_path, pred_exp], [gt_wildcard])

        assert len(result) == 2
        assert all(aligned_gt.meta is gt_wildcard.meta for _, aligned_gt in result)


# ---------------------------------------------------------------------------
# PathMeta validation
# ---------------------------------------------------------------------------


class TestPathMetaValidation:
    def test_multiple_path_gts_raises(self):
        pred = make_loaded_array(_path())
        gt1 = make_loaded_array(_path())
        gt2 = make_loaded_array(_path())

        with pytest.raises(ValueError, match="path-based"):
            align_arrays([pred], [gt1, gt2])

    def test_multiple_path_preds_raises(self):
        pred1 = make_loaded_array(_path())
        pred2 = make_loaded_array(_path())
        gt = make_loaded_array(_path())

        with pytest.raises(ValueError, match="path-based"):
            align_arrays([pred1, pred2], [gt])

    def test_multiple_path_gts_raises_even_with_structured_preds(self):
        """Two PathMeta GTs are ambiguous regardless of pred type — error message reports 0 PathMeta preds."""
        pred1 = make_loaded_array(_exp(dataset="ds1"))
        pred2 = make_loaded_array(_exp(dataset="ds2"))
        gt1 = make_loaded_array(_path())
        gt2 = make_loaded_array(_path())

        with pytest.raises(ValueError, match="path-based"):
            align_arrays([pred1, pred2], [gt1, gt2])


# ---------------------------------------------------------------------------
# Axis intersection
# ---------------------------------------------------------------------------


class TestAxisIntersection:
    def test_common_subjects_kept(self):
        pred = make_loaded_array(_exp(), subjects=("s1", "s2"))
        gt = make_loaded_array(_ann(), subjects=("s2", "s3"))

        ((aligned_pred, aligned_gt),) = align_arrays([pred], [gt])

        assert aligned_pred.axes.subjects == ["s2"]
        assert aligned_gt.axes.subjects == ["s2"]

    def test_common_labels_kept(self):
        pred = make_loaded_array(_exp(), labels=("x", "y", "z"))
        gt = make_loaded_array(_ann(), labels=("x", "z", "w"))

        ((aligned_pred, aligned_gt),) = align_arrays([pred], [gt])

        assert aligned_pred.axes.labels == ["x", "z"]
        assert aligned_gt.axes.labels == ["x", "z"]

    def test_common_cameras_kept(self):
        pred = make_loaded_array(_exp(), cameras=("cam1", "cam2"))
        gt = make_loaded_array(_ann(), cameras=("cam2", "cam3"))

        ((aligned_pred, _),) = align_arrays([pred], [gt])

        assert aligned_pred.axes.cameras == ["cam2"]

    def test_common_frames_kept(self):
        pred = make_loaded_array(_exp(), frames=("f0", "f1", "f2"))
        gt = make_loaded_array(_ann(), frames=("f1", "f2", "f3"))

        ((aligned_pred, _),) = align_arrays([pred], [gt])

        assert aligned_pred.axes.frames == ["f1", "f2"]

    def test_intersection_order_follows_pred(self):
        """Label order in the result follows pred's order, not GT's."""
        pred = make_loaded_array(_exp(), labels=("z", "x", "y"))
        gt = make_loaded_array(_ann(), labels=("x", "y", "z"))

        ((aligned_pred, _),) = align_arrays([pred], [gt])

        assert aligned_pred.axes.labels == ["z", "x", "y"]

    def test_data_trimmed_to_intersection(self):
        pred = make_loaded_array(_exp(), subjects=("s1", "s2"), labels=("x", "y", "z"))
        gt = make_loaded_array(_ann(), subjects=("s2",), labels=("x", "z"))

        ((aligned_pred, aligned_gt),) = align_arrays([pred], [gt])

        assert aligned_pred.data.shape == (1, 1, 1, 2)
        assert aligned_gt.data.shape == (1, 1, 1, 2)

    def test_pred_data_values_correct_after_intersection(self):
        """Sequential integer data lets us verify the right slice was selected."""
        pred = make_loaded_array(_exp(), subjects=("s1", "s2"))
        gt = make_loaded_array(_ann(), subjects=("s2",))

        ((aligned_pred, _),) = align_arrays([pred], [gt])

        # pred data is arange(2).reshape(2,1,1,1): s1→0, s2→1
        # after intersecting to s2 (index 1 in pred): value 1
        np.testing.assert_array_equal(aligned_pred.data, pred.data[1:2])

    def test_gt_data_values_correct_after_intersection(self):
        """GT data is sliced to the common labels, not just pred."""
        pred = make_loaded_array(_exp(), subjects=("s2",))
        gt = make_loaded_array(_ann(), subjects=("s1", "s2"))

        ((_, aligned_gt),) = align_arrays([pred], [gt])

        # gt data is arange(2).reshape(2,1,1,1): s1→0, s2→1
        # after intersecting to s2 (index 1 in gt): value 1
        np.testing.assert_array_equal(aligned_gt.data, gt.data[1:2])

    def test_pred_and_gt_correspond_to_same_labels(self):
        """Core alignment guarantee: position i in pred and GT refer to the same label."""
        pred = make_loaded_array(_exp(), subjects=("s1", "s2", "s3"), labels=("x", "y", "z"))
        gt = make_loaded_array(_ann(), subjects=("s3", "s2"), labels=("z", "x"))

        ((aligned_pred, aligned_gt),) = align_arrays([pred], [gt])

        # Common subjects in pred order: s2, s3. Common labels in pred order: x, z.
        assert aligned_pred.axes.subjects == ["s2", "s3"]
        assert aligned_pred.axes.labels == ["x", "z"]
        assert aligned_gt.axes.subjects == ["s2", "s3"]
        assert aligned_gt.axes.labels == ["x", "z"]

        # pred data: arange(3*1*1*3).reshape(3,1,1,3)
        # s2=index1, s3=index2; x=index0, z=index2
        np.testing.assert_array_equal(aligned_pred.data, pred.data[[1, 2], :, :, :][:, :, :, [0, 2]])

        # gt data: arange(2*1*1*2).reshape(2,1,1,2)
        # s2=index1, s3=index0; x=index1, z=index0
        np.testing.assert_array_equal(aligned_gt.data, gt.data[[1, 0], :, :, :][:, :, :, [1, 0]])

    def test_data_intersection_does_not_skip_pair(self):
        """Empty data axis intersection is not a failure — data axis is optional."""
        pred = make_loaded_array(_exp(), data=("u",))
        gt = make_loaded_array(_ann(), data=("v",))

        result = align_arrays([pred], [gt])

        assert len(result) == 1
        aligned_pred, _ = result[0]
        assert aligned_pred.axes.data == []

    def test_both_axes_have_all_labels_in_common(self):
        """Identical axis labels → no trimming, data unchanged."""
        pred = make_loaded_array(_exp(), subjects=("s1", "s2"), labels=("x", "y"))
        gt = make_loaded_array(_ann(), subjects=("s1", "s2"), labels=("x", "y"))

        ((aligned_pred, aligned_gt),) = align_arrays([pred], [gt])

        np.testing.assert_array_equal(aligned_pred.data, pred.data)
        np.testing.assert_array_equal(aligned_gt.data, gt.data)

    def test_shared_axes_object_after_alignment(self):
        """Both aligned arrays should reference the same axes object."""
        pred = make_loaded_array(_exp(), subjects=("s1", "s2"))
        gt = make_loaded_array(_ann(), subjects=("s1", "s2"))

        ((aligned_pred, aligned_gt),) = align_arrays([pred], [gt])

        assert aligned_pred.axes is aligned_gt.axes


# ---------------------------------------------------------------------------
# Empty intersection — pair skipped
# ---------------------------------------------------------------------------


class TestEmptyIntersection:
    @pytest.mark.parametrize(
        "axis,pred_kw,gt_kw",
        [
            ("subjects", {"subjects": ("s1",)}, {"subjects": ("s2",)}),
            ("cameras", {"cameras": ("c1",)}, {"cameras": ("c2",)}),
            ("frames", {"frames": ("f0",)}, {"frames": ("f1",)}),
            ("labels", {"labels": ("x",)}, {"labels": ("y",)}),
        ],
    )
    def test_empty_required_axis_skips_pair(self, axis, pred_kw, gt_kw, caplog):
        pred = make_loaded_array(_exp(), **pred_kw)
        gt = make_loaded_array(_ann(), **gt_kw)

        with caplog.at_level(logging.WARNING):
            result = align_arrays([pred], [gt])

        assert result == []
        assert axis in caplog.text


# ---------------------------------------------------------------------------
# Broadcast single
# ---------------------------------------------------------------------------


class TestBroadcastSingle:
    def test_single_camera_mismatch_paired_with_broadcast(self):
        pred = make_loaded_array(_exp(), cameras=("pred_cam",))
        gt = make_loaded_array(_ann(), cameras=("gt_cam",))

        result = align_arrays([pred], [gt], broadcast_single=True)

        assert len(result) == 1

    def test_broadcast_keeps_pred_label(self):
        pred = make_loaded_array(_exp(), cameras=("pred_cam",))
        gt = make_loaded_array(_ann(), cameras=("gt_cam",))

        ((aligned_pred, _),) = align_arrays([pred], [gt], broadcast_single=True)

        assert aligned_pred.axes.cameras == ["pred_cam"]

    def test_no_broadcast_single_mismatch_skips_pair(self, caplog):
        pred = make_loaded_array(_exp(), cameras=("pred_cam",))
        gt = make_loaded_array(_ann(), cameras=("gt_cam",))

        with caplog.at_level(logging.WARNING):
            result = align_arrays([pred], [gt], broadcast_single=False)

        assert result == []

    def test_broadcast_does_not_apply_when_multiple_labels(self, caplog):
        """broadcast_single only fires when both sides have exactly 1 label — not for larger axes."""
        pred = make_loaded_array(_exp(), cameras=("c1", "c2"))
        gt = make_loaded_array(_ann(), cameras=("c3", "c4"))

        with caplog.at_level(logging.WARNING):
            result = align_arrays([pred], [gt], broadcast_single=True)

        assert result == []

    def test_broadcast_does_not_apply_when_only_pred_has_one(self, caplog):
        """Both sides must have exactly 1 label — pred=1, GT=2 should not broadcast."""
        pred = make_loaded_array(_exp(), cameras=("c1",))
        gt = make_loaded_array(_ann(), cameras=("c2", "c3"))

        with caplog.at_level(logging.WARNING):
            result = align_arrays([pred], [gt], broadcast_single=True)

        assert result == []

    def test_broadcast_does_not_apply_when_only_gt_has_one(self, caplog):
        """Both sides must have exactly 1 label — pred=2, GT=1 should not broadcast."""
        pred = make_loaded_array(_exp(), cameras=("c1", "c2"))
        gt = make_loaded_array(_ann(), cameras=("c3",))

        with caplog.at_level(logging.WARNING):
            result = align_arrays([pred], [gt], broadcast_single=True)

        assert result == []

    def test_broadcast_data_unchanged(self):
        """When broadcast fires, data is not sliced — both arrays pass through as-is."""
        pred = make_loaded_array(_exp(), cameras=("pred_cam",))
        gt = make_loaded_array(_ann(), cameras=("gt_cam",))

        ((aligned_pred, aligned_gt),) = align_arrays([pred], [gt], broadcast_single=True)

        np.testing.assert_array_equal(aligned_pred.data, pred.data)
        np.testing.assert_array_equal(aligned_gt.data, gt.data)
