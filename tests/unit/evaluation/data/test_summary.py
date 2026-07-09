from pathlib import Path

import pandas as pd
import pytest

from nicetoolbox.configs.schemas.evaluation_aggr import AggSpec
from nicetoolbox.configs.schemas.evaluation_group_by import GroupBySpec
from nicetoolbox.evaluation.data.input_loader import AnnotationMeta, ExperimentMeta, PathMeta, SubsequenceInfo
from nicetoolbox.evaluation.data.summary import (
    aggregate_summary,
    arrays_to_dataframe,
    pair_arrays_to_df,
    resolve_group_levels,
    summarize_with_group_by,
)
from tests.unit.evaluation.data.conftest import make_loaded_array

_DUMMY_PATH = Path("dummy.npz")
_DUMMY_KEY = "k"


def _exp(algorithm: str = "algo_a") -> ExperimentMeta:
    return ExperimentMeta(
        dataset="ds1",
        sequence="seq01",
        component="body_joints",
        algorithm=algorithm,
        subsequence=SubsequenceInfo(0, 0, 100),
        npz_path=_DUMMY_PATH,
        npz_key=_DUMMY_KEY,
    )


def _ann() -> AnnotationMeta:
    return AnnotationMeta(
        dataset="ds1",
        sequence="seq01",
        component="body_joints",
        npz_path=_DUMMY_PATH,
        npz_key=_DUMMY_KEY,
    )


def _path(stem: str = "result") -> PathMeta:
    return PathMeta(npz_path=Path(f"{stem}.npz"), npz_key=_DUMMY_KEY)


def _at(df, **levels) -> float:
    """Look up the scalar value at the given index level coordinates."""
    return df.xs(tuple(levels.values()), level=list(levels.keys()))["value"].item()


# ---------------------------------------------------------------------------
# Single array: combined shape + coordinate value assertions
# ---------------------------------------------------------------------------


class TestSingleArray:
    def test_minimal_array(self):
        arr = make_loaded_array(_exp())
        df = arrays_to_dataframe([arr])

        assert len(df) == 1
        assert _at(df, subject="s1", camera="c1", frame="f0", label="j1") == 0.0

    def test_multiple_subjects(self):
        arr = make_loaded_array(_exp(), subjects=("s1", "s2"))
        df = arrays_to_dataframe([arr])

        assert len(df) == 2
        assert _at(df, subject="s1", camera="c1", frame="f0", label="j1") == 0.0
        assert _at(df, subject="s2", camera="c1", frame="f0", label="j1") == 1.0

    def test_labels_vary_fastest(self):
        arr = make_loaded_array(_exp(), labels=("j1", "j2", "j3"))
        df = arrays_to_dataframe([arr])

        assert len(df) == 3
        assert _at(df, subject="s1", camera="c1", frame="f0", label="j1") == 0.0
        assert _at(df, subject="s1", camera="c1", frame="f0", label="j2") == 1.0
        assert _at(df, subject="s1", camera="c1", frame="f0", label="j3") == 2.0

    def test_subjects_vary_slowest(self):
        arr = make_loaded_array(_exp(), subjects=("s1", "s2"), labels=("j1", "j2"))
        df = arrays_to_dataframe([arr])

        # C-order: (s1,j1)=0, (s1,j2)=1, (s2,j1)=2, (s2,j2)=3
        assert len(df) == 4
        assert _at(df, subject="s1", camera="c1", frame="f0", label="j1") == 0.0
        assert _at(df, subject="s1", camera="c1", frame="f0", label="j2") == 1.0
        assert _at(df, subject="s2", camera="c1", frame="f0", label="j1") == 2.0
        assert _at(df, subject="s2", camera="c1", frame="f0", label="j2") == 3.0

    def test_all_axes(self):
        arr = make_loaded_array(
            _exp(),
            subjects=("s1", "s2"),
            cameras=("c1", "c2"),
            frames=("f0", "f1", "f2"),
            labels=("j1", "j2", "j3", "j4"),
        )
        df = arrays_to_dataframe([arr])

        # 2 × 2 × 3 × 4 = 48
        assert len(df) == 48
        # s2 block starts at offset 1×(2×3×4)=24; c1,f0,j1 are all first → value=24
        assert _at(df, subject="s2", camera="c1", frame="f0", label="j1") == 24.0


# ---------------------------------------------------------------------------
# MultiIndex structure: level names wired to correct data
# ---------------------------------------------------------------------------


class TestMultiIndexStructure:
    def test_level_names_experiment_meta(self):
        arr = make_loaded_array(_exp())
        df = arrays_to_dataframe([arr])

        expected_names = [
            "dataset",
            "sequence",
            "subsequence",
            "subsequence_start",
            "subsequence_length",
            "component",
            "algorithm",
            "npz_key",
            "subject",
            "camera",
            "frame",
            "label",
        ]
        assert list(df.index.names) == expected_names
        assert _at(df, subject="s1", camera="c1", frame="f0", label="j1") == 0.0

    def test_level_names_annotation_meta(self):
        arr = make_loaded_array(_ann())
        df = arrays_to_dataframe([arr])

        expected_names = [
            "dataset",
            "sequence",
            "component",
            "npz_key",
            "subject",
            "camera",
            "frame",
            "label",
        ]
        assert list(df.index.names) == expected_names
        assert _at(df, subject="s1", camera="c1", frame="f0", label="j1") == 0.0

    def test_level_names_path_meta(self):
        arr = make_loaded_array(_path("my_file"))
        df = arrays_to_dataframe([arr])

        expected_names = ["npz_file_name", "npz_key", "subject", "camera", "frame", "label"]
        assert list(df.index.names) == expected_names
        assert _at(df, subject="s1", camera="c1", frame="f0", label="j1") == 0.0

    def test_meta_values_broadcast_to_all_rows(self):
        arr = make_loaded_array(_exp(), frames=("f0", "f1"), labels=("j1", "j2"))
        df = arrays_to_dataframe([arr])

        assert len(df) == 4
        assert df.index.get_level_values("algorithm").tolist() == ["algo_a"] * 4
        # C-order: (f0,j1)=0, (f0,j2)=1, (f1,j1)=2, (f1,j2)=3
        assert _at(df, subject="s1", camera="c1", frame="f1", label="j2") == 3.0


# ---------------------------------------------------------------------------
# Multiple arrays: concatenation correctness and per-array isolation
# ---------------------------------------------------------------------------


class TestMultipleArrays:
    def test_two_arrays_row_count_and_values(self):
        arr_a = make_loaded_array(_exp("algo_a"))
        arr_b = make_loaded_array(_exp("algo_b"))
        df = arrays_to_dataframe([arr_a, arr_b])

        assert len(df) == 2
        assert _at(df, algorithm="algo_a", subject="s1", camera="c1", frame="f0", label="j1") == 0.0
        assert _at(df, algorithm="algo_b", subject="s1", camera="c1", frame="f0", label="j1") == 0.0

    def test_two_arrays_no_cross_contamination(self):
        arr_a = make_loaded_array(_exp("algo_a"), subjects=("s1", "s2"), labels=("j1", "j2"))
        arr_b = make_loaded_array(_exp("algo_b"), subjects=("s1", "s2"), labels=("j1", "j2"))
        df = arrays_to_dataframe([arr_a, arr_b])

        assert len(df) == 8
        # Each array has independent sequential data; (s2,j2) is index 3 in both
        assert _at(df, algorithm="algo_a", subject="s2", camera="c1", frame="f0", label="j2") == 3.0
        assert _at(df, algorithm="algo_b", subject="s2", camera="c1", frame="f0", label="j2") == 3.0


# ---------------------------------------------------------------------------
# Guards: assertion and error conditions
# ---------------------------------------------------------------------------


class TestGuards:
    @pytest.mark.parametrize("other", ["ann", "path"])
    def test_raises_on_mismatched_meta_keys(self, other):
        arr_exp = make_loaded_array(_exp())
        arr_other = make_loaded_array(_ann() if other == "ann" else _path())

        with pytest.raises(AssertionError):
            arrays_to_dataframe([arr_exp, arr_other])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            arrays_to_dataframe([])


# ---------------------------------------------------------------------------
# aggregate_summary
# ---------------------------------------------------------------------------


class TestAggregateSummary:
    def test_averages_correctly_per_group(self):
        df = pd.DataFrame(
            {
                "component": ["body_joints"] * 4,
                "algorithm": ["algo_a", "algo_a", "algo_b", "algo_b"],
                "npz_key": ["k"] * 4,
                "score": [0.8, 0.6, 0.9, 0.7],
            }
        )
        result = aggregate_summary(df, "score", ExperimentMeta)

        assert len(result) == 2
        assert result.loc[result["algorithm"] == "algo_a", "score"].item() == pytest.approx(0.7)
        assert result.loc[result["algorithm"] == "algo_b", "score"].item() == pytest.approx(0.8)

    def test_partial_always_iterate_columns(self):
        # "npz_key" absent — groups by the two present always_iterate columns only
        df = pd.DataFrame(
            {
                "component": ["body_joints", "body_joints"],
                "algorithm": ["algo_a", "algo_b"],
                "score": [0.5, 0.9],
            }
        )
        result = aggregate_summary(df, "score", ExperimentMeta)

        assert len(result) == 2
        assert "npz_key" not in result.columns

    def test_extra_columns_not_used_for_grouping(self):
        # AnnotationMeta.always_iterate = {"component", "npz_key"} — "algorithm" is not in it
        df = pd.DataFrame(
            {
                "component": ["body_joints", "body_joints"],
                "npz_key": ["k", "k"],
                "algorithm": ["algo_a", "algo_b"],
                "score": [0.4, 0.8],
            }
        )
        result = aggregate_summary(df, "score", AnnotationMeta)

        # Both rows collapse into one group → mean of 0.4 and 0.8
        assert len(result) == 1
        assert result["score"].item() == pytest.approx(0.6)
        assert "algorithm" not in result.columns

    def test_raises_when_no_always_iterate_columns_present(self):
        df = pd.DataFrame({"subject": ["s1", "s2"], "score": [0.5, 0.9]})

        with pytest.raises(ValueError, match="Cannot aggregate summary"):
            aggregate_summary(df, "score", ExperimentMeta)


# ---------------------------------------------------------------------------
# pair_arrays_to_df
# ---------------------------------------------------------------------------


class TestPairArraysToDf:
    def test_pred_and_gt_values_at_correct_coordinates(self):
        pred = make_loaded_array(_exp(), labels=("j1", "j2"))
        gt = make_loaded_array(_ann(), labels=("j1", "j2"))
        df = pair_arrays_to_df([(pred, gt)])

        assert len(df) == 2
        assert df.xs("j1", level="label")["pred"].item() == 0.0
        assert df.xs("j1", level="label")["gt"].item() == 0.0
        assert df.xs("j2", level="label")["pred"].item() == 1.0
        assert df.xs("j2", level="label")["gt"].item() == 1.0

    def test_gt_meta_is_dropped(self):
        # Both ExperimentMeta but different algorithms — only pred's algorithm appears in output
        pred = make_loaded_array(_exp("algo_pred"))
        gt = make_loaded_array(_exp("algo_gt"))
        df = pair_arrays_to_df([(pred, gt)])

        assert len(df) == 1
        assert df.index.get_level_values("algorithm")[0] == "algo_pred"

    def test_value_columns_not_in_index(self):
        pred = make_loaded_array(_exp())
        gt = make_loaded_array(_ann())
        df = pair_arrays_to_df([(pred, gt)])

        assert "pred" in df.columns
        assert "gt" in df.columns
        assert "pred" not in df.index.names
        assert "gt" not in df.index.names

    def test_custom_value_names(self):
        pred = make_loaded_array(_exp())
        gt = make_loaded_array(_ann())
        df = pair_arrays_to_df([(pred, gt)], value_names=("prediction", "ground_truth"))

        assert "prediction" in df.columns
        assert "ground_truth" in df.columns
        assert "pred" not in df.columns
        assert "gt" not in df.columns

    def test_multiple_pairs_concatenated(self):
        pair1 = (make_loaded_array(_exp("algo_a")), make_loaded_array(_ann()))
        pair2 = (make_loaded_array(_exp("algo_b")), make_loaded_array(_ann()))
        df = pair_arrays_to_df([pair1, pair2])

        assert len(df) == 2
        assert set(df.index.get_level_values("algorithm").tolist()) == {"algo_a", "algo_b"}

    def test_empty_pairs_returns_empty_dataframe(self):
        df = pair_arrays_to_df([])

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_raises_on_axis_mismatch(self):
        pred = make_loaded_array(_exp(), labels=("j1", "j2"))
        gt = make_loaded_array(_ann(), labels=("j1", "j3"))  # j3 ≠ j2 → one label unmatched

        with pytest.raises(ValueError, match="Axis merge lost rows"):
            pair_arrays_to_df([(pred, gt)])


# ---------------------------------------------------------------------------
# resolve_group_levels
# ---------------------------------------------------------------------------


def _mi(*names: str) -> pd.DataFrame:
    """Empty DataFrame whose MultiIndex carries the given level names."""
    return pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=list(names)))


class TestResolveGroupLevels:
    def test_always_iterate_included_without_user_dims(self):
        df = _mi("component", "algorithm", "npz_key", "subject")
        result = resolve_group_levels(df, ExperimentMeta, GroupBySpec(dims=[]))

        assert set(result) == {"component", "algorithm", "npz_key"}

    def test_user_dims_appended_after_always_iterate(self):
        df = _mi("component", "algorithm", "npz_key", "subject")
        result = resolve_group_levels(df, ExperimentMeta, GroupBySpec(dims=["subject"]))

        assert set(result) == {"component", "algorithm", "npz_key", "subject"}
        always_positions = [result.index(c) for c in ["component", "algorithm", "npz_key"]]
        assert result.index("subject") > max(always_positions)

    def test_deduplication_when_user_requests_always_iterate_col(self):
        df = _mi("component", "algorithm", "npz_key", "subject")
        result = resolve_group_levels(df, ExperimentMeta, GroupBySpec(dims=["algorithm", "subject"]))

        assert result.count("algorithm") == 1
        assert result.index("algorithm") < result.index("subject")

    def test_frame_excluded_by_default(self):
        df = _mi("component", "algorithm", "npz_key", "frame", "label")
        result = resolve_group_levels(df, ExperimentMeta, GroupBySpec(dims=None))

        assert "frame" not in result

    def test_custom_exclude_removes_always_iterate_col(self):
        df = _mi("component", "algorithm", "npz_key")
        result = resolve_group_levels(df, ExperimentMeta, GroupBySpec(dims=[]), exclude=frozenset({"component"}))

        assert "component" not in result
        assert {"algorithm", "npz_key"}.issubset(result)

    def test_unavailable_always_iterate_cols_not_in_result(self):
        df = _mi("component", "algorithm")  # "npz_key" absent
        result = resolve_group_levels(df, ExperimentMeta, GroupBySpec(dims=[]))

        assert "npz_key" not in result
        assert set(result) == {"component", "algorithm"}

    def test_value_cols_never_included_with_wildcard(self):
        df = _mi("component", "algorithm", "npz_key", "subject", "label")
        result = resolve_group_levels(df, ExperimentMeta, GroupBySpec(dims=None))

        assert "pred" not in result
        assert "gt" not in result


# ---------------------------------------------------------------------------
# summarize_with_group_by
# ---------------------------------------------------------------------------


class TestSummarizeWithGroupBy:
    def test_empty_arrays_returns_empty_dataframe(self):
        result = summarize_with_group_by([], GroupBySpec(dims=[]), AggSpec.of_type("mean"))

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_aggregates_over_frames_when_pooled(self):
        arr = make_loaded_array(_exp(), frames=("f0", "f1", "f2"))
        result = summarize_with_group_by([arr], GroupBySpec(dims=[]), AggSpec.of_type("mean"))

        # Pool everything except always_iterate → 1 row, mean of [0, 1, 2] = 1.0
        assert len(result) == 1
        assert result["mean"].item() == pytest.approx(1.0)
        assert "algorithm" in result.columns

    def test_user_group_by_splits_by_subject(self):
        arr = make_loaded_array(_exp(), subjects=("s1", "s2"), frames=("f0", "f1"))
        result = summarize_with_group_by([arr], GroupBySpec(dims=["subject"]), AggSpec.of_type("mean"))

        # s1: values [0, 1] → mean=0.5; s2: values [2, 3] → mean=2.5
        assert len(result) == 2
        assert result.loc[result["subject"] == "s1", "mean"].item() == pytest.approx(0.5)
        assert result.loc[result["subject"] == "s2", "mean"].item() == pytest.approx(2.5)

    def test_custom_agg_column_name(self):
        arr = make_loaded_array(_exp())
        result = summarize_with_group_by([arr], GroupBySpec(dims=[]), AggSpec.of_type(mean="my_score"))

        assert "my_score" in result.columns
        assert "mean" not in result.columns
