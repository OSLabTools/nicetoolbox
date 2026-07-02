import pytest

from nicetoolbox.configs.schemas.evaluation_input_block import ExperimentInput
from nicetoolbox.evaluation.data.input_loader import _resolve_experiment_source
from tests.unit.evaluation.data.conftest import (
    MULTI_DATASET,
    MULTI_VIDEO,
    SINGLE_DATASET,
    THREE_ALGORITHMS,
    THREE_DATASETS,
    TWO_ALGORITHMS,
    expected_experiment_metas,
    make_experiment_config,
)

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_single_entry(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, SINGLE_DATASET, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks")

        result = _resolve_experiment_source(block, cfg)

        assert result == expected_experiment_metas(tmp_path, SINGLE_DATASET, mapping, "landmarks")

    def test_two_algorithms(self, tmp_path):
        cfg = make_experiment_config(tmp_path, SINGLE_DATASET, TWO_ALGORITHMS)
        block = ExperimentInput(component="body_joints", npz_key="landmarks")

        result = _resolve_experiment_source(block, cfg)

        assert result == expected_experiment_metas(tmp_path, SINGLE_DATASET, TWO_ALGORITHMS, "landmarks")
        assert len(result) == 2

    def test_two_datasets(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, MULTI_DATASET, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks")

        result = _resolve_experiment_source(block, cfg)

        assert result == expected_experiment_metas(tmp_path, MULTI_DATASET, mapping, "landmarks")
        assert len(result) == 2

    def test_multiple_subsequences(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, MULTI_VIDEO, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks")

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 2
        assert result[0].subsequence.subsequence_index == 0
        assert result[1].subsequence.subsequence_index == 1

    def test_resolved_subsequence_values_propagated(self, tmp_path):
        # Even when the declared inputs use timestamp strings, the resolved
        # SubsequenceInfo carries integer frame values from subsequence_meta.toml.
        datasets = {
            "ds1": {
                "fps": 60,
                "videos": [
                    {
                        "session_ID": "s01",
                        "sequence_ID": "seq01",
                        "video_start": "00:01:00",
                        "video_length": "00:00:30",
                    },
                ],
            },
        }
        cfg = make_experiment_config(tmp_path, datasets, {"algo_a": ["body_joints"]})
        block = ExperimentInput(component="body_joints", npz_key="landmarks")

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 1
        assert isinstance(result[0].subsequence.video_start, int)
        assert isinstance(result[0].subsequence.video_length, int)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_filter_by_dataset(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, MULTI_DATASET, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks", dataset="ds1")

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 1
        assert result[0].dataset == "ds1"

    def test_filter_by_algorithm(self, tmp_path):
        cfg = make_experiment_config(tmp_path, SINGLE_DATASET, TWO_ALGORITHMS)
        block = ExperimentInput(component="body_joints", npz_key="landmarks", algorithm="algo_b")

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 1
        assert result[0].algorithm == "algo_b"

    def test_filter_by_session(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, MULTI_DATASET, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks", session="s01")

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 1
        assert result[0].session == "s01"

    def test_filter_by_subsequence_index(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, MULTI_VIDEO, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks", subsequence=1)

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 1
        assert result[0].subsequence.subsequence_index == 1

    def test_wildcard_returns_all(self, tmp_path):
        cfg = make_experiment_config(tmp_path, MULTI_DATASET, TWO_ALGORITHMS)
        block = ExperimentInput(component="body_joints", npz_key="landmarks")

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 4  # 2 datasets x 2 algorithms

    def test_filter_by_dataset_list(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, THREE_DATASETS, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks", dataset=["ds1", "ds3"])

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 2
        assert {m.dataset for m in result} == {"ds1", "ds3"}

    def test_filter_by_algorithm_list(self, tmp_path):
        cfg = make_experiment_config(tmp_path, SINGLE_DATASET, THREE_ALGORITHMS)
        block = ExperimentInput(component="body_joints", npz_key="landmarks", algorithm=["algo_a", "algo_c"])

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 2
        assert {m.algorithm for m in result} == {"algo_a", "algo_c"}

    def test_filter_by_session_list(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, THREE_DATASETS, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks", session=["s01", "s02"])

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 2
        assert {m.session for m in result} == {"s01", "s02"}

    def test_filter_by_subsequence_list(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        three_videos = {
            "ds1": {
                "fps": 30,
                "videos": [
                    {"session_ID": "s01", "sequence_ID": "seq01", "video_start": 0, "video_length": 100},
                    {"session_ID": "s01", "sequence_ID": "seq01", "video_start": 100, "video_length": 100},
                    {"session_ID": "s01", "sequence_ID": "seq01", "video_start": 200, "video_length": 100},
                ],
            },
        }
        cfg = make_experiment_config(tmp_path, three_videos, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks", subsequence=[0, 2])

        result = _resolve_experiment_source(block, cfg)

        assert len(result) == 2
        assert {m.subsequence.subsequence_index for m in result} == {0, 2}

    def test_filter_list_no_match_returns_empty(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, SINGLE_DATASET, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks", dataset=["x", "y"])

        result = _resolve_experiment_source(block, cfg)

        assert result == []

    def test_filter_no_match_returns_empty(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, SINGLE_DATASET, mapping)
        block = ExperimentInput(component="body_joints", npz_key="landmarks", dataset="nonexistent")

        result = _resolve_experiment_source(block, cfg)

        assert result == []


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_npz_raises(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, SINGLE_DATASET, mapping)
        # Delete the npz file that was created by the factory
        npz = tmp_path / "ds1" / "s01" / "seq01" / "body_joints" / "algo_a.npz"
        npz.unlink()

        block = ExperimentInput(component="body_joints", npz_key="landmarks")

        with pytest.raises(FileNotFoundError, match="Expected NPZ output is missing"):
            _resolve_experiment_source(block, cfg)

    def test_dataset_missing_from_properties_raises(self, tmp_path):
        mapping = {"algo_a": ["body_joints"]}
        cfg = make_experiment_config(tmp_path, SINGLE_DATASET, mapping)
        # Remove dataset from dataset_config but leave it in run_config
        del cfg.dataset_config["ds1"]

        block = ExperimentInput(component="body_joints", npz_key="landmarks")

        with pytest.raises(KeyError, match="ds1"):
            _resolve_experiment_source(block, cfg)
