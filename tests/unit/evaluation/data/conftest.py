"""
Fixtures for unit tests related to data discovery and loading in evaluation.
"""

from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest

from nicetoolbox.evaluation.data.discovery import (
    ChunkWorkItem,
    DiscoveryEngine,
    FrameInfo,
)
from nicetoolbox.evaluation.data.loaders import AnnotationLoader, PredictionLoader
from nicetoolbox.evaluation.in_out import IO
from tests.unit.evaluation.data.helpers import DiscoveryMockDataBuilder


@pytest.fixture
def discovery_engine_factory(mocker):  # mocker is a fixture from pytest-mock plug-in
    """
    A factory fixture that creates a DiscoveryEngine instance with mocked dependencies.
    Allows for overriding configs and mock data on a per-test basis.
    """

    def _factory(
        run_config_overrides=None,
        dataset_props_overrides=None,
        eval_config_overrides=None,
        mock_pred_desc=None,
        mock_annot_desc=None,
        gt_needed=True,
    ):
        # 1. Use the builder to create default or overridden configs
        builder = DiscoveryMockDataBuilder()
        run_config = builder.build_run_config(run_config_overrides)
        dataset_props = builder.build_dataset_config(dataset_props_overrides)
        eval_config = builder.build_evaluation_config(eval_config_overrides)

        # 2. Mock the external dependencies (IO and np.load)
        mock_io = create_autospec(IO, instance=True)
        mock_io.get_detector_results_file.return_value = "fake/pred.npz"
        mock_io.path_to_annotations = dataset_props.path_to_annotations

        # Mock np.load to return different descriptions based on the path
        def np_load_side_effect(path, allow_pickle=True):  # noqa: ARG001
            mock_context = MagicMock()
            if path == "fake/pred.npz" and mock_pred_desc:
                mock_context.__enter__.return_value = {
                    "data_description": np.array(mock_pred_desc)
                }
            elif path == dataset_props.path_to_annotations and mock_annot_desc:
                mock_context.__enter__.return_value = {
                    "data_description": np.array(mock_annot_desc)
                }
            else:
                # Default case or error case
                raise FileNotFoundError
            return mock_context

        mocker.patch(
            "nicetoolbox.evaluation.data.discovery.np.load",
            side_effect=np_load_side_effect,
        )

        # 3. Create and return the real DiscoveryEngine instance
        # We can disable gt_needed to specifically test the _load_and_process method
        if not gt_needed:
            eval_config.metric_types["point_cloud_metrics"].gt_required = False

        return DiscoveryEngine(
            io_manager=mock_io,
            run_config=run_config,
            dataset_properties=dataset_props,
            evaluation_config=eval_config,
        )

    return _factory


@pytest.fixture
def mock_pred_loader():
    """Provides a mock PredictionLoader."""
    return create_autospec(PredictionLoader, instance=True)


@pytest.fixture
def mock_annot_loader():
    """Provides a mock AnnotationLoader."""
    return create_autospec(AnnotationLoader, instance=True)


@pytest.fixture
def mock_chunk_factory():
    """
    A factory fixture to create mock ChunkWorkItem objects,
    allowing customization for different test scenarios.
    """

    def _factory(**kwargs):
        # Provide default values for all attributes accessed in the code under test.
        defaults = {
            "session": "S1",
            "sequence": "Seq1",
            "metric_type": "A",
            "component": "c1",
            "algorithm": "algo1",
            "pred_path": Path("fake/pred.npz"),
            "pred_data_key": "3d",
            "annot_path": Path("fake/gt.npz"),
            "annot_data_key": "S1__3d",
            "frames": [
                create_autospec(
                    FrameInfo,
                    instance=True,
                    pred_slicing_indices=(0, 0, 0),
                    annot_slicing_indices=(0, 0, 0),
                ),
                create_autospec(
                    FrameInfo,
                    instance=True,
                    pred_slicing_indices=(1, 1, 1),
                    annot_slicing_indices=(1, 1, 1),
                ),
            ],
            "pred_reconciliation_map": {},
            "gt_reconciliation_map": {},
        }
        # Override defaults with any provided kwargs
        defaults.update(kwargs)
        return create_autospec(ChunkWorkItem, instance=True, **defaults)

    return _factory
