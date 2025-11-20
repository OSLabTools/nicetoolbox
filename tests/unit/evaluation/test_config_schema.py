from pathlib import Path

import pytest
from pydantic import ValidationError

from nicetoolbox.evaluation.config_schema import IOConfig


def test_io_config_validates_with_correct_data():
    """
    Given: A dictionary with all the required fields for IOConfig.
    When:  IOConfig.model_validate is called with this data.
    Then:  No exception is raised, and the attributes are set correctly.
    """
    data = {
        "experiment_folder": "/path/to/exp",
        "output_folder": "/path/to/out",
        "eval_visualization_folder": "/path/to/viz",
    }
    config = IOConfig.model_validate(data)

    assert config.experiment_folder == Path("/path/to/exp")
    assert isinstance(config.output_folder, Path)
    assert config.experiment_io == {}  # Check that default factory worked


def test_io_config_fails_if_required_field_is_missing():
    """
    Given: A dictionary missing a required field ('output_folder').
    When:  IOConfig.model_validate is called.
    Then:  A Pydantic ValidationError is raised.
    """
    data = {
        "experiment_folder": "/path/to/exp",
        "eval_visualization_folder": "/path/to/viz",
    }
    # 'with pytest.raises(...)' is the standard way to assert that an
    # exception is expected. The test will fail if no exception is raised.
    with pytest.raises(ValidationError):
        IOConfig.model_validate(data)
