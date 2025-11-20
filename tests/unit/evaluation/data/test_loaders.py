from pathlib import Path

import numpy as np
import pytest

from nicetoolbox.evaluation.data.loaders import AnnotationLoader, PredictionLoader

# --- Fixture ---


@pytest.fixture
def create_mock_npz_file(tmp_path):  # tmp_path is a built-in pytest fixture
    """
    A fixture that creates a mock .npz file in a temporary directory
    and returns its path and the data it contains.
    """
    file_path = tmp_path / "mock_data.npz"
    data = {
        "3d": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "2d": np.ones((5, 5), dtype=np.int32),
    }
    np.savez_compressed(file_path, **data)
    return file_path, data


# --- Tests for PredictionLoader ---


def test_prediction_loader_loads_correct_array(create_mock_npz_file):
    """
    Given: A valid .npz file with a '3d' data key.
    When:  load_full_array is called for that key.
    Then:  It should return the correct numpy array.
    """
    # Arrange
    loader = PredictionLoader()
    file_path, original_data = create_mock_npz_file

    # Act
    loaded_array = loader.load_full_array(path=file_path, data_key="3d")

    # Assert
    assert isinstance(loaded_array, np.ndarray)
    assert np.array_equal(loaded_array, original_data["3d"])


def test_prediction_loader_raises_key_error_for_missing_key(create_mock_npz_file):
    """
    Given: A valid .npz file.
    When:  load_full_array is called with a data_key that does not exist.
    Then:  It should raise a KeyError.
    """
    # Arrange
    loader = PredictionLoader()
    file_path, _ = create_mock_npz_file

    # Act & Assert
    with pytest.raises(KeyError, match="Data key 'missing_key' not found"):
        loader.load_full_array(path=file_path, data_key="missing_key")


def test_prediction_loader_raises_file_not_found_for_missing_file():
    """
    Given: A path to a file that does not exist.
    When:  load_full_array is called.
    Then:  It should raise a FileNotFoundError (propagated from np.load).
    """
    # Arrange
    loader = PredictionLoader()
    non_existent_path = Path("non/existent/file.npz")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        loader.load_full_array(path=non_existent_path, data_key="any_key")


def test_prediction_loader_caches_results(create_mock_npz_file):
    """
    Tests that the LRU cache is working by directly inspecting its state
    via the cache_info() method.
    """
    # Arrange
    loader = PredictionLoader()
    file_path, _ = create_mock_npz_file

    # Act & Assert: First call
    # Before any calls, we expect 0 hits and 0 misses
    initial_info = loader._cached.cache_info()
    assert initial_info.hits == 0
    assert initial_info.misses == 0

    loader.load_full_array(path=file_path, data_key="3d")

    # After the first call, we expect 1 miss and 0 hits
    first_call_info = loader._cached.cache_info()
    assert first_call_info.hits == 0
    assert first_call_info.misses == 1

    # Act & Assert: Second call (should be a cache hit)
    loader.load_full_array(path=file_path, data_key="3d")

    # After the second call, we expect 1 hit and still 1 miss
    second_call_info = loader._cached.cache_info()
    assert second_call_info.hits == 1
    assert second_call_info.misses == 1

    # Act & Assert: Clear cache and call again
    loader.close_files()  # This calls _cached.cache_clear()

    # After clearing, the info should be reset
    cleared_info = loader._cached.cache_info()
    assert cleared_info.hits == 0
    assert cleared_info.misses == 0

    loader.load_full_array(path=file_path, data_key="3d")

    # After calling again on a cleared cache, we expect 1 miss
    final_info = loader._cached.cache_info()
    assert final_info.hits == 0
    assert final_info.misses == 1


# --- Tests for AnnotationLoader ---


def test_annotation_loader_initialization_raises_error_if_file_missing():
    """
    Given: A path to an annotation file that does not exist.
    When:  The AnnotationLoader is initialized.
    Then:  It should raise a FileNotFoundError.
    """
    non_existent_path = Path("non/existent/annotations.npz")
    with pytest.raises(FileNotFoundError):
        AnnotationLoader(path_to_annotations=non_existent_path)


def test_annotation_loader_loads_correct_array(create_mock_npz_file):
    """
    Given: A valid annotation .npz file.
    When:  load_full_array is called for a valid key.
    Then:  It should return the correct numpy array.
    """
    # Arrange
    file_path, original_data = create_mock_npz_file
    loader = AnnotationLoader(path_to_annotations=file_path)

    # Act
    loaded_array = loader.load_full_array(data_key="2d")

    # Assert
    assert isinstance(loaded_array, np.ndarray)
    assert np.array_equal(loaded_array, original_data["2d"])


def test_annotation_loader_returns_none_for_missing_key(create_mock_npz_file):
    """
    Given: A valid annotation file.
    When:  load_full_array is called for a non-existent key.
    Then:  It should return None without raising an error.
    """
    # Arrange
    file_path, _ = create_mock_npz_file
    loader = AnnotationLoader(path_to_annotations=file_path)

    # Act
    result = loader.load_full_array(data_key="missing_key")

    # Assert
    assert result is None


def test_annotation_loader_caches_results(create_mock_npz_file):
    """
    Tests that the LRU cache is working by directly inspecting its state
    via the cache_info() method.
    """
    # Arrange
    file_path, _ = create_mock_npz_file
    loader = AnnotationLoader(path_to_annotations=file_path)

    # Act & Assert: First call
    loader.load_full_array(data_key="3d")
    first_call_info = loader._cached.cache_info()
    assert first_call_info.hits == 0
    assert first_call_info.misses == 1

    # Act & Assert: Second call (should hit the cache)
    loader.load_full_array(data_key="3d")
    second_call_info = loader._cached.cache_info()
    assert second_call_info.hits == 1
    assert second_call_info.misses == 1

    # Act & Assert: Call with a different key (should be a miss)
    loader.load_full_array(data_key="2d")
    third_call_info = loader._cached.cache_info()
    assert third_call_info.hits == 1
    assert third_call_info.misses == 2

    # Act & Assert: Clear cache and call again
    loader.close_files()
    loader.load_full_array(data_key="3d")
    final_info = loader._cached.cache_info()
    assert final_info.hits == 0
    assert final_info.misses == 1
