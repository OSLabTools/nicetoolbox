"""
    Functions to check created data
"""
import sys
import os
import pytest
import numpy as np

RUNNER_OUTPUT_FOLDER = "../artifacts/example_data/output/experiments/functional_test_experiment"
EXPECTED_OUTPUT_FOLDER = "../artifacts/example_data/output/experiments/functional_test_experiment"
THRESHOLD = 0.001

print(f"current working dir -- : {os.getcwd()}")

def read_npz_file(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data

def find_npz_files(directory):
    """Recursively find all NPZ files in the given directory."""
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(root, file))
    return npz_files

@pytest.mark.parametrize("folder_pair", [(RUNNER_OUTPUT_FOLDER, EXPECTED_OUTPUT_FOLDER)])
def test_video_folders(folder_pair):
    folder_runner, folder_expected = folder_pair
    runner_list = [f for f in os.listdir(folder_runner) if os.path.isdir(f)]
    expected_list = [f for f in os.listdir(folder_expected) if os.path.isdir(f)]
    assert len(runner_list) == len(expected_list), "Number of video folders are not same"

@pytest.mark.parametrize("folder_pair", [(RUNNER_OUTPUT_FOLDER, EXPECTED_OUTPUT_FOLDER)])
def test_npz_files(folder_pair):
    folder_runner, folder_expected = folder_pair
    files_runner = sorted(find_npz_files(folder_runner))
    files_expected = sorted(find_npz_files(folder_expected))
    print(f"Number of runner npz files:{len(files_runner)}\nNumber of expected npz files:{len(files_expected)}")
    assert len(files_runner) == len(files_expected), "Number of NPZ files are different"
    for file_runner, file_expected in zip(files_runner, files_expected):
        data_runner = read_npz_file(file_runner)
        data_expected = read_npz_file(file_expected)
        print(f"File - runner:{file_runner}\n file - expected:{len(file_expected)}")
        for key in data_expected:
            if key != "data_description":
                assert data_runner[key].shape == data_expected[key].shape, f"Shapes differ for {key}: runner - {data_runner[key].shape} vs expected - {data_runner[key].shape}"

                differences = ~np.isclose(data_runner[key], data_expected[key], atol=0.1, equal_nan=True)
                indices_of_differences = np.where(differences)
                assert np.all(~differences), f"There is difference for runner {file_runner} and expected {file_expected} files that exceeds the threshold - Indices of differences: {indices_of_differences}"

