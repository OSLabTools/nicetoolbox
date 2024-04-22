"""
    Functions to check created data
"""
import sys
import os
import pytest
import oslab_utils.filehandling as fh
import numpy as np

RUNNER_OUTPUT_FOLDER = "../artifacts/example_data/input_data/dyadic_communication/experiements/20240421_dyadic_communication_functional_tests"
LOCAL_OUTPUT_FOLDER = "../artifacts/example_data/output/20240417_dyadic_communication_functional_tests"
THRESHOLD = 0.001

def find_hdf5_files(directory):
    """Recursively find all HDF5 files in the given directory."""
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".h5"):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files

@pytest.mark.parametrize("folder_pair", [(RUNNER_OUTPUT_FOLDER, LOCAL_OUTPUT_FOLDER)])
def test_hdf5_files(folder_pair):
    folder_runner, folder_local = folder_pair
    files_runner = sorted(find_hdf5_files(folder_runner))
    files_local = sorted(find_hdf5_files(folder_local))
    print(f"Number of runner files:{len(files_runner)}\nNumber of local files:{len(files_local)}")
    assert len(files_runner) == len(files_local), "Number of HDF5 files in both folders should be the same"
    for file_runner, file_local in zip(files_runner, files_local):
        data_runner, data_list_runner = fh.read_hdf5(file_runner)
        (data_local, data_list_local) = fh.read_hdf5(file_local)

        for idx in range(len(data_list_local)):
            assert data_runner[idx].shape == data_local[idx].shape, f"Shapes differ for dataset: runner - {data_runner.shape} vs local - {data_local.shape}"
            print(f'Runner {data_runner[idx].shape}, Local: {data_local[idx].shape}')

            if data_runner[idx].dtype == bool and data_local[idx].dtype == bool:
                differences = data_runner[idx] != data_local[idx]
                # indices_of_differences = np.where(differences)
                assert np.sum(differences) == 0, f"There is difference for runner {file_runner} and local {file_local} files"
            elif np.issubdtype(data_runner.dtype, np.number) and np.issubdtype(data_local.dtype, np.number):
                differences = np.where(np.abs(data_runner[idx] - data_local[idx]) > THRESHOLD)
                assert differences[0].size == 0, f"number of differences found: {len(list(zip(*differences)))} between runner {file_runner} and local {file_local} files"
            else:
                raise ValueError("Unsupported array data types.")


