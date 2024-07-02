"""

"""
import logging
import sys
import h5py
import toml
import json
import numpy as np

# TODO: Functions into archive? hdf5 format outdated
def save_to_hdf5(array_list,  groups_list, output_file, index=[]):
    """save the numpy array into given output_file

    Parameters
    ----------
    array_list: list of numpy.arrays
        the data that will be saved as hdf5 file. each data into the list will be saved as a separate group
    groups_list: list of strings
        data group names given into a list
    output_file: str
        path where the hdf5 file should be saved
    index: str
        if it is given, will be saved as the data row attributes, i.e., filename_list
        Please note that, all groups will share the same index values

    Returns
    -------
        Save the data as hdf5 file
    """
    # Check the given parameters
    if len(array_list) != len(groups_list):
        raise ValueError("The lengths of the array and group lists don't match.")
    if index:
        for array in array_list:
            if array.shape[0]!=len(index):
                raise ValueError("The length of the index does not match with data shape.")

    with h5py.File(output_file, 'w') as f:
        for i, array in enumerate(array_list):
            dataset = f.create_group(groups_list[i]).create_dataset("data", data=array_list[i])
            if index:
                for i, index_key in enumerate(index):
                    dataset.attrs[index_key] = i


def load_config(config_file):
    if config_file.endswith('toml'):
        config = toml.load(config_file)
    else:
        raise NotImplementedError(
                f"config_file type {config_file} is not supported currently. "
                f"Implemented is toml.")
    return config


def save_toml(dic, file_path):
    with open(file_path, 'w') as file:
        string = toml.dump(dic, file)


def assert_and_log(condition, message):
    try:
        assert condition, message
    except AssertionError as e:
        logging.error(f"Assertion failed: {e}")
        sys.exit(1)


def load_json_file(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        return data


def read_hdf5(path):
    """Read data from hdf5 file.

    Parameters
    ----------
    path: the path of hdf5 file

    Returns
    -------
    A list holds the data of each group
    """
    data = []
    try:
        with h5py.File(path, 'r') as f:
            group_names = list(f.keys())
            for group in f.keys():
                data.append(np.array(f[group]["data"]))
        return data, group_names
    except ValueError: #ToDo debug
        print("MemoryError: Not enough memory to load data.")
        return None



def get_ordered_attributes(dset):
    """ Function to get data attributes in the correct order

    Parameters
    ----------
    dset: dataset

    Returns
    -------

    """
    attr_dict = {}
    for attr_name, attr_value in dset.attrs.items():
        attr_dict[attr_value] = attr_name
    ordered_attrs = [attr_dict[i] for i in sorted(attr_dict.keys())]
    return ordered_attrs


def get_attribute_by_index(index, hdf5_file_path, group_name):
    with h5py.File(hdf5_file_path, 'r') as f:
        # Get the dataset
        dset = f[f"{group_name}/data"]

        # Iterate over all attributes to find the attribute name corresponding to the given index
        for attr_name, attr_value in dset.attrs.items():
            if attr_value == index:
                return attr_name
    return None