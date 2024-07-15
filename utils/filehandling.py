"""
    Helper functions for reading, writing and parsing text files
"""
import h5py
import numpy as np
import json


def save_toml(dic, file_path):
    with open(file_path, 'w') as file:
        string = toml.dump(dic, file)


def load_json_file(json_path: str) -> dict:
    """
    Load JSON data from a file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        dict: The JSON data loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON data.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
        return data

# TODO: Functions into archive? hdf5 format outdated
def save_to_hdf5(array_list: list, groups_list: list, output_file: str, index: list = []) -> None:
    """
    Save the numpy arrays into a given HDF5 file. Each array is saved as a separate group.

    Args:
        array_list (list of numpy.ndarray):
            The data that will be saved as HDF5 file. Each data in the list will be saved as a 
            separate group.
        groups_list (list of str):
            Data group names given into a list.
        output_file (str):
            Path where the HDF5 file should be saved.
        index (list of str, optional):
            If it is given, it will be saved as the data row attributes, i.e., filename_list.
            Please note that, all groups will share the same index values. 
            Default is an empty list.

    Returns:
        None

    Raises:
        ValueError:
            If the lengths of the array and group lists don't match, or if the lengths of the index
            does not match with data shape.
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

def read_hdf5(path: str) -> tuple | None:
    """
    Read data from an HDF5 file and return it as a list of numpy arrays along with group names.

    Args:
        path (str): The path to the HDF5 file.

    Returns:
        data (list of numpy.ndarray): A list of numpy arrays containing the data from the HDF5 file.
        group_names (list of str): A list of group names present in the HDF5 file.

    Raises:
        ValueError: If there is not enough memory to load the data.
    """
    data = []
    try:
        with h5py.File(path, 'r') as f:
            group_names = list(f.keys())
            for group in f.keys():
                data.append(np.array(f[group]["data"]))
        return data, group_names
    except ValueError: # ToDo debug
        print("MemoryError: Not enough memory to load data.")
        return None


def get_ordered_attributes(dset):
    """
    Function to get data attributes in the correct order.
    
    This function takes a HDF5 dataset as input and returns a list of attribute names 
    sorted in ascending order based on their corresponding attribute values.

    Parameters:
        dset (h5py.Dataset): The HDF5 dataset from which to retrieve the attributes.

    Returns:
        list: A list of attribute names sorted in ascending order based on their values.

    Example:
        >>> import h5py
        >>> with h5py.File('my_file.h5', 'r') as f:
        >>>     dset = f['my_group/my_dataset']
        >>>     ordered_attrs = get_ordered_attributes(dset)
        >>>     print(ordered_attrs)
        ['attr1', 'attr2', 'attr3']
    """
    attr_dict = {}
    for attr_name, attr_value in dset.attrs.items():
        attr_dict[attr_value] = attr_name
    ordered_attrs = [attr_dict[i] for i in sorted(attr_dict.keys())]
    return ordered_attrs


def get_attribute_by_index(index: int, hdf5_file_path: str, group_name: str) -> str | None:
    """
    Retrieves the attribute name corresponding to a given index in an HDF5 dataset.

    Args:
        index (int): The index value to search for in the dataset attributes.
        hdf5_file_path (str): The path to the HDF5 file.
        group_name (str): The name of the group within the HDF5 file where the dataset is located.

    Returns:
        str | None: The attribute name corresponding to the given index, or None if no matching attribute is found.

    Example:
        >>> with h5py.File('my_file.h5', 'r') as f:
        >>>     attr_name = get_attribute_by_index(123, 'my_file.h5', 'my_group')
        >>>     print(attr_name)  # Output: 'filename123'
    """
    with h5py.File(hdf5_file_path, 'r') as f:
        # Get the dataset
        dset = f[f"{group_name}/data"]

        # Iterate over all attributes to find the attribute name corresponding to the given index
        for attr_name, attr_value in dset.attrs.items():
            if attr_value == index:
                return attr_name
    return None

