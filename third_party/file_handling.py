"""

"""

import h5py
import toml


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
