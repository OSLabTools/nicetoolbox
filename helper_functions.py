import os
import numpy as np
import pandas as pd

def read_npz_file(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data

def find_npz_files(directory):
    """Recursively find all HDF5 files in the given directory."""
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(root, file))
    return npz_files

def convert_npz_to_csv_files(npz_path, output_folder):
        filename = os.path.basename(npz_path)
        component_name = os.path.basename(os.path.dirname(npz_path))
        video_name = os.path.basename(os.path.dirname(os.path.dirname(npz_path)))

        data = read_npz_file(npz_path)
        data_desc = data['data_description']

        for key in data:
            if key != "data_description":
                arr = data[key]
                data_desc_arr = data_desc.item()[key]
                arr_dimensions = len(arr.shape)

                if len(set(data_desc_arr['axis3'])) == 1:
                    data_desc_arr['axis3'] = [f'{value}_{idx}' for idx, value in enumerate(data_desc_arr['axis3'])]

                # first 3 dimensions always, Subject, Camera, Frames
                # if array has 4 dimensions - column names will be dimension4[i]
                # if array has 5 dimensions - column names will be dimension4[i]_dimension5[idx]
                rows = []
                index_tuples = []
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        for k in range(arr.shape[2]):
                            flat_values = arr[i, j, k].flatten()
                            if arr_dimensions == 4:
                                # Create column labels based on the flattened structure
                                column_labels = [
                                    f"{data_desc_arr['axis3'][int(idx)]}" for idx in range(len(flat_values))
                                ]
                            elif arr_dimensions == 5:
                                column_labels = [
                                    f"{data_desc_arr['axis3'][idx//3]}_{data_desc_arr['axis4'][int(idx%3)]}" for idx in range(len(flat_values))
                                ]
                            rows.append(flat_values)
                            index_tuples.append((i, j, k))

                # Create a DataFrame
                df = pd.DataFrame(rows, columns=column_labels, index=pd.MultiIndex.from_tuples(index_tuples, names=["Subject", "Camera", "Frame"]))
                df.reset_index(inplace=True)

                # Relabel subject and camera columns
                subjects_dict = {i: data_desc_arr['axis0'][i] for i in range(len(data_desc_arr['axis0']))}
                df['Subject'] = df['Subject'].map(subjects_dict).fillna(df['Subject'])

                if data_desc_arr['axis1']:
                    cameras_dict ={i: data_desc_arr['axis1'][i] for i in range(len(data_desc_arr['axis1']))}
                else:
                    cameras_dict = {0: 'none'}
                df['Camera'] = df['Camera'].map(cameras_dict).fillna(df['Camera'])

                output_filename = f'{video_name}_{component_name}_{filename.split(".")[0]}_{key}.csv'
                df.to_csv(os.path.join(output_folder, output_filename), index=False)

if __name__ == '__main__':
    VIDEO_FOLDER = r"F:\example_data\example_data\output\functional_test_experiment\mpi_inf_3dhp_S1_s421_l10"
    OUTPUT_FOLDER = r"F:\example_data\example_data\output\csv_files"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    npz_files_list = find_npz_files(VIDEO_FOLDER)
    for file in npz_files_list:
        convert_npz_to_csv_files(file, OUTPUT_FOLDER)