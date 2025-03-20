import os
import shutil
from pathlib import Path

import numpy as np
import toml

from nicetoolbox.utils import config as cf


class MiniDatasetLoader:
    def __init__(self, source_url, target_folder, temp_directory):
        self.source_url = source_url
        self.target_folder = target_folder
        self.temp_directory = temp_directory

    def download_data(self):
        if self.target_folder.is_dir():
            shutil.rmtree(self.target_folder)
        os.makedirs(self.target_folder, exist_ok=True)
        os.system(f"wget -q -P {self.temp_directory} {self.source_url}")
        os.system(
            f"unzip -q {self.temp_directory / 'index.html?dl=1'} -d "
            f"{self.target_folder}"
        )


class MockDataLoader:
    def __init__(self, width=1028, height=752):
        self.width = width
        self.height = height

    def get_white_image(self, max_value=255):
        return np.ones((self.height, self.width)) * max_value

    def get_black_image(self):
        return np.zeros((self.height, self.width))

    def get_image_with_random_nans(self, value=1):
        image = np.random.uniform(-1, 1, (self.height, self.width))
        image[image > 0] = value
        image[image <= 0] = np.nan
        if not np.all(image == image):
            image = self.get_image_with_random_nans(value)
        return image

    def get_random_positive_image(self, max_value=255):
        return np.random.randint(0, max_value, (self.height, self.width))


def update_machine_specifics(output_folder_path, machine_specifics_file):
    with open("machine_specific_paths.toml") as file:
        data = file.readlines()

    # find the line containing the output folder path
    out_line = None
    for i, line in enumerate(data):
        if "output_folder_path = " in line:
            out_line = i

    data[out_line] = f"output_folder_path = '{output_folder_path}'\n"
    with open(machine_specifics_file, "w") as file:
        file.writelines(data)


def create_machine_specifics(machine_specifics_file, datasets_folder, outputs_folder):
    os.makedirs(os.path.dirname(machine_specifics_file), exist_ok=True)
    os.system(
        "make create_machine_specifics "
        f"MACHINE_SPECIFICS={machine_specifics_file} "
        f"DATASETS_DIR={datasets_folder} "
        f"OUTPUTS_DIR={outputs_folder}"
    )


def generate_npz_file_names_suffix(run_config, dataset_name):
    for video in run_config["run"][dataset_name]["videos"]:
        video["dataset_name"] = dataset_name
        io_config = cf.config_fill_auto(run_config["io"])
        io_config = cf.config_fill_placeholders(io_config, video)
        io_config = cf.config_fill_placeholders(io_config, io_config)
        io_config = cf.config_fill_placeholders(io_config, io_config)

        experiment_dir = Path(
            os.path.relpath(io_config["out_sub_folder"], "<output_folder_path>")
        )

        for component in run_config["run"][dataset_name]["components"]:
            for algorithm in run_config["component_algorithm_mapping"][component]:
                yield Path(experiment_dir / component / f"{algorithm}.npz")


def generate_npz_file_names(run_config, dataset_name, outputs_folder):
    for file_name in generate_npz_file_names_suffix(run_config, dataset_name):
        yield outputs_folder / file_name


def create_detectors_config(detectors_config_file):
    detectors_config = toml.load(Path("configs") / "detectors_config.toml")
    detectors_config["algorithms"]["xgaze_3cams"]["window_length"] = 5
    detectors_config["frameworks"]["mmpose"]["window_length"] = 5

    cf.save_config(detectors_config, detectors_config_file)
