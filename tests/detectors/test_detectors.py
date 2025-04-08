import logging
import os
import shutil as sht
import time
from pathlib import Path

import data_loader as dl
import toml
from fixtures import BaseTestCase

from nicetoolbox.detectors.main import main as detector_main
from nicetoolbox.utils import comparisons as cp
from nicetoolbox.utils import config as cf

logger = logging.getLogger("testing")
dd = "-" * 20


def check_npz_creation(run_config_file, dataset_name, outputs_folder):
    run_config = toml.load(run_config_file)
    all_exist = True

    logger.info(dd + f" {dataset_name}: Check creation of npz files " + dd)
    for npz_file in dl.generate_npz_file_names(
        run_config, dataset_name, outputs_folder
    ):
        if npz_file.exists():
            logger.info(f"{npz_file} ............... exists.")
        else:
            logger.info(f"{npz_file} ............... IS MISSING.")
            all_exist = False

    return all_exist


def check_results_against_expected(
    run_config_file, dataset_name, datasets_folder, outputs_folder
):
    run_config = toml.load(run_config_file)
    all_expected = True

    logger.info(dd + f" {dataset_name}: Compare npz to expected results " + dd)
    for npz_suffix in dl.generate_npz_file_names_suffix(run_config, dataset_name):
        prediction_file = outputs_folder / npz_suffix
        expectation_file = datasets_folder / dataset_name / npz_suffix

        if cp.compare_npz_files(
            prediction_file, expectation_file, rtol=1e-1, atol=1e-1
        ):
            logger.info(f"{prediction_file} ............... matches.")
        else:
            logger.info(f"{prediction_file} ............... DIFFERS.")
            all_expected = False

    return all_expected


class TestDyadicCommunication(BaseTestCase):
    dataset_name = "dyadic_communication"
    source_url = "https://keeper.mpdl.mpg.de/f/c87059d9a6af42cbbadb/?dl=1"

    target_folder = Path("artifacts")
    datasets_folder = target_folder / "datasets"
    outputs_folder = target_folder

    machine_specifics_file = target_folder / "machine_specific_paths.toml"
    detectors_config_file = target_folder / "detectors_config.toml"
    run_config_file = datasets_folder / dataset_name / "detectors_run_file.toml"
    dataset_properties_file = datasets_folder / dataset_name / "dataset_properties.toml"

    @classmethod
    def setUpClass(self) -> None:
        super().setUpClass()

        dataset = dl.MiniDatasetLoader(
            self.source_url, self.datasets_folder, self.temp_directory
        )
        dataset.download_data()

        dl.create_machine_specifics(
            self.machine_specifics_file, self.datasets_folder, self.outputs_folder
        )
        dl.create_detectors_config(self.detectors_config_file)
        self.create_run_config(self)

    def create_run_config(self):
        run_config = toml.load(Path("configs") / "detectors_run_file.toml")

        run_config["visualize"] = False
        run_config["run"] = {
            self.dataset_name: dict(
                components=list(run_config["component_algorithm_mapping"].keys()),
                videos=[
                    dict(
                        session_ID="PIS_ID_000",
                        sequence_ID="",
                        video_start=0,
                        video_length=10,
                    ),
                    dict(
                        session_ID="PIS_ID_000",
                        sequence_ID="",
                        video_start=450,
                        video_length=10,
                    ),
                    dict(
                        session_ID="PIS_ID_000",
                        sequence_ID="",
                        video_start=880,
                        video_length=10,
                    ),
                ],
            )
        }
        run_config["io"]["experiment_name"] = self.dataset_name + "_testing"
        run_config["io"]["dataset_properties"] = str(self.dataset_properties_file)
        run_config["io"]["detectors_config"] = str(self.detectors_config_file)
        # save updated run_file
        cf.save_config(run_config, self.run_config_file)
        # add required run_file_check by copying
        sht.copyfile(
            Path("configs") / "detectors_run_file_check.toml",
            self.run_config_file.parent / "detectors_run_file_check.toml",
        )

    def test_a_main(self):
        detector_main(str(self.run_config_file), str(self.machine_specifics_file))

    def test_b_npz_creation(self):
        self.assertTrue(
            check_npz_creation(
                self.run_config_file, self.dataset_name, self.outputs_folder
            )
        )

    def test_c_expected_output(self):
        self.assertTrue(
            check_results_against_expected(
                self.run_config_file,
                self.dataset_name,
                self.datasets_folder,
                self.outputs_folder,
            )
        )


# @unittest.skip("")
class TestCommunicationMultiview(BaseTestCase):
    dataset_name = "communication_multiview"

    target_folder = Path("artifacts")
    datasets_folder = target_folder / "datasets"
    outputs_folder = target_folder

    machine_specifics_file = target_folder / "machine_specific_paths.toml"
    run_config_file = Path("configs") / "detectors_run_file.toml"

    @classmethod
    def setUpClass(self) -> None:
        super().setUpClass()

        dl.create_machine_specifics(
            self.machine_specifics_file, self.datasets_folder, self.outputs_folder
        )

        string = f"make download_dataset DATASETS_DIR={self.datasets_folder}"
        os.system(f"cd {str(Path.cwd())} && " + string)

        # rename experiment output folder: <yyyymmdd> to the actual date
        sht.move(
            self.datasets_folder / self.dataset_name / "experiments" / "<yyyymmdd>",
            self.datasets_folder
            / self.dataset_name
            / "experiments"
            / time.strftime("%Y%m%d", time.localtime()),
        )

    def test_a_main(self):
        detector_main(str(self.run_config_file), str(self.machine_specifics_file))

    def test_b_npz_creation(self):
        self.assertTrue(
            check_npz_creation(
                self.run_config_file, self.dataset_name, self.outputs_folder
            )
        )

    def test_c_expected_output(self):
        self.assertTrue(
            check_results_against_expected(
                self.run_config_file,
                self.dataset_name,
                self.datasets_folder,
                self.outputs_folder,
            )
        )
