# very basic smoke tests that run detectors, evaluation and visualizer
# and check if they don't crash without checking actual output
# you would be surprised how much time it actually saves

import shutil
from pathlib import Path

import pytest
import toml

from nicetoolbox.detectors.main import main as run_detectors
from nicetoolbox.evaluation.main import main_evaluation_run as run_evaluation
from nicetoolbox.visual.media.main import main as run_visualizer

# useful for debug - we can skip detectors run
# if we already have results under temp folder
RUN_DETECTORS = True

INPUT_CONFIGS_FOLDER_PATH = Path("configs")
INPUT_MACHINE_SPECIFIC_PATH = Path("machine_specific_paths.toml")
INPUT_DETECTORS_RUN_FILE_PATH = INPUT_CONFIGS_FOLDER_PATH / "detectors_run_file.toml"
INPUT_VISUALIZER_CONFIG_PATH = INPUT_CONFIGS_FOLDER_PATH / "visualizer_config.toml"
INPUT_EVALUATION_CONFIG_PATH = INPUT_CONFIGS_FOLDER_PATH / "evaluation_config.toml"

TMP_PATH = Path("../functional_tests/temp")
TMP_OUTPUT_PATH = TMP_PATH / "outputs"
TMP_MACHINE_SPECIFIC_PATH = TMP_PATH / INPUT_MACHINE_SPECIFIC_PATH.name
TMP_DETECTORS_RUN_FILE_PATH = TMP_PATH / INPUT_DETECTORS_RUN_FILE_PATH.name
TMP_VISUALIZER_CONFIG_PATH = TMP_PATH / INPUT_VISUALIZER_CONFIG_PATH.name
TMP_EVALUATION_CONFIG_PATH = TMP_PATH / INPUT_EVALUATION_CONFIG_PATH.name


def create_temp_machine_specific():
    # we load existing machine specific
    machine_specific = toml.load(INPUT_MACHINE_SPECIFIC_PATH)
    # patch output directory to temporary folder
    machine_specific["output_folder_path"] = str(TMP_OUTPUT_PATH)
    # and save this temp machine specific to temp folder
    with open(TMP_MACHINE_SPECIFIC_PATH, "w") as f:
        toml.dump(machine_specific, f)


def create_temp_detectors_run_file():
    # load existing detectors run file
    detectors_run_file = toml.load(INPUT_DETECTORS_RUN_FILE_PATH)
    # patch error_level to STRICT
    detectors_run_file["error_level"] = "STRICT"
    # and save this to temp folder
    with open(TMP_DETECTORS_RUN_FILE_PATH, "w") as f:
        toml.dump(detectors_run_file, f)


# detectors run first as fixture, as it's output is input for
# evaluation and visualizer
@pytest.fixture(scope="module")
def detectors_output():
    if RUN_DETECTORS:
        # recreate temporary output folder
        if TMP_PATH.exists():
            shutil.rmtree(TMP_PATH)
        TMP_PATH.mkdir()

        # we create a temporary machine specific and run file
        # just to make sure that we start from scratch and
        # don't interfere with any existing users output
        create_temp_machine_specific()
        create_temp_detectors_run_file()

        # run detectors
        run_detectors(str(TMP_DETECTORS_RUN_FILE_PATH), str(TMP_MACHINE_SPECIFIC_PATH))

    # get experiments output folder
    tmp_experiment_dir = TMP_OUTPUT_PATH / "experiments"
    experiment_outputs = list(tmp_experiment_dir.iterdir())
    # remove any folder that has _eval suffix
    experiment_outputs = [f for f in experiment_outputs if not f.name.endswith("_eval")]
    assert len(experiment_outputs) == 1, f"Expected one experiment output folder, instead: {experiment_outputs}"
    detectors_output = experiment_outputs[0]

    return detectors_output


@pytest.mark.functional
def test_detectors_smoke(detectors_output: Path):
    # this is very basic check that we got anything from detectors run
    # for checking if this actually anything useful, we have regression tests
    # TODO: we can extend it by checking if we have any npz, csv, etc
    # but if it's crash, it should crash much earlier in fixture
    assert detectors_output.exists()


def create_temp_visualizer_config(detectors_output: Path):
    # load existing visualizer config
    visualizer_config = toml.load(INPUT_VISUALIZER_CONFIG_PATH)
    # patch output directory to temporary folder and force headless mode
    visualizer_config["spawn_viewer"] = False
    visualizer_config["io"]["experiment_folder"] = str(detectors_output)
    # and save this to temp folder
    with open(TMP_VISUALIZER_CONFIG_PATH, "w") as f:
        toml.dump(visualizer_config, f)


@pytest.mark.functional
def test_visualizer_smoke(detectors_output: Path):
    # first we need to patch visualizer config with detectors output
    create_temp_visualizer_config(detectors_output)
    # run visualizer
    run_visualizer(str(TMP_VISUALIZER_CONFIG_PATH), str(TMP_MACHINE_SPECIFIC_PATH))


def create_temp_evaluation_config(detectors_output: Path):
    # load existing evaluation
    evaluation_config = toml.load(INPUT_EVALUATION_CONFIG_PATH)
    # patch output directory to temporary folder
    evaluation_config["io"]["experiment_folder"] = str(detectors_output)
    # and save this to temp folder
    with open(TMP_EVALUATION_CONFIG_PATH, "w") as f:
        toml.dump(evaluation_config, f)


@pytest.mark.functional
def test_evaluation_smoke(detectors_output: Path):
    # first we need to patch evaluation config with detectors output
    create_temp_evaluation_config(detectors_output)
    # run evaluation
    run_evaluation(str(TMP_EVALUATION_CONFIG_PATH), str(TMP_MACHINE_SPECIFIC_PATH))
