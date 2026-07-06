from pathlib import Path

from pydantic import BaseModel, Field

from nicetoolbox.configs.config_loader import ConfigLoader
from nicetoolbox.configs.placeholders import get_placeholders, resolve_placeholders
from nicetoolbox.configs.schemas.dataset_properties import DatasetProperties
from nicetoolbox.configs.schemas.detectors_config import DetectorsConfig
from nicetoolbox.configs.schemas.detectors_run_file import DetectorsRunFile
from nicetoolbox.configs.schemas.machine_specific_paths import MachineSpecificConfig
from nicetoolbox.configs.schemas.project_config import ProjectConfig
from nicetoolbox.configs.utils import default_runtime_placeholders, dict_to_model
from nicetoolbox.detectors.config_handler import Configuration

PROJECT_FOLDER = Path(".")

auto_mock = {
    "git_hash": "ffffffff",
    "commit_message": "special commit message",
    "me": "user-name",
    "yyyymmdd": "2012-12-12",
    "today": "2012-12-12",
    "time": "16_12",
    "pwd": "home/nicetoolbox",
}
runtime_mock = default_runtime_placeholders()


def test_load_detectors_config():
    """Test if we can load detectors config setup"""
    cfg_loader = ConfigLoader(auto_mock, runtime_mock)

    # inject project folder path so <project_folder_path> placeholder is available
    cfg_loader.extend_global_ctx({"project_folder_path": str(PROJECT_FOLDER.resolve())})

    # project config (registers <configs_folder_path>, <datasets_folder_path>, <output_folder_path>)
    project_config = cfg_loader.load_config(PROJECT_FOLDER / "nice_project.toml", ProjectConfig)
    cfg_loader.extend_global_ctx(project_config)

    # machine specific
    machine_specific = cfg_loader.load_config(Path("machine_specific_paths.toml"), MachineSpecificConfig)
    cfg_loader.extend_global_ctx(machine_specific)

    # run file (or vis_config/evaluation_config)
    run_file = cfg_loader.load_config(Path("configs/detectors_run_file.toml"), DetectorsRunFile)
    # we register only [io] part of run_file
    cfg_loader.extend_global_ctx(run_file.io)

    # detectors and dataset are not added to global context
    detectors = cfg_loader.load_config(Path("configs/detectors_config.toml"), DetectorsConfig)
    dataset = cfg_loader.load_config(Path("configs/dataset_properties.toml"), DatasetProperties)

    # check that we resolved all placeholders
    assert get_placeholders(project_config) <= runtime_mock
    assert get_placeholders(machine_specific) <= runtime_mock
    assert get_placeholders(run_file) <= runtime_mock
    assert get_placeholders(detectors) <= runtime_mock
    assert get_placeholders(dataset) <= runtime_mock

    # simulate runtime resolution
    for sequence_ID in ["sequence_1", "sequence_2"]:
        ctx = {"cur_sequence_ID": sequence_ID}
        res_dataset = cfg_loader.resolve(dataset, ctx)
        example_field = res_dataset["communication_multiview"].data_input_folder
        assert sequence_ID in str(example_field)


def test_runtime_sequence_context_composition():
    """
    Exercise the full runtime composition path.

    Load all configs via `Configuration`, then iterate `iter_sequence_contexts()`
    which builds `SequenceRuntimeConfig` and resolves it via case 3 (BaseModel).
    Catches cross-schema field-name collisions and other issues that only surface
    on the composed model — invisible to per-schema unit tests and per-file TOML
    load tests.
    """
    config = Configuration(
        project_folder=PROJECT_FOLDER,
        machine_specifics_file=Path("machine_specific_paths.toml"),
        run_config_file=Path("configs/detectors_run_file.toml"),
    )
    contexts = list(config.iter_sequence_contexts())
    assert len(contexts) >= 1
    for ctx in contexts:
        # Only later-injected runtime placeholders (cur_algorithm_name,
        # cur_component_name, ...) may remain unresolved at this stage.
        assert get_placeholders(ctx) <= runtime_mock


def test_load_config_default_placeholder_not_resolved():
    """
    Regression test: placeholders embedded in Pydantic field *defaults* are not
    resolved by ConfigLoader.load_config.

    load_config resolves placeholders on the raw TOML dict, then calls
    dict_to_model to construct the Pydantic model.  Fields absent from the TOML
    are filled in by Pydantic from their Python defaults *after* resolution has
    already run, so any placeholder in a default (e.g.
    ``Field(default=Path("<base_dir>/result"))``) is never seen by the resolver.

    The fix requires a second resolve() pass on the constructed model inside
    load_config.  Until that fix lands, the assertion below documents the broken
    behaviour.
    """

    class ConfigWithDefault(BaseModel):
        name: str
        path: Path = Field(default=Path("<base_dir>/result"))

    # Simulate what load_config does: resolve the raw dict, then construct model.
    raw = {"name": "<app_name>"}  # 'path' intentionally absent — default will apply
    placeholders = {"app_name": "MyApp", "base_dir": "/data"}

    resolved_raw = resolve_placeholders(raw, placeholders)
    model = dict_to_model(resolved_raw, ConfigWithDefault)

    assert model.name == "MyApp"
    # BUG: path default was never seen by the resolver — placeholder survives.
    assert model.path == Path("<base_dir>/result")  # remove once bug is fixed
    assert model.path != Path("/data/result")  # remove once bug is fixed
