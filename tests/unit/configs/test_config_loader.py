from nicetoolbox.configs.config_loader import ConfigLoader
from nicetoolbox.configs.placeholders import get_placeholders
from nicetoolbox.configs.schemas.dataset_properties import DatasetProperties
from nicetoolbox.configs.schemas.detectors_config import DetectorsConfig
from nicetoolbox.configs.schemas.detectors_run_file import DetectorsRunFile
from nicetoolbox.configs.schemas.machine_specific_paths import MachineSpecificConfig
from nicetoolbox.configs.utils import default_runtime_placeholders

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

    # machine specific
    machine_specific = cfg_loader.load_config(
        "machine_specific_paths.toml", MachineSpecificConfig
    )
    # each config has access to machine config (i.e. <datasets_folder_path>)
    # so we register all fields in global context
    cfg_loader.extend_global_ctx(machine_specific)

    # run file (or vis_config/evaluation_config)
    run_file = cfg_loader.load_config(
        "configs/detectors_run_file.toml", DetectorsRunFile
    )
    # we register only [io] part of run_file
    cfg_loader.extend_global_ctx(run_file.io)

    # detectors dataser are not added to globabl context
    detectors = cfg_loader.load_config("configs/detectors_config.toml", DetectorsConfig)
    dataset = cfg_loader.load_config(
        "configs/dataset_properties.toml", DatasetProperties
    )

    # check that we resolved all placeholders
    assert get_placeholders(machine_specific) <= runtime_mock
    assert get_placeholders(run_file) <= runtime_mock
    assert get_placeholders(detectors) <= runtime_mock
    assert get_placeholders(dataset) <= runtime_mock

    # simullate runtime resolution
    for session_ID in ["sesion_1", "sesion_2"]:
        ctx = {"cur_session_ID": session_ID}
        res_dataset = cfg_loader.resolve(dataset, ctx)
        example_field = res_dataset["communication_multiview"].data_input_folder
        assert session_ID in str(example_field)
