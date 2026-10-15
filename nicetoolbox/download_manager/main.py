import argparse
import logging
from pathlib import Path

from ..detectors import config_handler as confh
from ..utils import logging_utils as log_ut
from .manager import AssetManager


def entry_point():
    """
    CLI Entry point for the Asset Manager.
    """
    parser = argparse.ArgumentParser(description="NICE Toolbox On-Demand Asset Downloader")

    parser.add_argument(
        "--project_folder_path",
        default=Path("."),
        type=Path,
        required=False,
        help="Path to the NICE Toolbox project folder containing nice_project.toml config",
    )
    parser.add_argument(
        "--machine_specifics",
        default=Path("machine_specific_paths.toml"),
        type=Path,
        required=False,
        help="Path to machine_specific_paths.toml config",
    )
    parser.add_argument(
        "--run_config",
        default="<configs_folder_path>/detectors_run_file.toml",
        type=Path,
        required=False,
        help="Path to detectors_run_file.toml, supports placeholders",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--algorithms", nargs="+", help="Download assets for specific algorithms (e.g., eth_xgaze hrnetw48)"
    )
    group.add_argument("--all", action="store_true", help="Download all available assets in the manifest")

    args = parser.parse_args()

    log_ut.init_console_logging()
    config = confh.Configuration(args.project_folder_path, args.machine_specifics, args.run_config)
    logging.info(f"Initialising asset manager; project: '{args.project_folder_path}', run_config: '{args.run_config}'")
    manager = AssetManager(config)

    if args.all:
        manager.verify_and_download(list(manager.manifest.keys()))
    else:  # explicit algorithms or from config
        manager.ensure_assets_for_config(args.algorithms)


if __name__ == "__main__":
    entry_point()
