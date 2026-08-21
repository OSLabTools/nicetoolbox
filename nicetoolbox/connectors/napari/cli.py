"""napari_connector CLI entry point."""

import argparse
from pathlib import Path

from .tasks import export_body_joints, import_body_joints

DEFAULT_IMPORT_BODY_JOINTS = Path("<configs_folder_path>/connectors/napari_import_body_joints.toml")
DEFAULT_EXPORT_BODY_JOINTS = Path("<configs_folder_path>/connectors/napari_export_body_joints.toml")

TASKS = {
    "import_body_joints": (import_body_joints, DEFAULT_IMPORT_BODY_JOINTS),
    "export_body_joints": (export_body_joints, DEFAULT_EXPORT_BODY_JOINTS),
}


def entry_point() -> None:
    parser = argparse.ArgumentParser(prog="napari_connector", description="NICE napari connector")
    sub = parser.add_subparsers(dest="task", required=True)

    for name in TASKS:
        p = sub.add_parser(name)
        p.add_argument("--project_folder_path", default=Path("."), type=Path)
        p.add_argument("--machine_specifics", default=Path("machine_specific_paths.toml"), type=Path)
        p.add_argument("--connector_config", type=Path)

    args = parser.parse_args()

    task_fn, default_config = TASKS[args.task]
    task_fn(args.project_folder_path, args.machine_specifics, args.connector_config or default_config)


if __name__ == "__main__":
    entry_point()
