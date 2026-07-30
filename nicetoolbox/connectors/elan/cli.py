"""elan_connector CLI entry point."""

import argparse
from pathlib import Path

from .tasks import export_transcription, import_gaze, import_transcription

DEFAULT_IMPORT_GAZE = Path("<configs_folder_path>/connectors/elan_import_gaze.toml")
DEFAULT_EXPORT_TRANSCRIPTION = Path("<configs_folder_path>/connectors/elan_export_transcription.toml")
DEFAULT_IMPORT_TRANSCRIPTION = Path("<configs_folder_path>/connectors/elan_import_transcription.toml")

TASKS = {
    "import_gaze": (import_gaze, DEFAULT_IMPORT_GAZE),
    "export_transcription": (export_transcription, DEFAULT_EXPORT_TRANSCRIPTION),
    "import_transcription": (import_transcription, DEFAULT_IMPORT_TRANSCRIPTION),
}


def entry_point() -> None:
    parser = argparse.ArgumentParser(prog="elan_connector", description="NICE ELAN connector")
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
