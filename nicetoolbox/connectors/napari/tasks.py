import logging
from pathlib import Path

from ...utils import logging_utils as log_ut
from ...utils.to_csv import convert_npz_to_csv_files
from ..config_handler import ConnectorConfigHandler
from .napari_configs import NapariExportBodyJointsConfig, NapariImportBodyJointsConfig
from .parser import napari_to_body_joints_npz
from .writer import body_joints_npz_to_napari


def import_body_joints(
    project_folder_path: Path,
    machine_specifics: Path,
    connector_config: Path,
) -> None:
    handler = ConnectorConfigHandler(
        project_folder_path,
        machine_specifics,
        connector_config,
        NapariImportBodyJointsConfig,
    )
    cfg = handler.connector_config

    log_ut.log_main_banner("NAPARI CONNECTOR: import_body_joints")
    logging.info(f"Project path: '{handler.project_folder}'")
    logging.info(f"Run config: '{connector_config}'")
    logging.info(f"Sequences: {list(cfg.run)}")

    for sequence_id, sequence in cfg.run.items():
        log_ut.log_banner(f"Sequence: {sequence_id}")
        logging.info(f"Input:  {sequence.input}")
        logging.info(f"Output: {sequence.output}")

        napari_to_body_joints_npz(sequence, cfg)

        if cfg.export_csv:
            convert_npz_to_csv_files(sequence.output, sequence.output.parent)


def export_body_joints(
    project_folder_path: Path,
    machine_specifics: Path,
    connector_config: Path,
) -> None:
    """Convert body_joints NPZ detector output into napari annotation files for manual correction."""
    handler = ConnectorConfigHandler(
        project_folder_path,
        machine_specifics,
        connector_config,
        NapariExportBodyJointsConfig,
    )
    cfg = handler.connector_config

    log_ut.log_main_banner("NAPARI CONNECTOR: export_body_joints")
    logging.info(f"Project path: '{handler.project_folder}'")
    logging.info(f"Run config: '{connector_config}'")
    logging.info(f"Sequences: {list(cfg.run)}")
    logging.info(f"NPZ key: {cfg.npz_key}")

    for sequence_id, sequence in cfg.run.items():
        log_ut.log_banner(f"Sequence: {sequence_id}")
        logging.info(f"Input:  {sequence.input}")
        logging.info(f"Output: {sequence.output}")

        body_joints_npz_to_napari(sequence, cfg)
