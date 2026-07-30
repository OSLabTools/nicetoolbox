import json
import logging
from pathlib import Path

import numpy as np

from ...configs.models.video_timestamp import timestamp_to_frame_index, timestamp_to_ms
from ...utils import logging_utils as log_ut
from ...utils.to_csv import convert_npz_to_csv_files
from ...utils.video import json_to_video_info, probe_video
from ..config_handler import ConnectorConfigHandler
from .elan_configs import (
    TRANSCRIPTION_4COL,
    ElanExportTranscriptionConfig,
    ElanImportGazeConfig,
    ElanImportTranscriptionConfig,
    ElanTranscriptionSequence,
)
from .elan_data import ElanData
from .elan_parser import parse_elan_file
from .elan_processing import VideoMeta, trim_tiers, validate_video_alignment
from .elan_writer import write_elan_txt
from .gaze_parser import eyes_npz_to_gaze
from .labeling_data import rename_subjects
from .labeling_from_elan import elan_data_to_hierarchical
from .npz_schema import schema_from_data
from .toolbox_writer import hierarchical_to_npz_dict
from .transcript_elan import tiers_to_transcript, transcript_to_tiers


def import_gaze(
    project_folder_path: Path,
    machine_specifics: Path,
    connector_config: Path,
) -> None:
    handler = ConnectorConfigHandler(
        project_folder_path,
        machine_specifics,
        connector_config,
        ElanImportGazeConfig,
    )
    cfg = handler.connector_config

    log_ut.log_main_banner("ELAN CONNECTOR: import_gaze")
    logging.info(f"Project path: '{handler.project_folder}'")
    logging.info(f"Run config: '{connector_config}'")
    logging.info(f"Sequences: {list(cfg.run)}")

    for sequence_id, sequence in cfg.run.items():
        log_ut.log_banner(f"Sequence: {sequence_id}")
        logging.info(f"Input:  {sequence.input}")
        logging.info(f"Output: {sequence.output}")

        video_info = json_to_video_info(probe_video(str(sequence.video)))
        video_meta = VideoMeta(fps=video_info.fps, duration_sec=video_info.duration_in_sec)
        logging.info(f"Video: fps={video_meta.fps}, duration={video_meta.duration_sec:.3f}s")

        fps = video_meta.fps
        start_sec = timestamp_to_frame_index(sequence.start, fps) / fps
        end_sec = video_meta.duration_sec if sequence.end == -1 else timestamp_to_frame_index(sequence.end, fps) / fps

        elan_data = parse_elan_file(sequence.input)
        validate_video_alignment(elan_data, video_meta)

        logging.info(f"Trimming to [{start_sec:.3f}s, {end_sec:.3f}s]...")
        trimmed = trim_tiers(elan_data, start_sec, end_sec)

        hier_data = elan_data_to_hierarchical(trimmed)
        logging.info(f"Converted: {hier_data}")

        if cfg.subjects:
            hier_data = rename_subjects(hier_data, cfg.subjects)

        schema = schema_from_data(hier_data)
        logging.info(f"Schema: {schema}")

        eyes_npz = hierarchical_to_npz_dict(
            hier_data,
            schema,
            fps,
            start_sec,
            end_sec,
            serialize="boolean",
            category_gap_fills={},
            reset_frames=sequence.reset_frames,
        )

        gaze_npz = eyes_npz_to_gaze(eyes_npz)

        sequence.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(sequence.output, **gaze_npz)
        logging.info(f"Saved gaze NPZ to: {sequence.output}")

        if cfg.export_csv:
            convert_npz_to_csv_files(sequence.output, sequence.output.parent)


def _window_seconds(sequence: ElanTranscriptionSequence) -> tuple[float, float]:
    """Resolve the sequence's [start, end) window in seconds.

    Transcription intervals are time-based, so an integer is read as seconds directly (unlike the
    frame-indexed gaze path). end = -1 means "until the end of the recording".
    """

    def to_seconds(value: int | str) -> float:
        # fps is unused when the value is a timestamp string.
        return timestamp_to_ms(value, fps=1) / 1000.0 if isinstance(value, str) else float(value)

    start_sec = to_seconds(sequence.start)
    end_sec = float("inf") if sequence.end == -1 else to_seconds(sequence.end)
    return start_sec, end_sec


def export_transcription(
    project_folder_path: Path,
    machine_specifics: Path,
    connector_config: Path,
) -> None:
    """Convert transcription JSON detector output into ELAN txt files for annotator correction."""
    handler = ConnectorConfigHandler(
        project_folder_path,
        machine_specifics,
        connector_config,
        ElanExportTranscriptionConfig,
    )
    cfg = handler.connector_config

    log_ut.log_main_banner("ELAN CONNECTOR: export_transcription")
    logging.info(f"Project path: '{handler.project_folder}'")
    logging.info(f"Run config: '{connector_config}'")
    logging.info(f"Sequences: {list(cfg.run)}")
    logging.info(f"Word tiers: {cfg.include_words}, speaker tiers: {cfg.include_speaker}")

    for sequence_id, sequence in cfg.run.items():
        log_ut.log_banner(f"Sequence: {sequence_id}")
        logging.info(f"Input:  {sequence.input}")
        logging.info(f"Output: {sequence.output}")

        with open(sequence.input) as f:
            transcript = json.load(f)
        logging.info(f"Tracks: {list(transcript)}")

        tiers = transcript_to_tiers(
            transcript,
            cfg.include_words,
            cfg.include_speaker,
            cfg.include_speaker_segments,
            cfg.mixed_segment_strategy,
        )

        start_sec, end_sec = _window_seconds(sequence)
        if start_sec > 0 or end_sec != float("inf"):
            logging.info(f"Trimming to [{start_sec:.3f}s, {end_sec:.3f}s]...")
            tiers = trim_tiers(ElanData(None, tiers), start_sec, end_sec).tiers

        write_elan_txt(sequence.output, tiers, TRANSCRIPTION_4COL)


def import_transcription(
    project_folder_path: Path,
    machine_specifics: Path,
    connector_config: Path,
) -> None:
    """Convert annotator-corrected ELAN txt files back into transcription JSON."""
    handler = ConnectorConfigHandler(
        project_folder_path,
        machine_specifics,
        connector_config,
        ElanImportTranscriptionConfig,
    )
    cfg = handler.connector_config

    log_ut.log_main_banner("ELAN CONNECTOR: import_transcription")
    logging.info(f"Project path: '{handler.project_folder}'")
    logging.info(f"Run config: '{connector_config}'")
    logging.info(f"Sequences: {list(cfg.run)}")

    for sequence_id, sequence in cfg.run.items():
        log_ut.log_banner(f"Sequence: {sequence_id}")
        logging.info(f"Input:  {sequence.input}")
        logging.info(f"Output: {sequence.output}")

        elan_data = parse_elan_file(sequence.input, TRANSCRIPTION_4COL)

        start_sec, end_sec = _window_seconds(sequence)
        if start_sec > 0 or end_sec != float("inf"):
            logging.info(f"Trimming to [{start_sec:.3f}s, {end_sec:.3f}s]...")
            elan_data = trim_tiers(elan_data, start_sec, end_sec)

        transcript = tiers_to_transcript(elan_data.tiers)
        logging.info(f"Tracks: {list(transcript)}")

        sequence.output.parent.mkdir(parents=True, exist_ok=True)
        with open(sequence.output, "w") as f:
            json.dump(transcript, f, indent=4)
        logging.info(f"Saved transcription JSON to: {sequence.output}")
