"""Parsing ELAN tab-delimited txt export files."""

import logging
import re
from pathlib import Path

from .elan_configs import GAZE_9COL, ElanColumnSpec
from .elan_data import ElanData, ElanHeader, Interval, Tier
from .elan_time import parse_time_cell

# "ms per sample" is optional: ELAN omits it for audio-only media.
_HEADER_LINE_RE = re.compile(
    r'"#file:///(?P<path>.+?)\s+--\s+'
    r"offset:\s*(?P<offset>\d+),\s+"
    r"duration:\s*\S+\s*/\s*\S+\s*/\s*(?P<duration_ms>\d+)"
    r"(?:,\s+ms per sample:\s*(?P<ms_per_sample>[\d.]+))?"
)


def parse_elan_file(file_path: Path, spec: ElanColumnSpec = GAZE_9COL) -> ElanData:
    if file_path.suffix == ".eaf":
        raise NotImplementedError(".eaf files aren't supported, please use ELAN .txt export")
    if file_path.suffix != ".txt":
        raise ValueError(f"Expected a .txt file, got: {file_path}")

    logging.info(f"Reading ELAN file: {file_path}")
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    logging.info("Trying to read ELAN header...")
    header = parse_header(lines)
    if header is not None:
        logging.info(f"Found {header}")
        data_start_line = header.data_start_line
    else:
        logging.warning("No ELAN header found!")
        data_start_line = 0

    logging.info("Parsing ELAN data...")
    tiers = parse_tiers(lines, data_start_line, spec)
    data = ElanData(header, tiers)
    logging.info(f"Parsed {data}")
    return data


def parse_header(lines: list[str]) -> ElanHeader | None:
    media_files: list[str] = []
    ms_per_samples: list[float] = []
    offsets: list[int] = []
    durations_ms: list[int] = []
    data_start_line = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"#'):
            match = _HEADER_LINE_RE.search(stripped)
            if not match:
                raise ValueError(f"Header line {i + 1} does not match expected format: {stripped[:120]}")
            media_files.append(match.group("path"))
            if match.group("ms_per_sample") is not None:
                ms_per_samples.append(float(match.group("ms_per_sample")))
            offsets.append(int(match.group("offset")))
            durations_ms.append(int(match.group("duration_ms")))
        elif stripped == "":
            continue
        else:
            data_start_line = i
            break

    if not media_files:
        return None

    if len(set(ms_per_samples)) > 1:
        raise ValueError(f"Inconsistent ms_per_sample across header lines: {ms_per_samples}")
    if len(set(offsets)) > 1:
        raise ValueError(f"Inconsistent offset across header lines: {offsets}")
    if len(set(durations_ms)) > 1:
        raise ValueError(f"Inconsistent duration across header lines: {durations_ms}")

    ms_per_sample = ms_per_samples[0] if ms_per_samples else None
    return ElanHeader(ms_per_sample, offsets[0], durations_ms[0], media_files, data_start_line)


def parse_tiers(lines: list[str], start_idx: int, spec: ElanColumnSpec = GAZE_9COL) -> list[Tier]:
    """Parse interval records into tiers using the given column layout."""
    tier_ivs: dict[str, list[Interval]] = {}

    for offset, line in enumerate(lines[start_idx:]):
        line_no = start_idx + offset + 1
        if not line.strip():
            continue

        fields = line.rstrip("\r\n").split("\t")
        if len(fields) < spec.width:
            raise ValueError(
                f"Line {line_no}: expected {spec.width} tab-separated fields, got {len(fields)}. "
                f"Check the ELAN export column layout against the configured spec: {spec!r}. Line: {line!r}"
            )
        if len(fields) > spec.width:
            raise ValueError(
                f"Line {line_no} (tier '{fields[spec.tier]}'): got {len(fields)} tab-separated fields, "
                f"expected {spec.width}. A literal tab inside the annotation text would cause this."
            )

        tier_name = fields[spec.tier]
        start_sec = parse_time_cell(fields[spec.start])
        end_sec = parse_time_cell(fields[spec.end])
        annotation = fields[spec.annotation].strip()

        if tier_name not in tier_ivs:
            tier_ivs[tier_name] = []
        tier_ivs[tier_name].append(Interval(start_sec, end_sec, annotation))

    return [Tier(tier_name, intervals) for tier_name, intervals in tier_ivs.items()]
