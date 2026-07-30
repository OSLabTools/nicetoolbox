"""
Writing ELAN tab-delimited txt files.
"""

import logging
from pathlib import Path

from .elan_configs import ElanColumnSpec
from .elan_data import Tier
from .elan_time import format_timecode


def sanitize_annotation(text: str) -> str:
    """Replace tabs/newlines in annotation text with spaces so they cannot break the layout.

    The tab-delimited format has no quoting, so a literal tab inside free-form transcript text
    would shift every following column on that line.
    """
    if "\t" in text or "\n" in text or "\r" in text:
        logging.warning(f"Annotation contains tab/newline characters, replaced with spaces: {text!r}")
        text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return text


def write_elan_txt(file_path: Path, tiers: list[Tier], spec: ElanColumnSpec) -> None:
    """Write tiers as an ELAN tab-delimited txt file using the given column layout."""
    lines: list[str] = []
    n_intervals = 0
    for tier in tiers:
        for interval in tier.intervals:
            fields = [""] * spec.width
            fields[spec.tier] = tier.tier_name
            fields[spec.start] = format_timecode(interval.start_sec)
            fields[spec.end] = format_timecode(interval.end_sec)
            fields[spec.annotation] = sanitize_annotation(interval.annotation)
            lines.append("\t".join(fields))
            n_intervals += 1

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(f"Wrote {len(tiers)} tier(s), {n_intervals} interval(s) to: {file_path}")
