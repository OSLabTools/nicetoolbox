"""Time cell conversion for ELAN tab-delimited files.

ELAN exports time columns either as float seconds (2.443) or as a timecode
(00:00:02.443), depending on the export settings. Reading is liberal (both accepted);
writing always emits timecode.
"""

from ...configs.models.video_timestamp import timestamp_to_ms


def parse_time_cell(cell: str) -> float:
    """Parse a time column value in seconds, accepting float or HH:MM:SS.mmm timecode."""
    cell = cell.strip()
    if not cell:
        raise ValueError("Empty time cell")

    try:
        return float(cell)
    except ValueError:
        pass

    # fps is unused when the value is a timestamp string, but the signature requires it.
    return timestamp_to_ms(cell, fps=1) / 1000.0


def format_timecode(seconds: float) -> str:
    """Format seconds as an ELAN timecode HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
