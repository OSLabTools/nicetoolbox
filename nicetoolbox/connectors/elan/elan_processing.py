"""Data processing functions for ELAN."""

import logging
from dataclasses import dataclass

from .elan_data import ElanData, Interval, Tier


@dataclass
class VideoMeta:
    fps: float
    duration_sec: float


def validate_video_alignment(elan_data: ElanData, video_meta: VideoMeta) -> None:
    if not elan_data.header:
        logging.warning("No ELAN header found — cannot validate against video metadata. Using ffprobe values directly.")
        return

    header = elan_data.header

    if header.offset != 0:
        raise NotImplementedError(
            f"ELAN header has non-zero offset ({header.offset} ms). Non-zero offsets are not supported."
        )

    if header.ms_per_sample is None:
        logging.warning("ELAN header has no 'ms per sample' (audio-only media) — skipping FPS validation.")
    else:
        elan_fps = 1000.0 / header.ms_per_sample
        if abs(elan_fps - video_meta.fps) > 0.01:
            raise ValueError(
                f"FPS mismatch: ELAN header says {elan_fps:.4f} fps, but ffprobe says {video_meta.fps:.4f} fps"
            )
        logging.info(f"FPS validated: ELAN={elan_fps:.4f}, ffprobe={video_meta.fps:.4f}")

    elan_duration_sec = header.duration_ms / 1000.0
    if abs(elan_duration_sec - video_meta.duration_sec) > 1.0:
        raise ValueError(
            f"Duration mismatch: ELAN header says {elan_duration_sec:.3f}s, "
            f"but ffprobe says {video_meta.duration_sec:.3f}s"
        )
    logging.info(f"Duration validated: ELAN={elan_duration_sec:.3f}s, ffprobe={video_meta.duration_sec:.3f}s")


def trim_tiers(elan_data: ElanData, start_sec: float, end_sec: float) -> ElanData:
    trimmed_tiers = []
    dropped = 0
    clipped = 0

    for tier in elan_data.tiers:
        trimmed_intervals = []
        for iv in tier.intervals:
            if iv.start_sec >= end_sec or iv.end_sec <= start_sec:
                dropped += 1
                continue
            new_start = max(iv.start_sec, start_sec)
            new_end = min(iv.end_sec, end_sec)
            if new_start != iv.start_sec or new_end != iv.end_sec:
                clipped += 1
            trimmed_intervals.append(Interval(new_start, new_end, iv.annotation))
        trimmed_tiers.append(Tier(tier.tier_name, trimmed_intervals))

    if dropped > 0:
        logging.warning(f"Dropped {dropped} intervals outside [{start_sec:.3f}s, {end_sec:.3f}s].")
    if clipped > 0:
        logging.warning(f"Clipped {clipped} intervals to fit [{start_sec:.3f}s, {end_sec:.3f}s].")

    return ElanData(elan_data.header, trimmed_tiers)
