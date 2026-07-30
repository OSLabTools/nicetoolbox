"""Data models for ELAN to Toolbox conversion."""

from dataclasses import dataclass


@dataclass
class ElanHeader:
    # None when no media line carries "ms per sample" (audio-only media omit it).
    ms_per_sample: float | None
    offset: int
    duration_ms: int
    media_files: list[str]
    data_start_line: int

    def __str__(self) -> str:
        ms = "unknown" if self.ms_per_sample is None else f"{self.ms_per_sample:.4f}"
        return (
            f"ElanHeader: ms_per_sample={ms}, "
            f"duration={self.duration_ms:.3f}ms, "
            f"offset={self.offset}, media files: {self.media_files}"
        )


@dataclass
class Interval:
    start_sec: float
    end_sec: float
    annotation: str


@dataclass
class Tier:
    tier_name: str
    intervals: list[Interval]


@dataclass
class ElanData:
    header: ElanHeader | None
    tiers: list[Tier]

    @property
    def tier_names(self) -> set[str]:
        return {tier.tier_name for tier in self.tiers}

    @property
    def labels(self) -> set[str]:
        return {interval.annotation for tier in self.tiers for interval in tier.intervals}

    def __str__(self) -> str:
        tier_count = len(self.tiers)
        interval_count = sum(len(tier.intervals) for tier in self.tiers)
        return f"ElanData: {tier_count} tiers, {interval_count} intervals"
