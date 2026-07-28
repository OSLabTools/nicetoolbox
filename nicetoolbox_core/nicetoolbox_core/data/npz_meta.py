from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class NpzMeta(ABC):
    """Abstract base for all NPZ metadata types."""

    npz_path: Path
    npz_key: str

    # ============== Evaluation Helpers =======================

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return user-facing metadata fields as a flat dict."""
        ...

    @staticmethod
    @abstractmethod
    def always_iterate() -> frozenset[str]:
        """Return columns that must always get their own row in summaries (never pooled)."""
        ...

    @staticmethod
    @abstractmethod
    def comparable_dim() -> str | None:
        """Return the dimension used to compare results side-by-side (e.g. as series/color in charts)."""
        ...

    def align_key(self) -> tuple | None:
        """Return the key used to pair this array with its counterpart during alignment."""
        return None


# =============================================================================
# Meta subclasses
# =============================================================================


@dataclass
class SubsequenceInfo:
    subsequence_index: int
    video_start: int  # resolved frame index
    video_length: int  # resolved frame count


@dataclass
class ExperimentMeta(NpzMeta):
    """Experiment meta (npz loaded during/after detectors run)."""

    dataset: str
    sequence: str
    component: str
    algorithm: str
    subsequence: SubsequenceInfo

    @classmethod
    def always_iterate(cls) -> frozenset[str]:
        return frozenset({"component", "algorithm", "npz_key"})

    @staticmethod
    def comparable_dim() -> str | None:
        return "algorithm"

    def align_key(self) -> tuple:
        return (self.dataset, self.sequence, self.component)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sequence": self.sequence,
            "subsequence": self.subsequence.subsequence_index,
            "subsequence_start": self.subsequence.video_start,
            "subsequence_length": self.subsequence.video_length,
            "component": self.component,
            "algorithm": self.algorithm,
            "npz_key": self.npz_key,
        }


@dataclass
class AnnotationMeta(NpzMeta):
    """Annotation meta (dataset specific npz, usually labeled manually)."""

    dataset: str
    sequence: str
    component: str

    @classmethod
    def always_iterate(cls) -> frozenset[str]:
        return frozenset({"component", "npz_key"})

    @staticmethod
    def comparable_dim() -> str | None:
        return None

    def align_key(self) -> tuple:
        return (self.dataset, self.sequence, self.component)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sequence": self.sequence,
            "component": self.component,
            "npz_key": self.npz_key,
        }


@dataclass
class PathMeta(NpzMeta):
    """Npz loaded directly from path without any meta information."""

    @classmethod
    def always_iterate(cls) -> frozenset[str]:
        return frozenset({"npz_file_name", "npz_key"})

    @staticmethod
    def comparable_dim() -> str | None:
        return "npz_file_name"

    def to_dict(self) -> dict[str, Any]:
        return {
            "npz_file_name": self.npz_path.stem,
            "npz_key": self.npz_key,
        }
