"""
Result containers and schemas for evaluation metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
import torch

from ..data.discovery import ChunkWorkItem, FrameInfo


# (1) Container during metric processing (update calls)
@dataclass(frozen=True)
class BatchResult:
    """Results container during batch evaluations. Further processing needed."""

    results_tensor: torch.Tensor
    results_description: List[str]
    meta_chunk: ChunkWorkItem
    meta_frames: List[FrameInfo]


# (2) Container for summary metrics after processing batches (compute calls)
@dataclass(frozen=True)
class AggregatedResult:
    """Result container for final summary metrics. No further processing needed."""

    value: float
    metric_type: str
    metric_name: str
    component: str
    algorithm: str
    person: str
    camera: str


# Metrics can be frame based or aggregated summaries (jpe vs accuracy)
MetricReturnType = Union[List[BatchResult], AggregatedResult]


# (1.1) Container for unpacking BatchResults into single frame results -> FrameResult
@dataclass(frozen=True)
class FrameResult:
    """Result container for final frame based metrics"""

    value: np.ndarray
    # (1) Used for generating file path
    metric_name: str
    metric_type: str
    session: str
    sequence: str
    component: str
    algorithm: str
    # (2) Used to re-grid to ndarray
    person: str
    camera: str
    frame: int
    # (3) Used for building data description
    axis3_description: List[str]


# (1.2) FrameResults are transformed to a ResultGrid, same structure as detector output
@dataclass
class ResultGrid:
    metric_name: str  # E.g. jpe, bone_length, ...
    values: np.ndarray  # Shape: [#person x #camera x #frames x #data]
    description: Dict[str, Any]  # {"axis0": ["person"], ..., "axis3": ["data"]}


# (1.3) One ResultFileGroup contains all ResultGrids (metrics) of a metric type
@dataclass
class ResultFileGroup:
    # (1) Meta data for file path generation
    session: str
    sequence: str
    component: str
    algorithm: str
    metric_type: str  # E.g. categorical metrics
    # (2) Entries of a single NPZ file
    grids: List[ResultGrid] = field(default_factory=list)  # E.g. acc, precision, ...
