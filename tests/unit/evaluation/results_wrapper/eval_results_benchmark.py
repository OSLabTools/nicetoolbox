"""
Utilities for generating a hierarchical tree of dummy evaluation result files.

This module is used for testing and demonstrating the EvaluationResults API.
It creates a folder structure populated with .npz files that mimic real-world
heterogeneous evaluation data, including different metrics, coordinates, and
experimental setups.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_DATASETS = ["dataset_A", "dataset_B"]
DEFAULT_ALGORITHMS = ["hrnet", "vitpose"]
DEFAULT_COMPONENTS = ["body_joints"]
DEFAULT_METRIC_NAMES = ("jpe", "pck")
DEFAULT_PERSONS = ["p1", "p2"]
DEFAULT_CAMERAS = ["c1", "c2", "c3", "c4"]
DEFAULT_NUM_FRAMES = 10000

# Default label sets per algorithm
DEFAULT_LABELS = {
    "hrnet": [
        "nose",
        "left_eye",
        "right_eye",
        "right_shoulder",
        "left_shoulder",
        *[f"body_joint_{i}" for i in range(20)],
    ],
    "vitpose": [
        "nose",
        "left_eye",
        "right_eye",
        "left_knee",
        "right_knee",
        *[f"body_joint_{i}" for i in range(20)],
    ],
}

# Default camera error distributions (Beta distribution parameters)
DEFAULT_CAMERA_DISTRIBUTIONS = {"c1": (2, 5), "c2": (5, 5), "c3": (5, 2), "c4": (2, 2)}

# Default sequence configurations
DEFAULT_SEQUENCE_CONFIG = {
    "dataset_A": {
        "run_01": {"persons": ["p1", "p2"], "cameras": ["c1", "c2", "c3", "c4"]},
        "run_02": {"persons": ["p1", "p2"], "cameras": ["c1", "c2"]},
        "run_03": {"persons": ["p1"], "cameras": ["c3", "c4"]},
        "run_04": {"persons": ["p2"], "cameras": ["c2", "c4"]},
    },
    "dataset_B": {
        "scene_A": {"persons": ["p1"], "cameras": ["c1", "c3"]},
        "scene_B": {"persons": ["p1", "p2"], "cameras": ["c2"]},
        "scene_C": {"persons": ["p2"], "cameras": ["c1", "c4"]},
        "scene_D": {"persons": ["p1", "p2"], "cameras": ["c3", "c4"]},
    },
}


# ============================================================================
# Configuration Dataclass
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for generating benchmark evaluation data."""

    random_seed: int = 42

    # Hierarchical structure (Meta data)
    datasets: List[str] = field(default_factory=lambda: DEFAULT_DATASETS.copy())
    algorithms: List[str] = field(default_factory=lambda: DEFAULT_ALGORITHMS.copy())
    components: List[str] = field(default_factory=lambda: DEFAULT_COMPONENTS.copy())
    metric_names: Tuple[str, ...] = DEFAULT_METRIC_NAMES
    sequence_config: Dict[str, Dict[str, Dict[str, List[str]]]] = field(
        default_factory=lambda: {
            ds: {seq: {k: v.copy() for k, v in cfg.items()} for seq, cfg in seqs.items()}
            for ds, seqs in DEFAULT_SEQUENCE_CONFIG.items()
        }
    )

    # Inside npz configurations
    camera_distributions: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_CAMERA_DISTRIBUTIONS.copy()
    )
    labels: Dict[str, List[str]] = field(default_factory=lambda: {k: v.copy() for k, v in DEFAULT_LABELS.items()})
    num_frames: int = DEFAULT_NUM_FRAMES
    # CHANGED: Add frame range for randomization
    frame_range: Optional[Tuple[int, int]] = None


# ============================================================================
# Core Functions
# ============================================================================


def create_dummy_npz(
    path: Path,
    shape: Tuple[int, ...],
    labels: List[str],
    persons: List[str],
    cameras: List[str],
    metric_names: Tuple[str, ...],
    camera_distributions: Dict[str, Tuple[float, float]],
    random_seed: int = 42,
) -> None:
    """Creates a single dummy .npz file with heterogeneous camera error distributions.

    Args:
        path: Output file path.
        shape: Shape of the data array (persons, cameras, frames, labels).
        labels: List of label names (e.g., joint names).
        persons: List of person identifiers.
        cameras: List of camera identifiers.
        metric_names: Tuple of metric names to generate.
        camera_distributions: Mapping from camera name to Beta distribution params.
        random_seed: Random seed for reproducibility.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    default_params = (3, 3)
    arrays = {}
    descs = {}

    for name in metric_names:
        full_array = np.empty(shape, dtype=np.float32)
        slice_shape = (shape[0], shape[2], shape[3])

        for i, cam_name in enumerate(cameras):
            a, b = camera_distributions.get(cam_name, default_params)
            camera_slice = rng.beta(a, b, size=slice_shape).astype(np.float32)
            full_array[:, i, :, :] = camera_slice

        arrays[name] = full_array
        descs[name] = {
            "dims": ["person", "camera", "frame", "label"],
            "axis0": persons,
            "axis1": cameras,
            "axis2": [f"frame_{i}" for i in range(shape[2])],
            "axis3": labels,
        }

    arrays["data_description"] = np.array(descs, dtype=object)
    np.savez(path, **arrays)


def setup_demo_tree(root: Path, config: Optional[BenchmarkConfig] = None) -> None:
    """Creates a hierarchical directory of dummy .npz evaluation files.

    Args:
        root: Root directory where the tree will be created.
        config: Configuration object. If None, uses defaults.
    """
    if config is None:
        config = BenchmarkConfig()

    root.mkdir(parents=True, exist_ok=True)

    for dataset in config.datasets:
        if dataset not in config.sequence_config:
            continue

        for seq, seq_cfg in config.sequence_config[dataset].items():
            for alg in config.algorithms:
                if alg not in config.labels:
                    labels = [f"label_{i}" for i in range(17)]
                else:
                    labels = config.labels[alg]

                for comp in config.components:
                    persons = seq_cfg.get("persons", DEFAULT_PERSONS)
                    cameras = seq_cfg.get("cameras", DEFAULT_CAMERAS)

                    if config.frame_range is not None:
                        min_frames, max_frames = config.frame_range
                        num_frames = np.random.randint(min_frames, max_frames + 1)
                    else:
                        num_frames = seq_cfg.get("num_frames", config.num_frames)

                    shape = (len(persons), len(cameras), num_frames, len(labels))
                    path = root / f"{dataset}__{seq}" / comp / f"{alg}__keypoint_metrics.npz"

                    create_dummy_npz(
                        path=path,
                        shape=shape,
                        labels=labels,
                        persons=persons,
                        cameras=cameras,
                        metric_names=config.metric_names,
                        camera_distributions=config.camera_distributions,
                        random_seed=config.random_seed,
                    )

    print(f"Benchmark data tree created in: {root}")


# ============================================================================
# Convenience Function
# ============================================================================


def create_simple_benchmark(
    root: Path,
    num_datasets: int = 2,
    num_sequences_per_dataset: int = 4,
    num_frames: int = 10000,
    num_persons: int = 2,
    num_cameras: int = 4,
    frame_range: Optional[Tuple[int, int]] = None,  # CHANGED: Add frame_range param
) -> None:
    """Creates a simple uniform benchmark with specified dimensions.

    Args:
        root: Root directory for the benchmark.
        num_datasets: Number of datasets to create.
        num_sequences_per_dataset: Number of sequences per dataset.
        num_frames: Number of frames per sequence (ignored if frame_range is set).
        num_persons: Number of persons per sequence.
        num_cameras: Number of cameras per sequence.
        frame_range: Optional tuple (min_frames, max_frames) for random frame counts.
    """
    config = BenchmarkConfig(num_frames=num_frames, frame_range=frame_range)

    config.datasets = [f"dataset_{i}" for i in range(num_datasets)]
    config.sequence_config = {}

    persons = [f"p{i}" for i in range(num_persons)]
    cameras = [f"c{i}" for i in range(num_cameras)]

    for dataset in config.datasets:
        config.sequence_config[dataset] = {
            f"seq_{j:02d}": {"persons": persons, "cameras": cameras} for j in range(num_sequences_per_dataset)
        }

    setup_demo_tree(root, config)


if __name__ == "__main__":
    # Example 1: Use defaults
    setup_demo_tree(Path("./benchmark_data_default"))

    # Example 2: Custom configuration with random frame counts
    custom_config = BenchmarkConfig(
        datasets=["my_dataset"],
        algorithms=["openpose"],
        labels={"openpose": [f"joint_{i}" for i in range(25)]},
        frame_range=(5000, 15000),
    )
    custom_config.sequence_config = {
        "my_dataset": {"recording_01": {"persons": ["subject_1"], "cameras": ["cam_front"]}}
    }
    setup_demo_tree(Path("./benchmark_data_custom"), custom_config)

    # Example 3: Simple uniform benchmark with random frames
    create_simple_benchmark(
        Path("./benchmark_data_simple"),
        num_datasets=1,
        num_sequences_per_dataset=2,
        num_frames=10000,
    )
