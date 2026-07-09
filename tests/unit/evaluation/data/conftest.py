from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from nicetoolbox.configs.placeholders import resolve_placeholders
from nicetoolbox.evaluation.data.input_loader import (
    AnnotationMeta,
    ArrayAxes,
    ExperimentMeta,
    LoadedArray,
    NpzMeta,
    SubsequenceInfo,
)
from nicetoolbox.utils.config import save_config

# ---------------------------------------------------------------------------
# LoadedArray factory
# ---------------------------------------------------------------------------


def make_loaded_array(
    meta: NpzMeta,
    subjects: tuple[str, ...] = ("s1",),
    cameras: tuple[str, ...] = ("c1",),
    frames: tuple[str, ...] = ("f0",),
    labels: tuple[str, ...] = ("j1",),
    data: tuple[str, ...] = (),
) -> LoadedArray:
    """Build a LoadedArray with sequential integer data matching the given axis labels.

    Data values are sequential integers so tests can verify correct slices after intersection.
    """
    shape = (len(subjects), len(cameras), len(frames), len(labels))
    if data:
        shape += (len(data),)
    arr_data = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)
    axes = ArrayAxes(list(subjects), list(cameras), list(frames), list(labels), list(data))
    return LoadedArray(meta=meta, data=arr_data, axes=axes)


# ---------------------------------------------------------------------------
# NPZ factory
# ---------------------------------------------------------------------------


def make_npz(
    path: Path,
    key: str = "landmarks",
    subjects: tuple = ("s1", "s2"),
    cameras: tuple = ("cam1",),
    frames: tuple = ("f0", "f1", "f2"),
    labels: tuple = ("x", "y", "z"),
    data: tuple | None = None,
) -> Path:
    """Create a minimal well-formed npz file at *path* and return it.

    Data values are sequential integers so tests can verify correct slices.
    """
    descr: dict = {
        key: {
            "axis0": list(subjects),
            "axis1": list(cameras),
            "axis2": list(frames),
            "axis3": list(labels),
        }
    }
    shape = (len(subjects), len(cameras), len(frames), len(labels))
    if data is not None:
        descr[key]["axis4"] = list(data)
        shape = shape + (len(data),)

    npz_data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data_description=np.array(descr, dtype=object), **{key: npz_data})
    return path


# ---------------------------------------------------------------------------
# Shared dataset / algorithm constants
# ---------------------------------------------------------------------------


SINGLE_DATASET = {
    "ds1": {
        "fps": 30,
        "sequences": [{"sequence_id": "seq01"}],
    },
}

MULTI_DATASET = {
    "ds1": {
        "fps": 30,
        "sequences": [{"sequence_id": "seq01"}],
    },
    "ds2": {
        "fps": 60,
        "sequences": [{"sequence_id": "seq02"}],
    },
}

THREE_DATASETS = {
    "ds1": {
        "fps": 30,
        "sequences": [{"sequence_id": "seq01"}],
    },
    "ds2": {
        "fps": 60,
        "sequences": [{"sequence_id": "seq02"}],
    },
    "ds3": {
        "fps": 25,
        "sequences": [{"sequence_id": "seq03"}],
    },
}

MULTI_VIDEO = {
    "ds1": {
        "fps": 30,
        "sequences": [
            {"sequence_id": "seq01_a", "video_start": 0, "video_length": 100},
            {"sequence_id": "seq01_b", "video_start": 100, "video_length": 100},
        ],
    },
}

# Maps algo_name → [component_names] (reverse of the old comp_algo_mapping)
TWO_ALGORITHMS = {"algo_a": ["body_joints"], "algo_b": ["body_joints"]}
THREE_ALGORITHMS = {"algo_a": ["body_joints"], "algo_b": ["body_joints"], "algo_c": ["body_joints"]}


# ---------------------------------------------------------------------------
# Lightweight stand-ins for experiment config objects
# ---------------------------------------------------------------------------


@dataclass
class FakeRunSequence:
    sequence_id: str
    video_start: int | str = 0
    video_length: int | str = 100


@dataclass
class FakeRunDataset:
    sequences: list[FakeRunSequence]


@dataclass
class FakeAnnotationComponent:
    path: Path


@dataclass
class FakeAnnotation:
    components: dict[str, FakeAnnotationComponent] = field(default_factory=dict)


@dataclass
class FakeSequenceConfig:
    sequence_id: str
    annotation: FakeAnnotation = field(default_factory=FakeAnnotation)


@dataclass
class FakeDatasetConfig:
    sequences: list[FakeSequenceConfig] = field(default_factory=list)


class FakeRunIO(BaseModel):
    # Pydantic so resolve_placeholders can walk into it.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    detector_final_result_folder: Path = Path("")
    out_sub_folder: Path = Path("")


class FakeRunFile(BaseModel):
    # Pydantic so resolve_placeholders can walk into it. `run` is nested dataclasses
    # (no placeholders inside), so arbitrary_types_allowed keeps them unchanged.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    run: dict[str, FakeRunDataset] = Field(default_factory=dict)
    algorithms: list[str] = Field(default_factory=list)
    io: FakeRunIO = Field(default_factory=FakeRunIO)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_experiment_config(tmp_path: Path, datasets: dict, algo_component_mapping: dict[str, list[str]]) -> MagicMock:
    """Build a mock DetectorsExperimentConfig for resolver tests.

    Each dataset entry supports:
    - ``fps``, ``sequences`` — for experiment source
    - ``annotation_components`` — optional dict of ``{comp_name: path_template}``
      for annotation source. Templates may use ``<cur_dataset_name>``,
      ``<cur_sequence_id>``, ``<cur_component_name>`` as a fixture-internal
      shorthand for building per-sequence paths — the helper resolves them
      before assigning to each ``FakeSequenceConfig``, so the paths handed to
      evaluation are already concrete (matching the real dataset loader's
      sibling-scope resolution).
    """
    placeholder_path = Path("<cur_dataset_name>") / "<cur_sequence_id>" / "<cur_component_name>"
    result_template = tmp_path / placeholder_path
    sub_folder_template = tmp_path / "<cur_dataset_name>" / "<cur_sequence_id>"

    run_datasets: dict[str, FakeRunDataset] = {}
    dataset_configs: dict[str, FakeDatasetConfig] = {}

    for ds_name, ds_spec in datasets.items():
        run_sequences = [FakeRunSequence(**s) for s in ds_spec["sequences"]]
        run_datasets[ds_name] = FakeRunDataset(sequences=run_sequences)

        # Build annotation components per-sequence with paths pre-resolved,
        # matching the real dataset loader's sibling-scope resolution.
        raw_components = ds_spec.get("annotation_components", {})
        per_sequence_components: dict[str, dict[str, FakeAnnotationComponent]] = {}
        for run_seq in run_sequences:
            components: dict[str, FakeAnnotationComponent] = {}
            for comp_name, ann_path in raw_components.items():
                ctx = {
                    "cur_dataset_name": ds_name,
                    "cur_sequence_id": run_seq.sequence_id,
                    "cur_component_name": comp_name,
                }
                resolved = Path(resolve_placeholders(ann_path, ctx))
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.touch()
                components[comp_name] = FakeAnnotationComponent(path=resolved)
            per_sequence_components[run_seq.sequence_id] = components

        dataset_configs[ds_name] = FakeDatasetConfig(
            sequences=[
                FakeSequenceConfig(
                    sequence_id=run_seq.sequence_id,
                    annotation=FakeAnnotation(components=per_sequence_components[run_seq.sequence_id]),
                )
                for run_seq in run_sequences
            ],
        )

        # Create experiment .npz files and per-sequence subsequence_meta.toml on disk
        ds_fps = ds_spec.get("fps", 30)
        for run_seq in run_sequences:
            for algo, components in algo_component_mapping.items():
                for comp in components:
                    npz_dir = tmp_path / ds_name / run_seq.sequence_id / comp
                    npz_dir.mkdir(parents=True, exist_ok=True)
                    (npz_dir / f"{algo}.npz").touch()

            sub_folder = tmp_path / ds_name / run_seq.sequence_id
            sub_folder.mkdir(parents=True, exist_ok=True)

            resolved_start = run_seq.video_start if isinstance(run_seq.video_start, int) else 0
            resolved_length = run_seq.video_length if isinstance(run_seq.video_length, int) else 100
            save_config(
                {
                    "sequence_id": run_seq.sequence_id,
                    "video_start": resolved_start,
                    "video_length": resolved_length,
                    "fps": ds_fps,
                },
                sub_folder / "subsequence_meta.toml",
            )

    # Build fake detector config: each algo has a config with .components set
    fake_detector_algorithms = {}
    for algo_name, components in algo_component_mapping.items():
        fake_algo_cfg = MagicMock()
        fake_algo_cfg.components = components
        fake_detector_algorithms[algo_name] = fake_algo_cfg

    run_file = FakeRunFile(
        run=run_datasets,
        algorithms=list(algo_component_mapping.keys()),
        io=FakeRunIO(detector_final_result_folder=result_template, out_sub_folder=sub_folder_template),
    )

    cfg = MagicMock()
    cfg.run_config = run_file
    cfg.dataset_config = dataset_configs
    cfg.detector_config = MagicMock()
    cfg.detector_config.algorithms = fake_detector_algorithms
    return cfg


# ---------------------------------------------------------------------------
# Expected output builders
# ---------------------------------------------------------------------------


def expected_experiment_metas(
    tmp_path: Path,
    datasets: dict,
    algo_component_mapping: dict[str, list[str]],
    npz_key: str,
) -> list[ExperimentMeta]:
    """Build the expected ExperimentMeta list matching make_experiment_config output."""
    out: list[ExperimentMeta] = []
    for ds_name, ds_spec in datasets.items():
        for subseq_idx, s in enumerate(ds_spec["sequences"]):
            video_start = s.get("video_start", 0)
            video_length = s.get("video_length", 100)
            for algo, components in algo_component_mapping.items():
                for comp in components:
                    out.append(
                        ExperimentMeta(
                            dataset=ds_name,
                            sequence=s["sequence_id"],
                            component=comp,
                            algorithm=algo,
                            subsequence=SubsequenceInfo(
                                subsequence_index=subseq_idx,
                                video_start=video_start,
                                video_length=video_length,
                            ),
                            npz_path=tmp_path / ds_name / s["sequence_id"] / comp / f"{algo}.npz",
                            npz_key=npz_key,
                        )
                    )
    return sorted(out, key=lambda x: str(x.npz_path))


def expected_annotation_metas(
    datasets: dict,
    npz_key: str,
) -> list[AnnotationMeta]:
    """Build the expected AnnotationMeta list matching make_experiment_config output."""
    out: list[AnnotationMeta] = []
    for ds_name, ds_spec in datasets.items():
        for s in ds_spec["sequences"]:
            ctx = {
                "cur_dataset_name": ds_name,
                "cur_sequence_id": s["sequence_id"],
            }
            for comp_name, ann_path in ds_spec.get("annotation_components", {}).items():
                resolved = Path(resolve_placeholders(ann_path, ctx))
                out.append(
                    AnnotationMeta(
                        dataset=ds_name,
                        sequence=s["sequence_id"],
                        component=comp_name,
                        npz_path=resolved,
                        npz_key=npz_key,
                    )
                )
    return sorted(out, key=lambda x: str(x.npz_path))
