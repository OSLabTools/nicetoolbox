from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, NonNegativeInt, PrivateAttr, ValidationError, model_validator

from ..models.dict_model import DictModel
from ..named_wildcards import discover_sequences_for_dataset
from ..placeholders import PLACEHOLDERS_TYPE, get_placeholders_str, resolve_placeholders_dict, resolve_placeholders_str


class VideoTrackConfig(BaseModel):
    """
    Configuration for a single video track (camera).

    A track has an arbitrary user-defined name, a path to its video file, and
    declares which subjects it sees. The path may contain a `*` wildcard and
    is expected to resolve to exactly one file (zero or multiple matches raise).
    """

    path: Path
    sees_subjects: List[int] = Field(min_length=1)


class SequenceVideo(BaseModel):
    """
    Configuration for dataset video modality.
    """

    cameras: Dict[str, VideoTrackConfig] = Field(default_factory=dict)


class AudioTrackConfig(BaseModel):
    """
    Configuration for a single audio track.

    A track is either:
    - Embedded: extracted from a camera's video file (has `camera` field)
    - Standalone: loaded from a separate audio file (has `path` field)

    Exactly one of `camera` or `path` must be set.
    """

    # Source: one of these must be set
    camera: Optional[str] = None  # Camera id to extract audio from. Mutually exclusive with path.
    path: Optional[Path] = None  # Path to standalone audio file. Mutually exclusive with camera.

    # Audio stream index in the source file (0-based). Relevant for multi-stream video files.
    stream: NonNegativeInt = 0
    # Audio channel index in the source file (0-based). Relevant for stereo, surround sound channel layouts
    # None will pass all channels to the audio detectors and let them to decide how to process multiple channels
    channel: Optional[NonNegativeInt] = None

    # Which subjects this track can hear. Indices into `subjects_descr`. Must be non-empty.
    hears_subjects: List[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source(self):
        """Ensure exactly one of camera or path is set."""
        has_camera = self.camera is not None and self.camera != ""
        has_path = self.path is not None

        if has_camera and has_path:
            raise ValueError("Audio track must have either 'camera' or 'path', not both.")
        if not has_camera and not has_path:
            raise ValueError("Audio track must have either 'camera' or 'path'.")
        return self

    @property
    def is_embedded(self) -> bool:
        """True if this track is extracted from a video file."""
        return self.camera is not None and self.camera != ""

    @property
    def is_standalone(self) -> bool:
        """True if this track is a standalone audio file."""
        return self.path is not None


class SequenceAudio(BaseModel):
    """
    Configuration for dataset audio modality.
    """

    tracks: Optional[Dict[str, AudioTrackConfig]] = Field(default_factory=dict)


class AnnotationComponentConfig(BaseModel):
    """
    Annotation source configuration for a single component.
    """

    path: Path


class SequenceAnnotation(BaseModel):
    """
    Optional per-component annotation paths used by evaluation input blocks.
    """

    components: Dict[str, AnnotationComponentConfig] = Field(default_factory=dict)


class SequenceConfig(BaseModel):
    """
    Configuration schema for a single sequence inside a dataset.

    Each sequence is fully self-describing: it owns its own cameras, audio
    tracks, subject list, and paths. Sequences within the same dataset need
    not share any structure.

    Arbitrary extra fields (e.g. `session_name`, `recording_name`) are
    allowed and used as placeholder variables inside this sequence's own
    string fields.
    """

    sequence_id: str

    subjects_descr: List[str]
    path_to_calibrations: Optional[Path] = None

    annotation: SequenceAnnotation = Field(default_factory=SequenceAnnotation)
    video: SequenceVideo = Field(default_factory=SequenceVideo)
    audio: SequenceAudio = Field(default_factory=SequenceAudio)


class DatasetConfig(BaseModel):
    """
    Configuration schema for a single dataset.

    A dataset is a flat list of sequences plus an optional shared template
    (whose fields are merged into every sequence at load time) and an
    optional filesystem discovery pattern.
    """

    discover_sequences: Optional[str] = None
    sequences: List[SequenceConfig]

    # Runtime fields
    _dataset_name: str = PrivateAttr()

    @model_validator(mode="after")
    def _check_unique_sequence_ids(self):
        seen: set[str] = set()
        for seq in self.sequences:
            if seq.sequence_id in seen:
                raise ValueError(f"Duplicate sequence_id '{seq.sequence_id}' in dataset.")
            seen.add(seq.sequence_id)
        return self


# Top-level config in dataset_properties.toml
class DatasetProperties(DictModel[str, DatasetConfig]):
    """
    Dictionary of dataset configurations, keyed by unique dataset name.
    Users can define any custom datasets.
    """

    def model_post_init(self, _):
        # injecting key into each DatasetConfig
        for name, ds in self.root.items():
            ds._dataset_name = name

    @classmethod
    def pre_placeholder_resolve(cls, raw: Dict[str, Any], placeholders: Dict[str, PLACEHOLDERS_TYPE]) -> Dict[str, Any]:
        # TODO: overall, this part is overcomplicated and smell, but I don't better way to do it
        # Dataset properties is unique, so we need to do couple of extra steps:
        # 1. Resolve placeholders at the dataset root (shallow) so discovery pattern is usable
        # 2. "Manually" validate template, discover_sequences and sequences fields
        # 3. Run filesystem sequence discovery and compose sequence_id from template when needed
        # 4. Merge explicit sequences on top of discovered ones (matched by sequence_id)
        # 5. Inject template values into every sequence (sequence keys override template)
        # In the end return config copy and continue standard stuff (final resolution + pydantic validation)

        # Because this function runs before pydantic, we need to validate types manually
        # Yes, this is inline pydantic check
        class _DatasetPreValidate(BaseModel):
            template: Optional[Dict[str, Any]] = None
            sequences: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
            discover_sequences: Optional[str] = None

        out: Dict[str, Any] = {}
        for name, dataset_raw in raw.items():
            # TODO: this will raise if there is runtime placeholder in dataset root
            # 1 - Resolve placeholders for this dataset (shallow, root only level, need for discovery)
            dataset_ret = resolve_placeholders_dict(dataset_raw, placeholders)
            # 2 - Validate selected fields that we need
            try:
                dataset = _DatasetPreValidate.model_validate(dataset_ret)
            except ValidationError as e:
                raise ValueError(f"Dataset '{name}': {e}") from e

            # 3 - Run sequence discovery
            template = dataset.template or {}
            discovered: List[Dict[str, Any]] = []
            if dataset.discover_sequences:
                discovered = discover_sequences_for_dataset(dataset.discover_sequences)
                # Compose sequence_id for each discovered entry using template's sequence_id
                # so explicit entries can override discovered ones by matching id.
                template_seq_id = template.get("sequence_id")
                if isinstance(template_seq_id, str):
                    for disc in discovered:
                        # If discovery already captured `sequence_id` directly (pattern used
                        # `[sequence_id]`), keep the capture — don't overwrite with template.
                        if "sequence_id" in disc:
                            continue
                        # Only compose if discovery captured every placeholder the template needs.
                        candidate = resolve_placeholders_str(template_seq_id, disc)
                        if get_placeholders_str(candidate):
                            continue
                        disc["sequence_id"] = candidate

            # 4 - Merge explicit sequences on top of discovered by sequence_id, then append the rest
            discovered_by_id = {d["sequence_id"]: d for d in discovered if "sequence_id" in d}
            merged: List[Dict[str, Any]] = list(discovered)
            for expl in dataset.sequences:
                expl_id = expl.get("sequence_id")
                if expl_id and expl_id in discovered_by_id:
                    discovered_by_id[expl_id].update(expl)
                else:
                    merged.append(expl)

            # 5 - Inject template values in each sequence (sequence keys override template)
            def _templ_resolve(template, seq):
                merged: Dict[str, Any] = {}
                for k, v in template.items():
                    merged[k] = v
                for k, v in seq.items():
                    merged[k] = v
                return merged

            full_sequences = [_templ_resolve(template, seq) for seq in merged]

            # Patch dataset raw with full list of sequences
            dataset_ret["sequences"] = full_sequences

            # Drop template, as upfront declared plaeholders will raise resolve error
            # TODO: keep it for logging? Some flag ignore resolution for specific field?
            dataset_ret.pop("template", None)

            out[name] = dataset_ret

        return out
