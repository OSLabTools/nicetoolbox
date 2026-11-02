from copy import deepcopy
from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator

from ...utils.graph import topological_sort
from .detectors_instances_configs import DETECTORS_REGISTRY, BaseAlgorithmConfig


def _resolve_template_inheritance(templates: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve template-to-template inheritance using topological sort (parents first)."""
    graph = {name: ([tmpl["template"]] if "template" in tmpl else []) for name, tmpl in templates.items()}
    try:
        order = topological_sort(graph)
    except ValueError as e:
        raise ValueError(f"Circular template inheritance detected: {e.args[0]}") from None

    resolved: Dict[str, Any] = {}
    for name in order:
        tmpl = dict(templates[name])
        parent_name = tmpl.pop("template", None)
        if parent_name:
            tmpl = {**resolved[parent_name], **tmpl}
        resolved[name] = tmpl
    return resolved


def _resolve_templates(algos: Dict[str, Any], templates: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each algorithm entry that has a `template` key, merge the named template
    dict with the algorithm dict (algorithm fields take precedence). The `template`
    key itself is consumed and not forwarded to the schema.
    """
    resolved_templates = _resolve_template_inheritance(templates)
    resolved = {}
    for instance_name, algo_dict in algos.items():
        if isinstance(algo_dict, BaseAlgorithmConfig):
            resolved[instance_name] = algo_dict
            continue
        template_name = algo_dict.get("template")
        if template_name is None:
            resolved[instance_name] = algo_dict
            continue
        if template_name not in resolved_templates:
            raise ValueError(f"Algorithm '{instance_name}' references unknown template '{template_name}'.")
        merged = {**resolved_templates[template_name], **algo_dict}
        resolved[instance_name] = merged
    return resolved


class VideoSubjectAssignmentConfig(BaseModel):
    """Video subjects assignments parameters."""

    min_overlap: float
    confidence_weight: float


class AudioSubjectAssignmentConfig(BaseModel):
    """Audio subjects assignments parameters."""

    # minimal fraction of a subject's pinned range a speaker cluster must cover to be assigned to it
    min_overlap: float = Field(ge=0.0, le=1.0)


class SubjectAssignmentConfig(BaseModel):
    """Per-modality parameters for the subject-assignment algorithm."""

    video: VideoSubjectAssignmentConfig
    audio: AudioSubjectAssignmentConfig


# Top-level config in detectors_config.toml
class DetectorsConfig(BaseModel):
    """
    Contains all detector configurations. The TOML key under [algorithms.*]
    is a user-defined instance name; the schema class is selected via the
    `algorithm_type` field.

    Optional [template.*] sections define shared field sets that algorithm
    entries can inherit via `template = "<name>"`. Algorithm fields override
    template fields.
    """

    algorithms: dict[str, BaseAlgorithmConfig] = {}
    templates: dict[str, Any] = {}
    subject_assignment: SubjectAssignmentConfig

    @model_validator(mode="before")
    @classmethod
    def parse_algorithms(cls, values):
        values = deepcopy(values)
        templates = values.get("templates", {})
        algos = values.get("algorithms", {})
        algos = _resolve_templates(algos, templates)
        parsed = DETECTORS_REGISTRY.parse_dict_by_type(algos, type_field_name="algorithm_type")
        for instance_name, cfg in parsed.items():
            cfg._instance_name = instance_name
        values["algorithms"] = parsed
        return values
