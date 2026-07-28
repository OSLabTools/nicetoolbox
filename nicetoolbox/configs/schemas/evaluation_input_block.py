from abc import ABC
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, BeforeValidator, Field

from nicetoolbox_core.data.loaded_array import NpzAxisFilters

from ..models.no_wildcard_str import NoWildcardStr


class BaseInputBlock(ABC, BaseModel):
    """Base for all input blocks."""

    source: Literal["experiment", "annotation", "path"]

    npz_key: NoWildcardStr  # TODO: allow list? breaks visualization
    subject: str | list[str] = "*"  # axis0
    camera: str | list[str] = "*"  # axis1
    label: str | list[str] = "*"  # axis3
    data: str | list[str] = "*"  # axis4

    def axis_filters(self) -> NpzAxisFilters:
        """Return detached axis filters used by array loading."""
        return NpzAxisFilters(
            subject=self.subject,
            camera=self.camera,
            label=self.label,
            data=self.data,
        )


class ExperimentInput(BaseInputBlock):
    """Default input block that discovers NPZ files from a saved experiment config."""

    source: Literal["experiment"] = "experiment"
    path: Path | None = None

    component: str | list[str]
    algorithm: str | list[str] = "*"

    dataset: str | list[str] = "*"
    sequence: str | list[str] = "*"
    subsequence: str | int | list[int] = "*"


class AnnotationInput(BaseInputBlock):
    """Input block that discovers annotation NPZ files from dataset properties."""

    source: Literal["annotation"] = "annotation"
    path: Path | None = None

    component: str | list[str]
    dataset: str | list[str] = "*"
    sequence: str | list[str] = "*"


class PathInput(BaseInputBlock):
    """Input block that resolves NPZ files from direct filesystem glob paths."""

    source: Literal["path"] = "path"
    path: Path | list[Path]  # glob path(s) to .npz files

    def paths_list(self) -> list[Path]:
        """Return path as a normalized list."""
        return self.path if isinstance(self.path, list) else [self.path]


# hack to inject default value of "expirement" into validator
def _default_source(v: object) -> object:
    if isinstance(v, dict) and "source" not in v:
        v = {**v, "source": "experiment"}
    return v


InputBlock = Annotated[
    Union[ExperimentInput, AnnotationInput, PathInput],
    BeforeValidator(_default_source),
    Field(discriminator="source"),
]
