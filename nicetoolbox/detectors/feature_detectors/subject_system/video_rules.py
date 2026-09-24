from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np

from nicetoolbox_core.data.loaded_array import load_array_from_path

from ....configs.schemas.subject_rules import BaseRuleConfig

# TODO: remove cycle? need this just for annotation
if TYPE_CHECKING:
    from .video_subject_detector import VideoSubjectDetector


# =======================================
#           Base Video Rule
# ========================================


class BaseVideoRule(ABC):
    rule_type: str  # subclass sets to the rule's discriminator string

    def __init__(self, detector: "VideoSubjectDetector", cam_idx: int, rule: BaseRuleConfig):
        self.detector = detector
        self.rule = rule
        self.subjects, self.cameras, self.frames = detector.data.canonical_axes
        self.subject_index = {name: i for i, name in enumerate(self.subjects)}

        # camera related look up
        self.cam_idx = cam_idx
        self.cam_name = self.cameras[cam_idx]
        width, height = detector.data.video_resolutions[self.cam_name]
        self.width = float(width)
        self.height = float(height)

    def empty_slab(self) -> np.ndarray:
        return np.full((len(self.subjects), len(self.frames), 4), np.nan, dtype=float)

    @abstractmethod
    def compute(self) -> np.ndarray:
        """
        Return the subject-region bounding box for this specific camera.

        Shape should be (n_subjects, n_frames, 4) with [x0, y0, x1, y1] on the last axis, ordered
        along the detector's canonical subjects and frames axes.
        """
        ...


# =======================================
#           Simple Video Rule
# ========================================


class NoneVideoRule(BaseVideoRule):
    """No subject is placed on this camera; every slot stays empty."""

    rule_type = "none"

    def compute(self):
        return self.empty_slab()


class AssignVideoRule(BaseVideoRule):
    """The whole frame is assigned to one named subject."""

    rule_type = "assign"

    def compute(self):
        slab = self.empty_slab()
        slab[self.subject_index[self.rule.subject], :, :] = [0.0, 0.0, self.width, self.height]
        return slab


class BBoxRule(BaseVideoRule):
    """Each subject gets a fixed rectangle on this camera."""

    rule_type = "bbox"

    def compute(self):
        slab = self.empty_slab()
        for subj_name, box in self.rule.boxes.items():
            slab[self.subject_index[subj_name], :, :] = box
        return slab


# ===================================================
#           Horizontal/Vertical split rules
# ===================================================


class BaseSplitRule(BaseVideoRule):
    """Base for rules that partition the frame along one axis, one slice per subject in order."""

    vertical: bool = False

    def compute(self):
        axis_length = self.width if self.vertical else self.height
        fractions = self._normalize_ratio(self.rule.ratio, len(self.rule.order))
        slab = self.empty_slab()
        offset = 0.0
        for subj_name, frac in zip(self.rule.order, fractions):
            span = frac * axis_length
            if self.vertical:
                box = [offset, 0.0, offset + span, self.height]
            else:
                box = [0.0, offset, self.width, offset + span]
            slab[self.subject_index[subj_name], :, :] = box
            offset += span
        return slab

    @staticmethod
    def _normalize_ratio(ratio, n: int) -> List[float]:
        # turn a ratio spec into n fractions that sum to 1.0
        if ratio is None:
            return [1.0 / n] * n
        if isinstance(ratio, (int, float)):
            if n != 2:
                raise ValueError(f"Scalar ratio requires order of length 2, got {n}.")
            r = float(ratio)
            return [r, 1.0 - r]
        values = [float(x) for x in ratio]
        if len(values) != n:
            raise ValueError(f"Ratio list length {len(values)} != order length {n}.")
        total = sum(values)
        if total <= 0:
            raise ValueError("Ratio values must sum to a positive number.")
        return [v / total for v in values]


class VSplitHandler(BaseSplitRule):
    """Splits the frame into vertical strips, one per subject in order."""

    rule_type = "vsplit"
    vertical = True


class HSplitHandler(BaseSplitRule):
    """Splits the frame into horizontal bands, one per subject in order."""

    rule_type = "hsplit"
    vertical = False


# =======================================
#           External split rule
# =======================================


class PathVideoRule(BaseVideoRule):
    """Reads per-frame subject regions from a precomputed npz on disk."""

    rule_type = "path"

    def compute(self):
        array = load_array_from_path(self.rule.path, "regions")
        if array is None:
            raise ValueError(
                f"video_subject_detector: camera '{self.cam_name}' PathRule "
                f"npz '{self.rule.path}' has no 'regions' key."
            )

        slab = self.empty_slab()
        axes = array.axes
        data = array.data
        subj_lookup = {name: idx for idx, name in enumerate(axes.subjects)}
        cam_lookup = {name: idx for idx, name in enumerate(axes.cameras)}
        if self.cam_name not in cam_lookup:
            raise ValueError(
                f"PathRule npz for camera '{self.cam_name}' has no camera axis entry "
                f"for '{self.cam_name}'. Available: {axes.cameras}."
            )
        if len(axes.frames) != len(self.frames):
            raise ValueError(
                f"PathRule npz frame count {len(axes.frames)} != sequence "
                f"frame count {len(self.frames)} for camera '{self.cam_name}'."
            )
        src_cam = cam_lookup[self.cam_name]
        for out_subj_idx, subj_name in enumerate(self.subjects):
            if subj_name not in subj_lookup:
                continue
            src_subj = subj_lookup[subj_name]
            slab[out_subj_idx, :, :] = data[src_subj, src_cam, :, :4]
        return slab
