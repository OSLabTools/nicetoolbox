"""
Pydantic models for input data recipes.

Recipes describe WHERE and HOW to load data for a specific modality.
They are created by the data preparation pipeline (nicetoolbox) and
consumed by data loaders (nicetoolbox_core).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# Video Recipe
# =============================================================================


class VideoInputRecipe(BaseModel):
    """Recipe for loading video frame sequences."""

    frames_folders: Dict[str, str]  # camera name -> resolved folder holding its frame files
    filename_template: str  # frame file name inside frames folder, formatted with `idx`
    range_start: int
    range_end: int
    step: int = 1
    fps: float
    resolutions: Dict[str, List[int]] = Field(default_factory=dict)

    @property
    def camera_names(self) -> List[str]:
        return list(self.frames_folders.keys())

    def frames_dir(self, camera: str) -> Path:
        """Folder holding the frame files of a camera."""
        return Path(self.frames_folders[camera])

    def frame_path(self, camera: str, idx: int) -> Path:
        """Path of a single frame file of a camera."""
        return self.frames_dir(camera) / self.filename_template.format(idx=idx)


# =============================================================================
# Audio Recipe
# =============================================================================


class AudioStreamRecipe(BaseModel):
    """Recipe for a single audio stream/track."""

    path: str
    source_path: str
    sample_rate: int
    channels: int
    source_type: str  # 'embedded' (from video) or 'standalone' (separate file)
    camera: Optional[str] = None  # Camera name if source_type is 'embedded'
    hears_subjects: List[int] = Field(min_length=1)  # Subject indices this track can hear.


class AudioInputRecipe(BaseModel):
    """Recipe for loading audio data."""

    start_time_ms: float
    end_time_ms: float

    streams: Dict[str, AudioStreamRecipe]  # Track name = key of dict


# =============================================================================
# Composed Recipe Container
# =============================================================================


class InputRecipes(BaseModel):
    """
    Container for all modality-specific input recipes.

    Passed to subprocesses and in-process loaders. Each loader
    reads only the recipe(s) it needs and ignores the rest.
    """

    video_input_recipe: Optional[VideoInputRecipe] = None
    audio_input_recipe: Optional[AudioInputRecipe] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputRecipes":
        """
        Reconstruct from a dict:

        1. Subprocess config format (has 'input_recipes' wrapper key):
           {"input_recipes": {"video_input_recipe": {...}, ...}}

        2. Flat format (direct output of to_flat_dict()):
           {"video_input_recipe": {...}, "audio_input_recipe": {...}}
        """
        # Unwrap if nested under 'input_recipes' key (subprocess config format)
        if "input_recipes" in data:
            data = data["input_recipes"]

        return cls(
            video_input_recipe=(
                VideoInputRecipe(**data["video_input_recipe"]) if "video_input_recipe" in data else None
            ),
            audio_input_recipe=(
                AudioInputRecipe(**data["audio_input_recipe"]) if "audio_input_recipe" in data else None
            ),
        )
