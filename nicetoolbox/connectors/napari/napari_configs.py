from pathlib import Path

from pydantic import BaseModel, NonNegativeInt, PositiveInt, model_validator

from ...configs.models.video_timestamp import VideoTimestamp


class NapariSequenceConfig(BaseModel):
    input: Path
    output: Path
    fps: PositiveInt  # frame rate of the annotated video; only used to resolve timestamps
    start: NonNegativeInt | VideoTimestamp
    end: int | VideoTimestamp  # -1 means end of annotations
    reset_frames: bool


class NapariImportBodyJointsConfig(BaseModel):
    log_level: str
    log_file_path: Path
    export_csv: bool

    run: dict[str, NapariSequenceConfig]

    subjects: dict[str, str] = {}
    cameras: dict[str, str] = {}


class NapariWindowConfig(BaseModel):
    size: PositiveInt  # frames kept per window
    stride: PositiveInt  # distance between window starts

    @model_validator(mode="after")
    def _no_overlap(self) -> "NapariWindowConfig":
        if self.stride < self.size:
            raise ValueError(f"window stride ({self.stride}) must be >= size ({self.size}); windows may not overlap")
        return self


class NapariExportSequenceConfig(BaseModel):
    input: Path  # body_joints NPZ
    output: Path  # napari project root; gets "labeled-data/<camera>/" with the .h5 and frames
    cameras: str | list[str]  # toolbox camera names to export; "*" means all
    frames_folder: Path  # nicetoolbox_input folder holding "<camera>/frames/<frame>.png"


class NapariExportBodyJointsConfig(BaseModel):
    log_level: str
    log_file_path: Path

    npz_key: str
    window: NapariWindowConfig | None = None

    run: dict[str, NapariExportSequenceConfig]
