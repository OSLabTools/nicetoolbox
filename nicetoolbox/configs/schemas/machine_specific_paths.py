from pydantic import BaseModel, Field


class MachineSpecificConfig(BaseModel):
    """
    Schema for machine-specific paths configuration.
    Contains paths and secrets that are fixed per physical machine.
    """

    hugging_face_token: str = Field(default="")
    ffmpeg_hwaccel: str
