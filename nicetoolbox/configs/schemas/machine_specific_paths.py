from pathlib import Path

from pydantic import BaseModel


class MachineSpecificConfig(BaseModel):
    """
    Schema for device specific paths configuration.
    """

    datasets_folder_path: Path
    output_folder_path: Path
    conda_path: Path
