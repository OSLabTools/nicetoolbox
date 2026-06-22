from typing import Dict, Literal

from pydantic import BaseModel, Field, RootModel


class AssetMetadata(BaseModel):
    """Metadata for a single asset."""

    url: str = Field(..., description="The HTTP URL for direct downloads, or the Repo ID for Hugging Face.")
    source: Literal["url", "huggingface"] = Field(
        ..., description="The download backend: 'url' for direct HTTP, 'huggingface' for HF hub."
    )
    access: Literal["hosted", "open", "gated"] = Field(
        ...,
        description="Access level: 'hosted' (direct URL), 'open' (public HF), 'gated' (requires token and license).",
    )


class AssetManifest(RootModel[Dict[str, AssetMetadata]]):
    """
    Validation schema for asset_manifest.toml.
    Validates a dictionary where the key is the target relative path (str)
    and the value contains the asset metadata (url and source).
    """

    pass
