from typing import Annotated, Dict, Literal, Union

from pydantic import BaseModel, Field, RootModel


class BaseAssetMetadata(BaseModel):
    """Fields shared by every asset, regardless of download backend."""

    url: str
    access: Literal["hosted", "open", "gated"]


class UrlAssetMetadata(BaseAssetMetadata):
    """A single file fetched over HTTP and written verbatim to the asset path."""

    source: Literal["url"]


class ZipAssetMetadata(BaseAssetMetadata):
    """A zip archive fetched over HTTP and extracted into the asset folder."""

    source: Literal["zip"]
    flatten_single_root: bool = False


class HuggingFaceAssetMetadata(BaseAssetMetadata):
    """A Hugging Face repo snapshot; `url` holds the repo id."""

    source: Literal["huggingface"]


# Discriminated on `source`, so each backend only accepts the fields that apply to it.
AssetMetadata = Annotated[
    Union[UrlAssetMetadata, ZipAssetMetadata, HuggingFaceAssetMetadata],
    Field(discriminator="source"),
]


class AssetManifest(RootModel[Dict[str, AssetMetadata]]):
    """
    Validation schema for asset_manifest.toml.
    Validates a dictionary where the key is the target relative path (str)
    and the value contains the asset metadata (url and source).
    """

    pass
