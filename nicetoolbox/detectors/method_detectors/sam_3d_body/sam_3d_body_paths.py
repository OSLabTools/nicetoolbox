"""
Paths for SAM 3D Body: Hugging Face cache, upstream repo (git submodule), and raw inference .npz.
"""

import os
from pathlib import Path

SAM3D_ASSETS_DIRNAME = "sam_3d_body"
SAM3D_REPO_DIRNAME = "sam-3d-body"
SUBMODULES_DIRNAME = "submodules"

SAM3D_BODY_OUTPUT_NPZ_STEM = "sam_3d_body"
SAM3D_BODY_LOCAL_NPZ_STEM = "sam_3d_body"
RAW_INFERENCE_NPZ_NAME = "sam_3d_body_inference_raw.npz"


def default_sam3d_repo_path(nicetoolbox_root: Path) -> Path:
    """Default checkout: <repo>/submodules/sam-3d-body."""
    return nicetoolbox_root / SUBMODULES_DIRNAME / SAM3D_REPO_DIRNAME


def default_sam3d_assets_root(nicetoolbox_root: Path) -> Path:
    return nicetoolbox_root / "nicetoolbox" / "detectors" / "assets" / SAM3D_ASSETS_DIRNAME


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_sam3d_repo(repo: str | None, nicetoolbox_root: Path) -> Path:
    """Resolve sam-3d-body checkout containing the sam_3d_body/ package."""
    target = (
        Path(repo.strip()).expanduser().resolve()
        if repo and str(repo).strip()
        else default_sam3d_repo_path(nicetoolbox_root)
    )
    if not (target / "sam_3d_body").is_dir():
        raise RuntimeError(
            f"SAM 3D Body upstream repo missing or incomplete at {target} (expected package dir "
            f"'{target / 'sam_3d_body'}'). Clone the fork via git submodule: "
            "`git submodule update --init submodules/sam-3d-body` "
            "(see `.gitmodules`), or set `sam3d_repo_path` in `[algorithms.sam_3d_body]` to a "
            "local checkout that contains `sam_3d_body/`."
        )
    return target


def ensure_hf_hub_cache_env(nicetoolbox_root: Path) -> Path:
    """Force Hugging Face to read exclusively from the pre-downloaded local assets root."""
    assets_dir = default_sam3d_assets_root(nicetoolbox_root)

    # Configure directory overrides
    os.environ["HF_HUB_CACHE"] = str(assets_dir)
    os.environ["HF_HOME"] = str(assets_dir)

    # Native Hugging Face toggle to disable all internet/API lookups completely
    os.environ["HF_HUB_OFFLINE"] = "1"
