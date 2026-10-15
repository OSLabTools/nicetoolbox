import os
from pathlib import Path
from typing import Optional

from .system import detect_arch, detect_os_type

# TODO: this global variables are a bit ugly
_ffmpeg_path: Optional[str] = None
_ffprobe_path: Optional[str] = None


def get_ffmpeg_asset_key() -> str:
    """
    Return the asset manifest key of the ffmpeg build for this platform.

    The key doubles as the binaries' folder path relative to assets_root.
    """
    return f"ffmpeg/{detect_os_type()}/{detect_arch()}"


def initialize_ffmpeg(assets_root: Path) -> None:
    """
    Resolve and validate the ffmpeg/ffprobe paths under assets_root and store them.

    Args:
        assets_root: Root directory that contains the ffmpeg/ subfolder tree.

    Raises:
        RuntimeError: If a binary is missing or not executable.
    """
    global _ffmpeg_path, _ffprobe_path

    suffix = ".exe" if detect_os_type() == "windows" else ""
    binary_dir = Path(assets_root) / get_ffmpeg_asset_key() / "bin"

    ffmpeg_path = binary_dir / f"ffmpeg{suffix}"
    ffprobe_path = binary_dir / f"ffprobe{suffix}"

    for name, path in (("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path)):
        if not path.exists():
            raise RuntimeError(f"{name} binary not found at: {path}")
        if not os.access(path, os.X_OK):
            raise RuntimeError(f"{name} binary is not executable: {path}")

    _ffmpeg_path = str(ffmpeg_path)
    _ffprobe_path = str(ffprobe_path)


def get_ffmpeg_path() -> str:
    """
    Return the absolute path to the ffmpeg executable.

    Raises:
        RuntimeError: If initialize_ffmpeg() has not been called.
    """
    if _ffmpeg_path is None:
        raise RuntimeError("ffmpeg not initialized. Call initialize_ffmpeg(assets_root) before use.")
    return _ffmpeg_path


def get_ffprobe_path() -> str:
    """
    Return the absolute path to the ffprobe executable.

    Raises:
        RuntimeError: If initialize_ffmpeg() has not been called.
    """
    if _ffprobe_path is None:
        raise RuntimeError("ffprobe not initialized. Call initialize_ffmpeg(assets_root) before use.")
    return _ffprobe_path
