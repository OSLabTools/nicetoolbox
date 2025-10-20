"""
Data loaders for evaluation module, including lazy loading and caching.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np


class PredictionLoader:
    """
    Lazily loads a single prediction data slice from a large NPZ file.
    Caches open file handles for efficiency.
    """

    def __init__(self):
        self._cached = lru_cache(maxsize=16)(self._load)

    def _load(self, path: Path, data_key: str) -> np.lib.npyio.NpzFile:
        with np.load(path, mmap_mode="r", allow_pickle=True) as npz_file:
            if data_key not in npz_file.files:
                raise KeyError(f"Data key '{data_key}' not found in NPZ file {path}")
            return npz_file[data_key].copy()

    def load_full_array(self, path: Path, data_key: str) -> np.ndarray:
        """
        Loads a full data array from an NPZ file using bounded cache.
        Args:
            path: Path to the .npz file.
            data_key: The key for the data array within the npz file (e.g., '2d', '3d').

        Returns:
            The full numpy array.
        """
        return self._cached(path, data_key)

    def close_files(self):
        """Clears the cache of loaded arrays."""
        self._cached.cache_clear()


class AnnotationLoader:
    """
    Lazily loads a single ground truth data slice from a NPZ archive.
    Uses an LRU cache to optimize access to internal arrays.
    """

    def __init__(self, path_to_annotations: Path):
        self._path = path_to_annotations
        if not self._path.exists():
            raise FileNotFoundError(f"Annotation file not found at: {self._path}")
        logging.info(f"AnnotationLoader initialized for path: {self._path}")

        self._cached = lru_cache(maxsize=16)(self._load)

    def _load(self, data_key: str) -> Optional[np.ndarray]:
        logging.debug(f"Loading annotation chunk: {self._path} -> key='{data_key}'")
        try:
            with np.load(self._path, mmap_mode="r", allow_pickle=True) as npz:
                if data_key in npz:
                    return npz[data_key]
        except Exception as e:
            logging.error(
                f"Failed to load annotation array '{data_key}' from {self._path}: {e}"
            )
            return None
        return None

    def load_full_array(self, data_key: str) -> Optional[np.ndarray]:
        """
        Loads a full data array from within the single annotation NPZ.

        Args:
            data_key: The key for the data array within the npz file (e.g.,

        Returns:
            The full numpy array or None if the key does not exist.
        """
        return self._cached(data_key)

    def close_files(self):
        """Clears the LRU cache."""
        logging.info("Clearing annotation loader cache.")
        self._cached.cache_clear()
