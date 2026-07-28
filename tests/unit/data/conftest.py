from pathlib import Path

import numpy as np


def make_npz(
    path: Path,
    key: str = "landmarks",
    subjects: tuple = ("s1", "s2"),
    cameras: tuple = ("cam1",),
    frames: tuple = ("f0", "f1", "f2"),
    labels: tuple = ("x", "y", "z"),
    data: tuple | None = None,
) -> Path:
    """Create a minimal well-formed npz file at *path* and return it.

    Data values are sequential integers so tests can verify correct slices.
    """
    descr: dict = {
        key: {
            "axis0": list(subjects),
            "axis1": list(cameras),
            "axis2": list(frames),
            "axis3": list(labels),
        }
    }
    shape = (len(subjects), len(cameras), len(frames), len(labels))
    if data is not None:
        descr[key]["axis4"] = list(data)
        shape = shape + (len(data),)

    npz_data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data_description=np.array(descr, dtype=object), **{key: npz_data})
    return path
