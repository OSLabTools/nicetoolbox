from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


@dataclass
class DifferentKeysError:
    filename: str
    expected_keys: list[str]
    generated_keys: list[str]


@dataclass
class DifferentShapeError:
    filename: str
    key: str
    expected_shape: tuple
    generated_shape: tuple


@dataclass
class DifferentValuesError(Exception):
    filename: str
    key: str
    max_abs_diff: float = np.nan
    mean_abs_diff: float = np.nan
    std: float = np.nan


def compare_npz_files(generated_path: Path, golden_path: Path, rtol: float = 1e-5, atol: float = 1e-8) -> list:
    try:
        generated_data = np.load(generated_path, allow_pickle=True)
        golden_data = np.load(golden_path, allow_pickle=True)
    except Exception as e:
        pytest.fail(f"Could not load NPZ files for comparison: {e}")

    errors = []
    generated_keys = sorted(generated_data.files)
    golden_keys = sorted(golden_data.files)
    if generated_keys != golden_keys:
        errors.append(DifferentKeysError(str(generated_path), golden_keys, generated_keys))

    for key in golden_data.files:
        if key not in generated_data:
            continue
        gen_arr = generated_data[key]
        gold_arr = golden_data[key]

        if gen_arr.shape != gold_arr.shape:
            errors.append(DifferentShapeError(str(generated_path), key, gold_arr.shape, gen_arr.shape))
            continue
        if np.issubdtype(gold_arr.dtype, np.floating):
            if not np.allclose(gen_arr, gold_arr, rtol=rtol, atol=atol, equal_nan=True):
                abs_dif = np.abs(gen_arr - gold_arr)
                max_abs_diff = np.nanmax(abs_dif)
                mean_abs_diff = np.nanmean(abs_dif)
                std = np.nanstd(gen_arr - gold_arr)
                errors.append(DifferentValuesError(str(generated_path), key, max_abs_diff, mean_abs_diff, std))
        else:
            if not np.array_equal(gen_arr, gold_arr):
                errors.append(DifferentValuesError(str(generated_path), key))

    return errors
