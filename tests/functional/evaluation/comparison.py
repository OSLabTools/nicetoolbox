"""
Helper functions for comparing generated files against golden files.
These are used in end-to-end tests to verify that the evaluation pipeline
produces the expected outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def compare_npz_files(
    generated_path: Path, golden_path: Path, rtol: float = 1e-5, atol: float = 1e-8
) -> None:
    """Compare two .npz files for key equality and numerical similarity.

    Args:
        generated_path: Path to the generated .npz file.
        golden_path: Path to the golden .npz file.
        rtol: Relative tolerance for floating point comparisons.
        atol: Absolute tolerance for floating point comparisons.

    Raises:
        Uses pytest.fail on file loading errors or assertion failures.
    """
    try:
        generated_data = np.load(generated_path, allow_pickle=True)
        golden_data = np.load(golden_path, allow_pickle=True)
    except Exception as e:
        pytest.fail(f"Could not load NPZ files for comparison: {e}")

    generated_keys = sorted(generated_data.files)
    golden_keys = sorted(golden_data.files)
    assert generated_keys == golden_keys, (
        f"NPZ files have different keys.\n"
        f"Generated: {generated_keys}\n"
        f"Golden: {golden_keys}"
    )

    for key in golden_data.files:
        gen_arr = generated_data[key]
        gold_arr = golden_data[key]

        assert gen_arr.shape == gold_arr.shape, f"Array '{key}' has mismatched shapes."

        # Use np.allclose for float arrays, direct comparison otherwise
        if np.issubdtype(gold_arr.dtype, np.floating):
            assert np.allclose(
                gen_arr, gold_arr, rtol=rtol, atol=atol, equal_nan=True
            ), f"Floating point array '{key}' differs from golden result."
        else:
            assert np.array_equal(
                gen_arr, gold_arr
            ), f"Non-float array '{key}' differs from golden result."


def compare_csv_files(generated_path: Path, golden_path: Path) -> None:
    """Compare two CSV files for equality using pandas.

    Comparison is robust to row order by sorting on all columns.

    Args:
        generated_path: Path to the generated CSV file.
        golden_path: Path to the golden CSV file.

    Raises:
        Uses pytest.fail on file loading errors or assertion failures.
    """
    try:
        generated_df = pd.read_csv(generated_path)
        golden_df = pd.read_csv(golden_path)
    except Exception as e:
        pytest.fail(f"Could not load CSV files for comparison: {e}")

    # Sort by all columns to make comparison independent of row order
    sorted_golden = golden_df.sort_values(by=list(golden_df.columns)).reset_index(
        drop=True
    )
    sorted_generated = generated_df.sort_values(
        by=list(generated_df.columns)
    ).reset_index(drop=True)

    try:
        assert_frame_equal(sorted_generated, sorted_golden, check_dtype=False)
    except AssertionError as e:
        pytest.fail(
            f"CSV content differs from golden result for "
            f"{generated_path.name}:\n{e}"
        )
