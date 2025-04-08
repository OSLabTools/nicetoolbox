import logging

import numpy as np

from nicetoolbox.utils import filehandling as fh

logger = logging.getLogger("testing")


def compare_npz_files(
    prediction_file, expectation_file, description="", rtol=1e-3, atol=1e-1
):
    if not prediction_file.exists() or not expectation_file.exists():
        return False
    prediction = fh.read_npz_file(prediction_file)
    expectation = fh.read_npz_file(expectation_file)
    return compare_dicts_of_general_nparrays(
        prediction, expectation, description, rtol, atol
    )


def compare_dicts_of_general_nparrays(
    prediction, expectation, description="", rtol=1e-3, atol=1e-1
):
    is_equal, common_keys = compare_dict_keys(prediction, expectation, description)

    for key in common_keys:
        is_equal &= compare_numerical_nparrays(
            prediction[key], expectation[key], key, rtol=rtol, atol=atol
        )

    return is_equal


def compare_dict_keys(dict1, dict2, description=""):
    dict1_keys = set(dict1.keys())
    dict2_keys = set(dict2.keys())

    is_equal = True
    if dict1_keys - dict2_keys != set():
        is_equal = False
        logger.debug(
            f"{description}: Keys {dict1_keys - dict2_keys} found "
            "in the first but not the second dictionary."
        )
    if dict2_keys - dict1_keys != set():
        is_equal = False
        logger.debug(
            f"{description}: Keys {dict2_keys - dict1_keys} found "
            "in the second but not the first dictionary."
        )

    return is_equal, dict1_keys & dict2_keys


def compare_numerical_nparrays(pred, expect, description="", rtol=1e-3, atol=1e-1):
    if pred.shape != expect.shape:
        logger.debug(
            f"{description}: Shapes of arrays differ: {pred.shape} vs "
            f"{expect.shape}."
        )
        return False

    if expect.dtype.kind in {"i", "u", "f", "c"}:
        if not np.allclose(pred, expect, equal_nan=True, rtol=rtol, atol=atol):
            diff = np.nanmean(np.abs(pred - expect))
            logger.debug(
                f"{description}: Arrays "
                f"differ by a mean absolute difference of {diff}."
            )
            return False
    elif expect.dtype.kind in {"U", "O", "S", "b"}:
        if not np.all(pred == expect):
            logger.debug(f"{description}: Arrays of objects/strings/bools differ.")
            return False
    else:
        logger.debug(
            f"{description}: Arrays of type {expect.dtype.kind} are not supported."
        )

    return True


def compare_dicts_of_collections_of_strings(dict1, dict2, description=None):
    if dict1 != dict2:
        _, common_keys = compare_dict_keys(dict1, dict2)

        for key in common_keys:
            if dict1[key] != dict2[key]:
                logger.debug(
                    f"{description}: Values of key '{key}' are not equal."
                    f"\ndict1 = {dict1[key]}\n dict2 = {dict2[key]}"
                )
        return False
    return True
