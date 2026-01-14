"""
Builds the pandas dataframe backbone using MultiIndex for high dimensional labeled data.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

INDEX_LEVELS = [
    "metric_name",  # Entry in npz file - first level for improved querying speed
    "dataset",  # Metadata extracted from file path below
    "sequence",
    "component",
    "algorithm",
    "metric_type",
    "person",  # Npz file internal dimensions below
    "camera",
    "frame",
    "label",
]


def _parse_metadata_from_path(path: Path) -> Dict[str, str] | None:
    """Extracts metadata from file path."""
    try:
        component = path.parent.name
        dataset_seq = path.parent.parent.name
        algorithm, metric_type = path.stem.split("__", 1)
        dataset, sequence = dataset_seq.split("__", 1)
        return {
            "component": component,
            "dataset": dataset,
            "sequence": sequence,
            "algorithm": algorithm,
            "metric_type": metric_type,
        }
    except (ValueError, IndexError) as err:
        logging.error(
            f"Could not parse metadata from path, skipping: {path}. Error:\n{err}",
            exc_info=True,
        )
        return None


def build_index(root: Path) -> pd.DataFrame:
    """
    Builds a DataFrame in "long" format by creating a MultiIndex from evaluation results
    stored in .npz files under the provided root folder. After collecting the
    MultiIndex and metric data from all files into individual Series, they are
    concatenated into a DataFrame with a single 'value' column.

    Args:
        root: The root folder containing evaluation result .npz files.

    Returns:
        A pandas DataFrame with a MultiIndex built from the evaluation results.
    """
    series_to_concat = []

    for path in root.rglob("*.npz"):
        metadata = _parse_metadata_from_path(path)
        if not metadata:
            continue

        try:
            with np.load(path, allow_pickle=True) as npz:
                descriptions = npz["data_description"].item()

                for metric_name, metric_array in npz.items():
                    if metric_name == "data_description":
                        continue
                    if metric_name not in descriptions:
                        logging.warning(f"Metric '{metric_name}' in file {path} has no description," " skipping.")
                        continue

                    # (1) Get description for the current metric name
                    d = descriptions[metric_name]
                    persons, cameras, labels = d["axis0"], d["axis1"], d["axis3"]
                    num_frames = len(d["axis2"])

                    # (2) Build MultiIndex for the current metric name
                    levels = [
                        [metric_name],
                        [metadata["dataset"]],
                        [metadata["sequence"]],
                        [metadata["component"]],
                        [metadata["algorithm"]],
                        [metadata["metric_type"]],
                        persons,
                        cameras,
                        np.arange(num_frames),
                        labels,
                    ]
                    idx = pd.MultiIndex.from_product(levels, names=INDEX_LEVELS)

                    # (3) Flatten metric scores (4d array from npz file) into 1d array
                    #     that aligns with the MultiIndex layout. The flattening order
                    #     must match the MultiIndex construction order above. We then
                    #     create a Series with the MultiIndex and the flattened data.
                    s = pd.Series(metric_array.flatten(), index=idx, dtype=np.float32)
                    series_to_concat.append(s)
        except Exception as e:
            logging.warning(f"Skipping file {path} due to error: {e}", exc_info=True)

    if not series_to_concat:
        raise FileNotFoundError(f"No valid data loaded from {root}")

    # Concatenate all series into a single-column DataFrame with column name 'value'
    df = pd.concat(series_to_concat).to_frame(name="value")

    return df
