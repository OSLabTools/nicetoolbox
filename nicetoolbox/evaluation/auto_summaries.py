"""
Creates automatic summaries from pre computed evaluation results given aggregation
configurations from the evaluation config.
"""

import logging
from typing import Dict

from ..configs.schemas.evaluation_config import AggregationConfig
from .in_out import IO
from .results_wrapper.core import EvaluationResults
from .results_wrapper.initialize import INDEX_LEVELS


def create_auto_summaries(
    io_manager: IO,
    results: EvaluationResults,
    aggregation_configs: Dict[str, AggregationConfig],
) -> None:
    """Creates automatic summaries based on the provided aggregation configurations and
    exports them to the csv output folder.

    Args:
        io_manager: The IO manager for handling input/output operations.
        results: The EvaluationResults object containing the evaluation data.
        aggregation_configs: A dictionary of aggregation configurations.
    """

    for summary_name, aggr_config in aggregation_configs.items():
        logging.info(f"Creating automatic summary: {summary_name}")
        results.reset()

        # (1) Check that selected metrics exist
        selected_metrics_exist = all(metric in results.available_metrics for metric in aggr_config.metric_names)
        if not selected_metrics_exist:
            logging.error(f"Selected metrics {aggr_config.metric_names} not present in results, " "Skipping.")
            continue

        # (2) Invert: aggregation dimensions to groupby dimensions
        group_by = [level for level in INDEX_LEVELS if level not in aggr_config.aggregate_dims]
        # Since aggregate_dims is a subset of "sequence", "person", "camera", "label",
        # "frame", group_by will always contain at least "metric_name", "dataset",
        # "component", "algorithm, "metric_type"

        # (3) Apply filtering, aggregation and exporting using wrapper api
        csv_path = (
            results.query(metric_name=aggr_config.metric_names, **aggr_config.filter)
            .aggregate(group_by=group_by, agg_funcs=aggr_config.aggr_functions)
            .to_csv(
                output_dir=io_manager.get_csv_folder() / "summaries",
                file_name=summary_name,
            )
        )
        logging.info(f"Summary '{summary_name}' exported to: {csv_path}")
