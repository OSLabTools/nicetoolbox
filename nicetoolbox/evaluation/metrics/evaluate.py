"""
Evaluation metrics runner and result processing.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..in_out import IO
from .base_metric import MetricFactory
from .results_schema import AggregatedResult, FrameResult, ResultFileGroup, ResultGrid


@dataclass
class EvalResults:
    """
    Container for evaluation results.

    Stores `file groups` which hold frame level metrics and also `summaries` which
    carry aggregated metrics. Provides a save function to export results to disk.

    Structure of saved `file groups`:
        NPZ file path - <experiment_folder>/<dataset_name>__<session>__<sequence>/
                        <component>/<algorithm>__<metric_type>.npz
        NPZ entries   - data_description.npy:
                        {"data_description": {metric_name: description}}
                        where each description is a dictionary with {
                            "axis0": ["person"], "axis1": ["camera"],
                            "axis2": ["frames"], "axis3": ["metric_dim"]
                            }
                        - <metric_name>.npy: ndarray of metric results, shape:
                        [#person x #camera x #frames x #metric_out_dim]

    Structure of saved `summaries`:
        CSV file path - <experiment_folder>/<dataset_name>_summary.csv
        CSV entries   - metric_type, metric, component, algorithm, value
    """

    file_groups: List[ResultFileGroup] = field(default_factory=list)
    summaries: List[AggregatedResult] = field(default_factory=list)

    def save(self, io_manager: IO) -> None:
        """
        Saves all evaluation results to disk for the given dataset.

        Args:
            io_manager (IO): IO manager for file operations.
        """
        dataset_name = io_manager.dataset_name
        if not self.file_groups and not self.summaries:
            logging.error(
                f"No results to save for dataset {dataset_name}. " "Ensure metrics are computed and stored correctly."
            )
        if self.file_groups:
            logging.info(f"Saving {len(self.file_groups)} npy groups for dataset {dataset_name}.")
            self._export_file_groups(io_manager)
        if self.summaries:
            logging.info(f"Saving {len(self.summaries)} aggregated metrics to file")
            self._export_summaries(io_manager, dataset_name)

    def _export_file_groups(self, io_manager) -> None:
        """
        Exports the results as NPZ files for each file group.

        Args:
            io_manager (IO): IO manager for file operations.
        """
        for group in self.file_groups:
            payload = {"data_description": {g.metric_name: g.description for g in group.grids}}
            for grid in group.grids:
                payload[grid.metric_name] = grid.values
            selection = f"{group.session}__{group.sequence}" if group.sequence else group.session
            out_folder = io_manager.get_out_folder(selection, group.component)
            file_name = f"{group.algorithm}__{group.metric_type}.npz"
            io_manager.save_npz(out_folder / file_name, **payload)

    def _export_summaries(self, io_manager, dataset_name):
        """Exports aggregated summary metrics to csv.

        Args:
            io_manager (IO): IO manager for file operations.
            dataset_name (str): Name of the current dataset.
        """
        metric_types = set(s.metric_type for s in self.summaries)

        for metric_type in metric_types:
            metric_summaries = [s for s in self.summaries if s.metric_type == metric_type]
            if not metric_summaries:
                continue

            summary_path = io_manager.output_folder / f"{dataset_name}_{metric_type}_summary.csv"
            summary_data = [
                {
                    "metric_type": s.metric_type,
                    "metric": s.metric_name,
                    "component": s.component,
                    "algorithm": s.algorithm,
                    "person": s.person,
                    "camera": s.camera,
                    "value": s.value,
                }
                for s in metric_summaries
            ]

            io_manager.save_summaries_to_csv(summary_path, summary_data)


class MetricRunner:
    """
    Drives all metrics: initialize, process samples, evaluate, collect results.
    """

    def __init__(self, loader, eval_cfg: dict):
        """
        Initializes the metric runner with a data loader and evaluation configuration.
        Calls the MetricFactory to create all metric handlers.

        Args:
            loader: DataLoader that yields batches of data.
            eval_cfg (dict): Configuration dictionary for evaluation, including
                             device and metric settings.
        """
        self.loader = loader
        self.device = eval_cfg.device
        self.metric_handlers = MetricFactory.create_all(eval_cfg, self.device)

    def evaluate(self) -> EvalResults:
        """
        Runs the full evaluation process: dispatches batches to metric handlers for
        processing, computes final metric results, and formats results.

        Returns:
            EvalResults: The final structured evaluation results.
        """
        if not self.metric_handlers:
            logging.warning("No metric handlers found. Skipping evaluation.")
            return EvalResults({})

        logging.info("Starting metric evaluation...")
        for batch_group in self.loader:
            for grouped_key, grouped_batch in batch_group.items():
                metric_type = grouped_key[0]

                # Dispatch to the appropriate metric handler
                for handler in self.metric_handlers:
                    if handler.name == metric_type:
                        handler.process(grouped_batch)
                        break

        logging.info("Computing final metric results...")
        raw_results = {}
        for handler in self.metric_handlers:
            raw_results[handler.name] = handler.evaluate()

        logging.info("Formatting raw metric results into export ready state...")
        result_processor = _ResultsProcessor(raw_results)
        results: EvalResults = result_processor.format_results()
        return results


class _ResultsProcessor:
    """
    Processes raw results from metric handlers into a structured format.
    """

    def __init__(self, raw_results_from_handlers: Dict):
        """
        Initializes the results processor with raw results from metric handlers."""
        self.raw_results = raw_results_from_handlers

    def format_results(self) -> EvalResults:
        """
        Formats the processed results into a structured evaluation results object.
        Unpacks and separates frame-level and aggregated results, groups frame-level
        results by file, and constructs the final EvalResults object.

        Returns:
            EvalResults: The structured evaluation results.
        """
        frame_results, summary_results = self._unpack_and_separate()
        grouped_by_file = self._group_by_file(frame_results)
        file_groups = self._create_file_groups(grouped_by_file)
        return EvalResults(file_groups=file_groups, summaries=summary_results)

    def _unpack_and_separate(self) -> Tuple[List[FrameResult], List[AggregatedResult]]:
        """
        Unpacks raw results into flat lists of frame-level and aggregated results.

        Returns:
            Tuple[List[FrameResult], List[AggregatedResult]]: Separated frame-level and
            aggregated results.
        """
        frame_results, aggregated_results = [], []
        for metric_type, handler_output in self.raw_results.items():
            for key, result_object in handler_output.items():
                if isinstance(result_object, list):  # Assumes List[BatchResult]
                    comp, algo, metric_name = key
                    for batch in result_object:
                        axis3_desc = batch.results_description
                        for i in range(len(batch.results_tensor)):
                            chunk, frame = batch.meta_chunk, batch.meta_frames[i]
                            frame_results.append(
                                FrameResult(
                                    value=batch.results_tensor[i].numpy(),
                                    metric_type=metric_type,
                                    metric_name=metric_name,
                                    session=chunk.session,
                                    sequence=chunk.sequence,
                                    component=comp,
                                    algorithm=algo,
                                    person=frame.person,
                                    camera=frame.camera,
                                    frame=frame.frame,
                                    axis3_description=axis3_desc,
                                )
                            )
                elif isinstance(result_object, AggregatedResult):
                    aggregated_results.append(result_object)
        return frame_results, aggregated_results

    def _group_by_file(self, flat_results: List[FrameResult]) -> Dict[Tuple, List[FrameResult]]:
        """
        Groups frame-level results by unique file identifiers.

        Args:
            flat_results (List[FrameResult]): Flat list of frame-level results.

        Returns:
            Dict[Tuple, List[FrameResult]]: Grouped frame-level results by
                (session, sequence, component, algorithm, metric_type).
        """
        grouped = defaultdict(list)
        for res in flat_results:
            key = (
                res.session,
                res.sequence,
                res.component,
                res.algorithm,
                res.metric_type,
            )
            grouped[key].append(res)
        return grouped

    def _create_file_groups(self, grouped_data: Dict) -> List[ResultFileGroup]:
        """
        Creates final ResultFileGroup objects from grouped frame-level results
        after gridding into structured arrays of shape
        [#person, #camera, #frames, #metric_out_dim].

        Args:
            grouped_data (Dict): Grouped frame-level results.

        Returns:
            List[ResultFileGroup]: List of structured result file groups.
        """
        final_groups = []
        for (ses, seq, comp, algo, metric_type), frame_list in grouped_data.items():
            grids = self._grid_results(frame_list)
            if grids:
                final_groups.append(ResultFileGroup(ses, seq, comp, algo, metric_type, grids))
        return final_groups

    def _grid_results(self, frame_results: List[FrameResult]) -> List[ResultGrid]:
        """
        Converts a list of frame-level results into structured grids.
        Each grid has shape [#person, #camera, #frames, #metric_out_dim]
        and is accompanied by a description of each axis.

        Args:
            frame_results (List[FrameResult]): List of frame-level results.

        Returns:
            List[ResultGrid]: List of structured result grids.
        """
        metrics_to_grid = defaultdict(list)
        for result in frame_results:
            metrics_to_grid[result.metric_name].append(result)

        all_persons, all_cameras, all_frames = set(), set(), set()
        for result in frame_results:
            all_persons.add(result.person)
            all_cameras.add(result.camera)
            all_frames.add(result.frame)

        p_list, c_list, f_list = (
            sorted(list(all_persons)),
            sorted(list(all_cameras)),
            sorted(list(all_frames)),
        )
        p_map, c_map, f_map = (
            {n: i for i, n in enumerate(p_list)},
            {n: i for i, n in enumerate(c_list)},
            {n: i for i, n in enumerate(f_list)},
        )

        result_grids = []
        for metric_name, fr_list in metrics_to_grid.items():
            metric_desc = fr_list[0].axis3_description
            metric_out_dim = fr_list[0].value.shape
            grid_shape = (len(p_list), len(c_list), len(f_list)) + metric_out_dim
            grid = np.full(grid_shape, np.NaN, dtype=np.float32)

            for res in fr_list:
                grid[p_map[res.person], c_map[res.camera], f_map[res.frame]] = res.value

            desc = {
                "axis0": p_list,
                "axis1": c_list,
                "axis2": [str(f) for f in f_list],  # TODO: frames should not be strings
                "axis3": metric_desc,
            }
            result_grids.append(ResultGrid(metric_name=metric_name, values=grid, description=desc))
        return result_grids
