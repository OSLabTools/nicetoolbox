"""
Dataset class for evaluation, implemented as an IterableDataset.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .discovery import ChunkWorkItem
from .loaders import AnnotationLoader, PredictionLoader


class EvaluationDataset(IterableDataset):
    def __init__(
        self,
        work_items: List[ChunkWorkItem],
        prediction_loader: PredictionLoader,
        annotation_loader: Optional[AnnotationLoader],
    ):
        """Initialize the EvaluationDataset with found work items, the prediction
        loader, and optionally the annotation loader.

        Args:
            work_items (List[ChunkWorkItem]): List of work items (chunks) to process.
            prediction_loader (PredictionLoader): Loader for prediction data.
            annotation_loader (Optional[AnnotationLoader]): Loader for annotation data.
        """
        self.work_items = work_items
        self.pred_loader = prediction_loader
        self.annot_loader = annotation_loader

        self.total = sum(len(chunk.frames) for chunk in self.work_items)
        if not self.work_items:
            logging.warning("EvaluationDataset initialized with zero work items.")

    def __len__(self):
        """Returns the total number of frames to be processed."""
        return self.total

    @staticmethod
    def _apply_reconciliation(
        data: np.ndarray, rec_map: Dict[str, Tuple[int, ...]]
    ) -> np.ndarray:
        """
        Applies a reconciliation map to a loaded data array. Slices the
        array according to the provided map. We use this to handle cases where
        the predictions and ground truth data have different dimensions or
        need to be reconciled in some way (e.g., different keypoint sets).

        Args:
            data (np.ndarray): The data array to reconcile.
            rec_map (Dict[str, Tuple[int, ...]]): The reconciliation map.

        Returns:
            np.ndarray: The reconciled data array.
        """
        if data is None or not rec_map:
            return data

        axis3 = rec_map.get("axis3")
        axis4 = rec_map.get("axis4")

        if axis3 is not None and axis4 is not None:
            sliced = data[np.ix_(axis3, axis4)]
        elif axis3 is not None:
            sliced = data[axis3, ...]
        elif axis4 is not None:
            sliced = data[..., axis4]

        return sliced

    def __iter__(self):
        """
        Iterates over all work items, loading the prediction and ground truth data,
        applying reconciliation maps if available, and yielding the processed data.

        Yields:
            Tuple[np.ndarray, Optional[np.ndarray], WorkItem]: A tuple containing:
                - The processed prediction data as a numpy array.
                - The processed ground truth data as a numpy array (or None).
                - The corresponding WorkItem instance for metadata.
        """
        # Outer loop: iterate through each chunk (expensive I/O operation)
        for chunk in self.work_items:
            try:
                # === (1) Load full prediction and GT arrays ===
                raw_pred_array = self.pred_loader.load_full_array(
                    path=chunk.pred_path, data_key=chunk.pred_data_key
                )
                raw_pred_array = raw_pred_array.astype(np.float32)

                raw_gt_array: Optional[np.ndarray] = None
                if self.annot_loader and chunk.annot_path and chunk.annot_data_key:
                    raw_gt_array = self.annot_loader.load_full_array(
                        data_key=chunk.annot_data_key
                    )
                    raw_gt_array = raw_gt_array.astype(np.float32)
            except Exception as e:
                logging.error(f"Failed to load data for {chunk}: {e}", exc_info=True)
                continue

            # Inner loop: iterate through each frame in the chunk
            for frame_info in chunk.frames:
                try:
                    # === (2) Extract the relevant data for this frame ===
                    p_idx, c_idx, f_idx = frame_info.pred_slicing_indices
                    raw_pred_data = raw_pred_array[p_idx, c_idx, f_idx, ...]

                    raw_gt_data = None
                    if raw_gt_array is not None and frame_info.annot_slicing_indices:
                        p_idx_gt, c_idx_gt, f_idx_gt = frame_info.annot_slicing_indices
                        raw_gt_data = raw_gt_array[p_idx_gt, c_idx_gt, f_idx_gt, ...]

                    # === (3) Apply reconciliation maps if available ===
                    pred_data = EvaluationDataset._apply_reconciliation(
                        raw_pred_data, chunk.pred_reconciliation_map
                    )
                    gt_data = EvaluationDataset._apply_reconciliation(
                        raw_gt_data, chunk.gt_reconciliation_map
                    )

                    yield pred_data, gt_data, chunk, frame_info

                except Exception as e:
                    logging.error(
                        f"Failed to process frame {frame_info} in chunk "
                        f"{chunk.pred_path}: {e}",
                        exc_info=True,
                    )
                    continue

    @staticmethod
    def collate_fn(batch: List[tuple]) -> Dict[str, Any]:
        """
        Custom collate function given to the torch DataLoader to batch the data from
        the iterable EvaluationDataset.

        This function groups the samples by their chunk's metric type, prediction shape,
        component, algorithm, and prediction data key. This ensures that each batch
        contains homogeneous data, allowing for easier handling at metric processing
        time.

        Args:
            batch (List[tuple]): A list of tuples (pred_data, gt_data, metadata).

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing for each group:
                - 'pred': Batched prediction data as a torch tensor.
                - 'gt': Batched ground truth data as a torch tensor or None.
                - 'metadata': A list of metadata dictionaries from each work item.
        """
        grouped_samples = defaultdict(list)
        for pred, gt, chunk, frame in batch:
            compound_key = (
                chunk.metric_type,
                pred.shape,
                chunk.session,
                chunk.component,
                chunk.algorithm,
                chunk.pred_data_key,
            )
            grouped_samples[compound_key].append((pred, gt, chunk, frame))

        final_batch = {}
        for key, samples in grouped_samples.items():
            preds, gts, chunks, frames = zip(*samples)
            # Ensure that all chunks are equal objects since we grouped them
            all_equal = all(chunk == chunks[0] for chunk in chunks)
            if not all_equal:
                logging.error(
                    "In collate_fn, not all chunks are equal within a group. "
                    "This should not happen."
                )
                raise ValueError("Inconsistent chunks in collate_fn grouping.")
            stacked_preds = torch.from_numpy(np.stack(preds))
            stacked_gts = None
            if all(g is not None for g in gts):
                stacked_gts = torch.from_numpy(np.stack(gts))

            final_batch[key] = {
                "pred": stacked_preds,
                "gt": stacked_gts,
                "chunk": chunks[0],
                "frames": list(frames),
            }
        return final_batch
