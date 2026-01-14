"""
Keypoint metrics for evaluating 3D joints data. Includes bone length calculation
and jump detection metrics. No ground truth required.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import toml
import torch

from ...configs.schemas.evaluation_config import EvaluationMetricType
from ..data.discovery import ChunkWorkItem, FrameInfo
from .base_metric import Metric, MetricHandler
from .results_schema import AggregatedResult, BatchResult, MetricReturnType


class KeypointMetric(MetricHandler):
    """
    Handler for all keypoint metrics.
    """

    def __init__(self, cfg: EvaluationMetricType, device: str):
        self.keypoints_mapping = toml.load(cfg.keypoint_mapping_file)
        super().__init__(cfg, device)

    @property
    def name(self) -> str:
        return "keypoint_metrics"

    def _create_metrics(self) -> Dict[str, Metric]:
        """Instantiates the individual metric objects this handler manages."""
        metric_map = {"bone_length": BoneLength, "jump_detection": JumpDetection}
        return {
            name: metric_map[name](self.keypoints_mapping)  # Metric class instantiated
            for name in self.cfg.metric_names
            if name in metric_map
        }


class BoneLength(Metric):
    """Calculates the bone lengths for each frame using 3D only keypoints."""

    def __init__(self, keypoints_mapping: Dict[str, Any]):
        self.keypoints_mapping = keypoints_mapping
        self.bone_dict = self.keypoints_mapping["human_pose"]["bone_dict"]
        self.bone_names = list(self.bone_dict.keys())
        super().__init__()

    def reset(self) -> None:
        """Reset the metric's internal state."""
        # For frame-based results, key is (comp, algo, "bone_length")
        self.storage: Dict[Tuple[str, str, str], List[BatchResult]] = defaultdict(list)
        # For summary, key is (comp, algo, person, bone_name)
        self.summary_storage: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)

    def get_axis3(self) -> List[str]:
        """Get the bone names the metric is concerned with."""
        return self.bone_names

    def update(
        self,
        preds: torch.Tensor,
        gts: None,
        meta_chunk: ChunkWorkItem,
        meta_frames: List[FrameInfo],
    ) -> None:
        """Computes the bone lengths and stores them."""
        _gts = gts  # This metric does not use ground truth data.
        if meta_chunk.pred_data_key != "3d":  # Only apply to 3D keypoints
            return

        if preds.shape[-1] != 4:
            raise ValueError(
                f"Predictions must have 4 coordinates for 3D keypoints + "
                f"confidence score. Current shape is {preds.shape}"
            )

        keypoint_names = meta_chunk.pred_data_description_axis3
        if not keypoint_names:
            raise ValueError("Bone length: No keypoints found in reconciliation map.")

        # Determine if any bone has both endpoints present in the reconciled keypoints
        keypoint_set = set(keypoint_names)
        has_valid_bone = any(kp1 in keypoint_set and kp2 in keypoint_set for kp1, kp2 in self.bone_dict.values())
        if not has_valid_bone:
            return

        preds = preds[:, :, :3]

        comp, algo = meta_chunk.component, meta_chunk.algorithm
        num_frames = preds.shape[0]
        bone_lengths = torch.full((num_frames, len(self.bone_names)), torch.nan, device=preds.device)
        for i, bone_name in enumerate(self.bone_names):
            keypoint1, keypoint2 = self.bone_dict[bone_name]

            if keypoint1 in keypoint_names and keypoint2 in keypoint_names:
                idx1 = keypoint_names.index(keypoint1)
                idx2 = keypoint_names.index(keypoint2)
                coord1 = preds[:, idx1, :]
                coord2 = preds[:, idx2, :]
                distance = torch.linalg.norm(coord1 - coord2, dim=-1)
                bone_lengths[:, i] = distance

                # Store lengths for summary per bone and person (always 3d so no camera)
                for frame_idx, frame_info in enumerate(meta_frames):
                    length = distance[frame_idx]
                    if not torch.isnan(length):
                        summary_key = (comp, algo, frame_info.person, bone_name)
                        self.summary_storage[summary_key].append(length.item())

        comp, algo = meta_chunk.component, meta_chunk.algorithm
        key = (comp, algo, "bone_length")
        description = self.get_axis3()
        self.storage[key].append(BatchResult(bone_lengths.cpu(), description, meta_chunk, meta_frames))

    def compute(self) -> Dict[Tuple[str, str, str], MetricReturnType]:
        """Compute the final metric from the stored state."""
        results: Dict[Tuple[str, str, str], MetricReturnType] = {}

        # (1) Add the frame-based results to the output dictionary
        results.update(self.storage)

        # (2) Create and add aggregated summary results
        for (comp, algo, person, bone_name), lengths in self.summary_storage.items():
            if not lengths:
                continue

            lengths_tensor = torch.tensor(lengths, dtype=torch.float32)
            mean_val = torch.mean(lengths_tensor).item()
            std_val = torch.std(lengths_tensor).item()

            # Create AggregatedResult for the mean
            mean_metric_name = f"mean_bone_length__{bone_name}"
            mean_key = (comp, algo, person, bone_name, "mean")
            results[mean_key] = AggregatedResult(
                value=mean_val,
                metric_type="keypoint_metrics",
                metric_name=mean_metric_name,
                component=comp,
                algorithm=algo,
                person=person,
                camera="3d",
            )

            # Create AggregatedResult for the standard deviation
            std_metric_name = f"std_bone_length__{bone_name}"
            std_key = (comp, algo, person, bone_name, "std")
            results[std_key] = AggregatedResult(
                value=std_val,
                metric_type="keypoint_metrics",
                metric_name=std_metric_name,
                component=comp,
                algorithm=algo,
                person=person,
                camera="3d",
            )
        return results


class JumpDetection(Metric):
    """Detects jumps in keypoint sequences based on sudden large movements."""

    def __init__(self, keypoints_mapping: Dict[str, Any]):
        self.keypoints_mapping = keypoints_mapping
        self.joint_diameter_map = self.keypoints_mapping["human_pose"]["joint_diameter_size"]
        super().__init__()

    def reset(self) -> None:
        """Initializes and resets the metric's internal state."""
        # For frame-based results
        self.storage: Dict[Tuple[str, str, str], List[BatchResult]] = defaultdict(list)
        # For aggregated summary results
        self.summary_counts: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
        # To handle continuity across batches
        self.last_preds: Dict[Tuple[str, str, str, str], torch.Tensor] = {}

    def get_axis3(self, meta_chunk: ChunkWorkItem) -> List[str]:
        """Get the names of the joints used for jump detection."""
        return meta_chunk.pred_data_description_axis3  # No reconciliation since no GT

    def update(
        self,
        preds: torch.Tensor,
        gts: None,
        meta_chunk: ChunkWorkItem,
        meta_frames: List[FrameInfo],
    ) -> None:
        """Detects jumps and stores the results."""
        _gts = gts  # This metric does not use ground truth data.
        if "3d" not in meta_chunk.pred_data_key:  # Only apply to 3D keypoints
            return

        if preds.shape[-1] != 4:
            raise ValueError(
                f"Predictions must have 4 coordinates for 3D keypoints + "
                f"confidence score. Current shape is {preds.shape}"
            )

        preds = preds[:, :, :3]

        keypoint_names = meta_chunk.pred_data_description_axis3
        # Create a threshold tensor based on the order of keypoints in the input
        # Convert diameter from mm to meters
        thresholds = torch.tensor(
            [self.joint_diameter_map.get(name, float("inf")) * 1000.0 for name in keypoint_names],
            device=preds.device,
        )

        # Group predictions by person and camera to handle state continuity correctly
        grouped_by_context = defaultdict(list)
        comp, algo = meta_chunk.component, meta_chunk.algorithm
        for i, frame in enumerate(meta_frames):
            context_key = (comp, algo, frame.person, frame.camera)
            grouped_by_context[context_key].append((preds[i], frame))

        for context_key, preds_and_frames in grouped_by_context.items():
            current_preds_list, current_frames_list = zip(*preds_and_frames)
            current_preds = torch.stack(current_preds_list)

            # Retrieve the last frame from the previous batch for this context
            last_pred = self.last_preds.get(context_key)

            if last_pred is not None:
                # Prepend last frame to correctly calculate movement for the first frame
                # of the current batch
                full_sequence = torch.cat([last_pred.unsqueeze(0), current_preds])
            else:
                full_sequence = current_preds

            if len(full_sequence) < 2:
                # Not enough data to calculate movement, store the frame and continue
                self.last_preds[context_key] = current_preds[-1]
                continue

            # Calculate movement between consecutive frames
            movements = torch.linalg.norm(full_sequence[1:] - full_sequence[:-1], dim=-1)
            # Detect jumps where movement exceeds the joint diameter threshold
            jumps = movements > thresholds

            # --- Store Frame-Based Results ---
            key = (comp, algo, "jump_detection")
            description = self.get_axis3(meta_chunk)
            self.storage[key].append(BatchResult(jumps.cpu(), description, meta_chunk, list(current_frames_list)))

            # --- Update Summary Counts ---
            # Sum jumps per frame and add to the total for this context
            total_jumps_in_batch = torch.sum(jumps).item()
            self.summary_counts[context_key] += total_jumps_in_batch

            # Update the state with the last prediction of the current batch
            self.last_preds[context_key] = current_preds[-1]

    def compute(self) -> Dict[Tuple[str, str, str], MetricReturnType]:
        """
        Compute the final metric, returning both frame-based results and a summary.
        """
        results: Dict[Tuple[str, str, str], MetricReturnType] = {}

        # (1) Add the frame-based results to the output dictionary
        results.update(self.storage)

        # (2) Create and add aggregated summary results
        for (comp, algo, person, camera), count in self.summary_counts.items():
            # Create a unique key for the summary metric to avoid overwriting
            summary_key = (comp, algo, person, camera, "jump_count")
            results[summary_key] = AggregatedResult(
                value=float(count),
                metric_type="keypoint_metrics",
                metric_name="jump_count",
                component=comp,
                algorithm=algo,
                person=person,
                camera=camera,
            )

        return results
