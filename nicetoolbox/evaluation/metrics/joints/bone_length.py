import logging

import numpy as np

from nicetoolbox.configs.schemas.evaluation_aggr import AggSpec
from nicetoolbox.configs.schemas.evaluation_group_by import GroupBySpec
from nicetoolbox_core.data.array_schema import VECTOR_3D_CONF_PER_LABEL, VECTOR_3D_PER_LABEL, AnyOf
from nicetoolbox_core.data.loaded_array import NpzArrayWithMeta

from ....configs.schemas.evaluation_metrics_config import BoneLengthConfig
from ...data.input_loader import get_meta_type, load_input
from ...data.plots import plot_candle_per_group, plot_score
from ...data.summary import aggregate_summary, summarize_with_group_by
from ..base_metric import BaseMetric
from ..metric_result import FrameResult, MetricResult, PlotResult, SummaryResult


class BoneLengthMetric(BaseMetric):
    """Compute L2 bone lengths per frame from joint positions.

    Single-input metric (no ground truth). Uses ``human_pose.bone_dict`` to map
    semantic joint names to bone endpoint pairs.
    """

    metric_config: BoneLengthConfig
    bones: dict[str, list[str]]
    input_schema = AnyOf(VECTOR_3D_PER_LABEL, VECTOR_3D_CONF_PER_LABEL)

    def _init_metric(self) -> None:
        # this metrics config guarantees, that group_by will contain sequence, subject and label
        # usually we have unique subjects per sequence
        # and mixing bone length of multiple subjects doesn't make any sense
        assert self.metric_config.summary_group_by.contains("sequence", "subject", "label")
        hp = self.config_handler.predictions_mapping.human_pose
        self.bones = hp.bone_dict

    def compute(self) -> MetricResult:
        # read input
        arrays = load_input(self.metric_config.predictions, schema=self.input_schema)
        meta = get_meta_type(arrays)

        # calculate frame by frame bone distances
        bone_length_arrays: list[NpzArrayWithMeta] = []
        for arr in arrays:
            result = self._compute_bone_lengths(arr)
            if result is not None:
                bone_length_arrays.append(result)
        if not bone_length_arrays:
            raise ValueError("Failed to compute Bone Length: no vailid bones found!")

        # generate detailed summary for each bone
        summary = summarize_with_group_by(
            bone_length_arrays,
            self.metric_config.summary_group_by,
            self.metric_config.summary_aggr,
        )

        # compute score (coefficient of variation) and aggregate it
        # TODO: this will average across cameras (allow config customization?)
        # be careful with camera specific algorithms - comparing might be unfair
        cv_summary = summarize_with_group_by(
            bone_length_arrays,
            self.metric_config.summary_group_by,
            AggSpec.of_type("cv"),
        )
        score = aggregate_summary(cv_summary, agg_col="cv", meta_type=meta)

        # optional visualization
        figures = {}
        if self.metric_config.visualize:
            compare_dim = meta.comparable_dim()

            # visualize all unique bones with their length and distributions as candles graph
            figures = plot_candle_per_group(
                bone_length_arrays,
                x_col="label",  # X is always per bone results (axis3)
                series_col=compare_dim,  # compare different algorithms or npz_keys
                split_by=GroupBySpec(dims=["subsequence", "subject"]),  # keep bones unique per subject and subsequence
                base_title=self.metric_name,
                y_label="Bone Length (meters)",
            )

            # visualize score bar chart for final score
            figures["coefficient_variation_score"] = plot_score(
                score,
                x_col=compare_dim,  # X is different allgorithms / npz_keys
                y_col="cv",  # Y is their average cv metrics
                title="Bone Length CV Score",
                x_label=compare_dim,
                y_label="CV (lower is better)",
            )

        return MetricResult(
            self.metric_name,
            frames=FrameResult({"bone_length": bone_length_arrays, "predictions": arrays}),
            plots=PlotResult(figures),
            summary=SummaryResult({"coefficient_variation_score": score, "summary": summary}),
        )

    def _compute_bone_lengths(self, arr: NpzArrayWithMeta) -> NpzArrayWithMeta | None:
        """Compute per-frame bone lengths for a single loaded array.

        Returns a MetaNpzArray with shape (subjects, cameras, frames, n_bones)
        where fields are bone names. Returns None if no valid bones found.
        """
        # Use spatial coordinates only (drop confidence if present)
        positions = arr.data[..., :3]
        joints = {name: i for i, name in enumerate(arr.axes.labels)}
        bones = self.bones

        lengths: list[np.ndarray] = []
        processed_bones: list[str] = []
        for bone_name, joint_pair in bones.items():
            joint_a, joint_b = joint_pair
            if joint_a not in joints or joint_b not in joints:
                logging.warning(
                    f"Bone '{bone_name}' requires joints [{joint_a}, {joint_b}], "
                    f"but available labels are {arr.axes.labels}. Skipping."
                )
                continue

            # calculate distance between joints
            idx_a = joints[joint_a]
            idx_b = joints[joint_b]
            diff = positions[:, :, :, idx_a, :] - positions[:, :, :, idx_b, :]
            lengths.append(np.linalg.norm(diff, axis=-1))
            processed_bones.append(bone_name)

        if not processed_bones:
            logging.warning(f"No valid bones found for array from {arr.meta}. Skipping.")
            return None

        # Stack into (subjects, cameras, frames, n_bones)
        result_data = np.stack(lengths, axis=-1)
        result_axes = arr.axes.replace(labels=processed_bones)
        return arr.replace(data=result_data, axes=result_axes)
