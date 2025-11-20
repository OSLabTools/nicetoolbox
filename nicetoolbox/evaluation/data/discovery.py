"""
Discovery engine to find all work items for evaluation on disk.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...configs.schemas.dataset_properties import DatasetConfig
from ...configs.schemas.detectors_run_file import DetectorsRunConfig
from ...configs.schemas.evaluation_config import EvaluationMetricType
from ..config_schema import FinalEvaluationConfig
from ..in_out import IO

VALID_NPZ_PRED_KEYS = dict(
    body_joints=["3d", "2d_interpolated"],
    hand_joints=["3d", "2d_interpolated"],
    face_landmarks=["3d", "2d_interpolated"],
    gaze_interaction=["gaze_mutual_3d", "gaze_look_at_3d"],
)


@dataclass(frozen=True)
class FrameInfo:
    """Holds the per-frame information within a chunk."""

    frame: int
    person: str
    camera: str
    pred_slicing_indices: Tuple[int, int, int]
    annot_slicing_indices: Optional[Tuple[int, int, int]] = None


@dataclass(frozen=True)
class ChunkWorkItem:
    """
    Represents a chunk of frames, typically from a single data array within a
    prediction file. This is the atomic unit of work for the Dataset iterator.
    """

    # ====== Context from configs instance meta data ======
    dataset_name: str
    session: str
    sequence: str
    component: str
    algorithm: str
    metric_type: str

    # ====== Loading Information ======
    pred_path: Path
    pred_data_key: str  # The key in the PRED NPZ file, e.g., '2d' or '3d'
    annot_path: Optional[Path]
    annot_data_key: Optional[str]  # The key in the ANNOT NPZ file, e.g., 'S1__Seq1__2d'

    # ====== Input Data Description ======
    # The original list of labels (e.g. keypoint names) from the predictions file
    pred_data_description_axis3: List[str]

    # ====== Generic Reconciliation Plan ======
    # Empty map means no reconciliation needed. E.g. if no GT or same labels.
    pred_reconciliation_map: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    gt_reconciliation_map: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    # ====== Per-Frame Information ======
    frames: List[FrameInfo] = field(default_factory=list)


class DiscoveryEngine:
    """
    Scans file structures and configs to generate a flat list of all WorkItems.
    Crucially, it reconciles the per-clip prediction structure with the
    dataset-wide annotation structure.
    """

    def __init__(
        self,
        io_manager: IO,
        run_config: DetectorsRunConfig,
        dataset_properties: DatasetConfig,
        evaluation_config: FinalEvaluationConfig,
    ):
        self.io: IO = io_manager
        self.run_config: DetectorsRunConfig = run_config
        self.dataset_props: DatasetConfig = dataset_properties
        self.eval_config: FinalEvaluationConfig = evaluation_config

        self.gt_needed = any(
            cfg.gt_required for cfg in self.eval_config.metric_types.values()
        )
        self.annot_descriptions = self._load_and_process_annot_descriptions()

    def _load_and_process_annot_descriptions(self) -> Optional[Dict]:
        """
        Loads annotation descriptions if needed and applies synonyms.

        Returns:
            Optional[Dict]: A dictionary of annotation descriptions with applied
            synonyms, or None if ground truth is not needed.
        """
        if not self.gt_needed:
            return None

        annot_path = self.io.path_to_annotations
        try:
            with np.load(annot_path, allow_pickle=True) as data:
                if "data_description" not in data:
                    logging.error(
                        f"'data_description' not found in annotation file: {annot_path}"
                    )
                    return None
                descriptions = data["data_description"].item()
        except FileNotFoundError:
            logging.error(f"Annotation file not found at: {annot_path}.")
            return None

        synonyms = self.dataset_props.synonyms or {}
        for _, desc in descriptions.items():
            if isinstance(desc, int):
                continue
            for axis in desc:
                if axis in synonyms:
                    # synonyms found!
                    for label, synonym in synonyms[axis].items():
                        if label in desc[axis]:
                            label_idx = desc[axis].index(label)
                            desc[axis][label_idx] = synonym

        logging.info("Successfully loaded and processed annotation descriptions.")
        return descriptions

    def discover_work_items(self) -> List[ChunkWorkItem]:
        """
        Discover all chunk work items based on the run configuration,
        dataset properties, and evaluation configuration.
        This method scans the file structure for predictions and creates
        reconciliation plans for the ground truth annotations if
        required, i.e. when pred and GT labels like joints are not aligned.

        Returns:
            List[ChunkWorkItem]: A list of all discovered chunk work items.
        """
        all_chunk_items: List[ChunkWorkItem] = []
        logging.info(
            f"Starting discovery for dataset '{self.run_config._dataset_name}'..."
        )
        for video_config in self.run_config.videos:
            session, sequence = video_config.session_ID, video_config.sequence_ID
            start_frame = video_config.video_start
            end_frame = video_config.video_start + video_config.video_length

            for mt, components in self.eval_config.prediction_components.items():
                metric_cfg: EvaluationMetricType = self.eval_config.metric_types[mt]

                for component in components:
                    for algorithm in self.eval_config.component_algorithm_mapping[
                        component
                    ]:
                        file = self.io.get_detector_results_file(
                            video_config, component, algorithm
                        )
                        try:
                            with np.load(file, allow_pickle=True) as data:
                                description = data["data_description"].item()
                        except (FileNotFoundError, Exception) as e:
                            logging.error(
                                f"Failed to load prediction file: {file}. Error: {e}. "
                                "Skipping this file."
                            )
                            continue

                        chunk_items = self._create_chunks_from_description(
                            description,
                            file,
                            metric_cfg,
                            start_frame,
                            end_frame,
                            session,
                            sequence,
                            component,
                            algorithm,
                            mt,
                        )
                        if chunk_items:
                            all_chunk_items.extend(chunk_items)

        logging.info(f"Discovery complete. Found {len(all_chunk_items)} work items.")
        return all_chunk_items

    def _create_chunks_from_description(
        self,
        pred_descriptions: dict,
        pred_path: Path,
        metric_cfg: EvaluationMetricType,
        start_frame: int,
        end_frame: int,
        session: str,
        sequence: str,
        component: str,
        algorithm: str,
        metric_type: str,
    ) -> List[ChunkWorkItem]:
        """
        Create chunk work items from the prediction description.

        `pred_descriptions` references the `data_description` entry in a NPZ file.
        It is a dictionary containing a data_description for each prediction entry in
        the NPZ file. E.g. for vitpose.npz, it contains: A description for '2d',
        '2d_interpolated', '3d', etc. Each description contains the axes
        'axis0' (persons), 'axis1' (cameras), 'axis2' (frames),
        'axis3' (data e.g. joints) and optionally 'axis4' (coordinates).

        Args:
            pred_descriptions (dict): The descriptions for all keys of the NPZ file.
            pred_path (Path): The path to the prediction file.
            metric_cfg (MetricTypeConfig): The configuration for the metric type.
            start_frame (int): The start frame of the video.
            end_frame (int): The end frame of the video.
            session (str): The session ID.
            sequence (str): The sequence ID.
            component (str): The component name.
            algorithm (str): The algorithm name.
            metric_type (str): The metric type.

        Returns:
            List[ChunkWorkItem]: A list of created chunk work items.
        """
        created_chunks = []

        for pred_desc_key, pred_desc in pred_descriptions.items():
            if pred_desc_key not in VALID_NPZ_PRED_KEYS[component]:
                continue  # Skip unsupported prediction keys
            axis3 = pred_desc.get("axis3", [])

            frame_info_list: List[FrameInfo] = []
            gt_desc, annot_path, annot_data_key = None, None, None

            for p_idx, person_name in enumerate(pred_desc.get("axis0", [])):
                for c_idx, cam_name in enumerate(pred_desc.get("axis1", [])):
                    for f_idx, frame_str in enumerate(pred_desc.get("axis2", [])):
                        frame_num = int(frame_str)
                        if not (start_frame <= frame_num < end_frame):
                            continue

                        annot_slicing_idxs = None
                        if metric_cfg.gt_required and self.annot_descriptions:
                            if gt_desc is None and annot_data_key is None:
                                dim = "3d" if "3d" in pred_desc_key.lower() else "2d"
                                gt_key = DiscoveryEngine._parse_gt_key(
                                    session, sequence, dim
                                )
                                if gt_key in self.annot_descriptions:
                                    gt_desc = self.annot_descriptions[gt_key]
                                    annot_path = self.dataset_props.path_to_annotations
                                    annot_data_key = gt_key

                            if gt_desc:
                                try:
                                    gt_p_idx = gt_desc["axis0"].index(person_name)
                                    gt_c_idx = gt_desc["axis1"].index(cam_name)
                                    gt_f_idx = gt_desc["axis2"].index(frame_str)
                                    annot_slicing_idxs = (gt_p_idx, gt_c_idx, gt_f_idx)
                                except (ValueError, KeyError):
                                    logging.warning(  # This combo doesn't exist in GT
                                        f"GT data not found for {session}, {sequence}, "
                                        f"{person_name}, {cam_name}, frame {frame_str}."
                                        " Skipping GT reconciliation."
                                    )
                        frame_info_list.append(
                            FrameInfo(
                                frame=frame_num,
                                person=person_name,
                                camera=cam_name,
                                pred_slicing_indices=(p_idx, c_idx, f_idx),
                                annot_slicing_indices=annot_slicing_idxs,
                            )
                        )

            if not frame_info_list:
                logging.warning(
                    f"No valid frames found for {session}, {sequence}, {component}, "
                    f"{algorithm}, {metric_type}."
                )
                continue

            # Reconciliation maps e.g. if labels differ between GT and preds
            if gt_desc:
                pred_rec_map, gt_rec_map = DiscoveryEngine._create_reconciliation_maps(
                    pred_desc, gt_desc
                )
            else:
                pred_rec_map, gt_rec_map = {}, {}

            created_chunks.append(
                ChunkWorkItem(
                    dataset_name=self.run_config._dataset_name,
                    session=session,
                    sequence=sequence,
                    component=component,
                    algorithm=algorithm,
                    metric_type=metric_type,
                    pred_path=pred_path,
                    pred_data_key=pred_desc_key,
                    annot_path=annot_path,
                    annot_data_key=annot_data_key,
                    pred_data_description_axis3=axis3,
                    pred_reconciliation_map=pred_rec_map,
                    gt_reconciliation_map=gt_rec_map,
                    frames=frame_info_list,
                )
            )

        return created_chunks

    @staticmethod
    def _parse_gt_key(session, sequence, gt_dim_key):
        if not sequence:
            gt_array_key = f"{session}__{gt_dim_key}"
        else:
            gt_array_key = f"{session}__{sequence}__{gt_dim_key}"
        return gt_array_key

    @staticmethod
    def _create_reconciliation_maps(
        pred_desc: dict, gt_desc: dict
    ) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]]]:
        """
        Create reconciliation maps for axis3 and axis4 between prediction and GT.

        This method is required when predictions and GT do not align perfectly,
        such as when they have different keypoint sets.

        Args:
            pred_desc (dict): Prediction data description.
            gt_desc (dict): Ground truth data description.

        Returns:
            Tuple[Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]]]:
                Tuple of (pred_rec_map, gt_rec_map).
        """
        pred_rec_map, gt_rec_map = {}, {}

        # Get labels for both axes from both descriptions
        pred_ax3 = tuple(pred_desc.get("axis3", []))
        gt_ax3 = tuple(gt_desc.get("axis3", []))
        pred_ax4 = tuple(pred_desc.get("axis4", []))
        gt_ax4 = tuple(gt_desc.get("axis4", []))

        # Find the common labels (overlap) for each axis
        common_set_ax3 = set(pred_ax3) & set(gt_ax3)
        common_set_ax4 = set(pred_ax4) & set(gt_ax4)

        # Fail fast if reconciliation is completely impossible on one of the axes.
        if not common_set_ax3 or not common_set_ax4:
            raise ValueError(
                "Prediction and Ground Truth have no common labels on one of "
                f"axis3 or axis4. axis3 preds: {pred_ax3}, GT: {gt_ax3}. "
                f"axis4 preds: {pred_ax4}, GT: {gt_ax4}"
            )
        # For Axis 3: Create map only if lists are different but have overlap.
        if common_set_ax3 and pred_ax3 != gt_ax3:
            common_ax3_ordered = tuple(
                label for label in pred_ax3 if label in common_set_ax3
            )
            pred_rec_map["axis3"] = tuple(
                pred_ax3.index(label) for label in common_ax3_ordered
            )
            gt_rec_map["axis3"] = tuple(
                gt_ax3.index(label) for label in common_ax3_ordered
            )

        # For Axis 4: Create map only if lists are different but have overlap.
        if common_set_ax4 and pred_ax4 != gt_ax4:
            common_ax4_ordered = tuple(
                label for label in pred_ax4 if label in common_set_ax4
            )
            pred_rec_map["axis4"] = tuple(
                pred_ax4.index(label) for label in common_ax4_ordered
            )
            gt_rec_map["axis4"] = tuple(
                gt_ax4.index(label) for label in common_ax4_ordered
            )

        return pred_rec_map, gt_rec_map
