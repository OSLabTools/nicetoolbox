"""
Base classes for metrics and metric handlers in the evaluation module. Metric factory
to instantiate handlers based on config.
"""

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..config_schema import EvaluationConfig, MetricTypeConfig
from ..data.discovery import ChunkWorkItem, FrameInfo
from .results_schema import MetricReturnType


class Metric(ABC):
    """
    Abstract base class for a single, stateful metric computation.
    """

    def __init__(self, **kwargs: Any):
        """Initializes the metric. Subclasses can use kwargs for specific setup."""
        _ = kwargs
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric's internal state to prepare for a new run."""
        pass

    @abstractmethod
    def get_axis3(self, meta_chunks: Optional[List[ChunkWorkItem]] = None) -> List[str]:
        """Get the metrics output description. E.g. bone names or joint names."""
        pass

    @abstractmethod
    def update(
        self,
        preds: torch.Tensor,
        gts: torch.Tensor,
        meta_chunks: List[ChunkWorkItem],
        meta_frames: List[FrameInfo],
    ) -> None:
        """Update metric state with a batch of data and its corresponding metadata."""
        pass

    @abstractmethod
    def compute(self) -> Dict[Tuple[str, str, str], MetricReturnType]:
        """Compute the final metric from the stored state and return as a dictionary."""
        pass


class MetricHandler(ABC):
    """
    Abstract base class for a metric handler (e.g., PointCloudMetric).
    A handler creates and manages one or more base Metric instances.
    """

    def __init__(self, cfg: MetricTypeConfig, device: str):
        """
        Initialize the metric handler with its config and device, creating its metrics.

        Args:
            cfg (MetricTypeConfig): Configuration for this metric type.
            device (str): Device to run the metrics on (e.g., 'cpu' or 'cuda').
        """
        if device.startswith("cuda") and not torch.cuda.is_available():
            logging.warning(
                f"Selected CUDA device {device} is not available." " Using cpu instead."
            )
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.cfg = cfg
        self.metrics: Dict[str, Metric] = self._create_metrics()

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the metric handler (e.g., 'point_cloud_metrics')."""
        pass

    @abstractmethod
    def _create_metrics(self) -> Dict[str, Metric]:
        """Instantiate the individual metric objects this handler manages."""
        pass

    def process(self, batch: Dict[str, Any]) -> None:
        """Process a homogeneous, grouped batch of data for all managed metrics."""
        preds = batch["pred"].to(self.device)
        gts = batch.get("gt")
        if gts is not None:
            gts = gts.to(self.device)

        meta_chunk, meta_frames = batch["chunk"], batch["frames"]

        for metric_name, metric in self.metrics.items():
            try:
                metric.update(preds, gts, meta_chunk, meta_frames)
            except Exception as e:
                logging.error(
                    f"Metric '{metric_name}' in handler '{self.name}' failed "
                    f"during update call: {e})",
                    exc_info=True,
                )

    def evaluate(self) -> Dict[Tuple[str, str, str], MetricReturnType]:
        """Compute final results from all managed metrics."""
        all_results = {}
        for metric_name, metric in self.metrics.items():
            try:
                metric_results = metric.compute()
                all_results.update(metric_results)
            except Exception as e:
                logging.error(
                    f"Metric'{metric_name}' in handler '{self.name}' failed "
                    f"during compute call: {e})",
                    exc_info=True,
                )
            finally:
                metric.reset()
        return all_results


class MetricFactory:
    """Static class that instantiates metric handlers based on the evaluation config."""

    @staticmethod
    def create_all(ev_cfg: EvaluationConfig, device: str) -> List[MetricHandler]:
        """
        Create all metric handlers based on the evaluation configuration.

        Args:
            ev_cfg (EvaluationConfig): The evaluation config listing metric types.
            device (str): The device to run the metrics on (e.g., 'cpu' or 'cuda').

        Returns:
            List[MetricHandler]: A list of metric handlers with instantiated metrics.
        """
        handlers: List[MetricHandler] = []
        metric_configs = ev_cfg.metric_types

        handler_map = {
            "point_cloud_metrics": ".point_cloud.PointCloudMetric",
            "keypoint_metrics": ".keypoint.KeypointMetric",
            "categorical_metrics": ".categorical.CategoricalMetric",
        }
        for name, cfg in metric_configs.items():
            if name in handler_map:
                try:
                    module_path, class_name = handler_map[name].rsplit(".", 1)
                    module = importlib.import_module(module_path, package=__package__)
                    handler_class = getattr(module, class_name)
                    handlers.append(handler_class(cfg, device))
                except ImportError as e:
                    logging.warning(
                        f"Could not import handler for '{name}': {e}. Skipping.",
                        exc_info=True,
                    )

        return handlers
