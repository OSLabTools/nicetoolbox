import logging
from typing import Iterable, List

from ..configs.schemas.detectors_config import DetectorsConfig
from .graph import topological_sort


def sort_detectors_order(
    detectors_config: DetectorsConfig,
    selected_algorithms: Iterable[str],
    strict: bool = True,
) -> List[str]:
    """
    Sort detectors algorithms in topological order.

    Args:
        detectors_config: All detectors configurations containing input_detector_names.
        selected_algorithms: Algorithm names to order.
        strict: If True, raise on missing dependencies. If False, log a warning and skip them.

    Raises:
        KeyError: On missing dependencies (only when strict mode).
        ValueError: On circular dependencies.
    """
    # get the dependencies for each selected detector.
    # Edges come from both the legacy `input_detector_names` (component/algorithm pairs)
    # and the new `inputs` table (DetectorInputConfig), so migration can be incremental:
    # non-migrated detectors keep working while migrated ones declare inputs via `inputs`.
    graph = {}
    for algo_name in selected_algorithms:
        config = detectors_config.algorithms[algo_name]
        input_deps = getattr(config, "input_detector_names", None) or []
        # TODO: deprecate input_detector_names
        edges = [pair[1] for pair in input_deps]
        edges += [inp.algorithm for inp in getattr(config, "inputs", {}).values()]
        graph[algo_name] = edges

    missing = []
    try:
        order = topological_sort(graph, missing)
    except ValueError as e:
        cycle = e.args[0]
        raise ValueError(
            f"Circular dependency detected among detectors: {cycle}. "
            f"Check 'input_detector_names' in your detectors config. "
            f"Selected algorithms: {selected_algorithms}"
        ) from None

    # raise if we have any missing deps
    if missing and strict:
        raise KeyError(
            f"Missing dependencies in detector configuration: "
            f"{'; '.join(f'{dep} -> {node}' for node, dep in missing)}. "
            f"Ensure all required detectors are included in the run config, "
            f"or set check_missing_detectors_dependencies = false in run_config.toml."
        )

    # or we are in non-strict mode - just write a warning
    for node, dep in missing:
        logging.warning(
            f"Detector '{node}' depends on '{dep}', which is not in the selected algorithms. "
            f"Assuming output from a previous run."
        )

    return order
