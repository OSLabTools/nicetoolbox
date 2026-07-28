import pytest

from nicetoolbox.configs.schemas.detectors_config import DetectorsConfig
from nicetoolbox.configs.schemas.detectors_instances_configs import BaseAlgorithmConfig, DetectorInputConfig
from nicetoolbox.utils.dependency_sort import sort_detectors_order, topological_sort


def test_no_dependencies():
    """All independent nodes => alphabetical order."""
    graph = {"c": [], "a": [], "b": []}
    assert topological_sort(graph) == ["a", "b", "c"]


def test_simple_chain():
    """A -> B -> C."""
    graph = {"c": ["b"], "b": ["a"], "a": []}
    assert topological_sort(graph) == ["a", "b", "c"]


def test_diamond_dependency():
    """A depends on B and C, both depend on D."""
    graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    result = topological_sort(graph)
    assert result.index("d") < result.index("b")
    assert result.index("d") < result.index("c")
    assert result.index("b") < result.index("a")
    assert result.index("c") < result.index("a")


def test_current_bipartite_structure():
    """Methods before features, matching current behavior."""
    graph = {
        "velocity_body": ["hrnetw48"],
        "gaze_distance": ["eth_xgaze", "hrnetw48"],
        "hrnetw48": [],
        "eth_xgaze": [],
    }
    result = topological_sort(graph)
    assert result.index("hrnetw48") < result.index("velocity_body")
    assert result.index("hrnetw48") < result.index("gaze_distance")
    assert result.index("eth_xgaze") < result.index("gaze_distance")


def test_deterministic_order():
    """Same graph always produces same output, regardless of insertion order."""
    graph_a = {"velocity_body": ["hrnetw48"], "hrnetw48": [], "spiga": []}
    graph_b = {"spiga": [], "hrnetw48": [], "velocity_body": ["hrnetw48"]}
    assert topological_sort(graph_a) == topological_sort(graph_b)


def test_circular_dependency_raises():
    graph = {"a": ["b"], "b": ["a"]}
    with pytest.raises(ValueError) as exc_info:
        topological_sort(graph)
    cycle = exc_info.value.args[0]
    assert "a" in cycle
    assert "b" in cycle


def test_missing_dependency_raises():
    graph = {"a": ["b"]}
    topological_sort(graph)


def test_missing_dependency_collected():
    """When missing list is provided, skip missing deps and collect them."""
    graph = {"a": ["b"], "c": ["d", "a"]}
    missing = []
    result = topological_sort(graph, missing=missing)
    assert result == ["a", "c"]
    assert ("a", "b") in missing
    assert ("c", "d") in missing


def test_missing_dependency_collected_empty():
    """When missing list is provided but no deps are missing, list stays empty."""
    graph = {"a": [], "b": ["a"]}
    missing = []
    result = topological_sort(graph, missing=missing)
    assert result == ["a", "b"]
    assert missing == []


def test_three_level_chain():
    """Arbitrary depth: feature -> feature -> method."""
    graph = {"summary": ["gaze_distance"], "gaze_distance": ["hrnetw48"], "hrnetw48": []}
    assert topological_sort(graph) == ["hrnetw48", "gaze_distance", "summary"]


def test_empty_graph():
    assert topological_sort({}) == []


def test_single_node():
    assert topological_sort({"a": []}) == ["a"]


# ---------------------------------------------------------------------------
# sort_detectors_order: edges from `inputs` and legacy `input_detector_names`
# ---------------------------------------------------------------------------


def _config(**algorithms) -> DetectorsConfig:
    return DetectorsConfig(algorithms=algorithms)


def test_order_from_inputs_table():
    """A detector declaring `inputs` is ordered after its upstream algorithm."""
    cfg = _config(
        hrnetw48=BaseAlgorithmConfig(algorithm_type="mmpose_2d"),
        body_distance_2d=BaseAlgorithmConfig(
            algorithm_type="body_distance_2d",
            inputs={"pose": DetectorInputConfig(component="body_joints", algorithm="hrnetw48", npz_key="2d_filtered")},
        ),
    )
    order = sort_detectors_order(cfg, ["body_distance_2d", "hrnetw48"])
    assert order.index("hrnetw48") < order.index("body_distance_2d")


def test_order_mixes_legacy_and_inputs():
    """Legacy `input_detector_names` and new `inputs` edges resolve in the same graph."""

    class LegacyConfig(BaseAlgorithmConfig):
        input_detector_names: list = []

    cfg = _config(
        hrnetw48=BaseAlgorithmConfig(algorithm_type="mmpose_2d"),
        velocity_body=LegacyConfig(algorithm_type="velocity_body", input_detector_names=[["body_joints", "hrnetw48"]]),
        body_distance_2d=BaseAlgorithmConfig(
            algorithm_type="body_distance_2d",
            inputs={"pose": DetectorInputConfig(component="body_joints", algorithm="hrnetw48", npz_key="2d_filtered")},
        ),
    )
    order = sort_detectors_order(cfg, ["velocity_body", "body_distance_2d", "hrnetw48"])
    assert order.index("hrnetw48") < order.index("velocity_body")
    assert order.index("hrnetw48") < order.index("body_distance_2d")
