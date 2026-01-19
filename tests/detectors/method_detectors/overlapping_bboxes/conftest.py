from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class MockDataScenario:
    """
    Configuration describing one mock data scenario
    for testing detector outputs.
    """

    num_subjects: int
    num_cameras: int
    num_frames: int
    # Optional list of tuples (subject, camera, frame) where data is missing
    missing: list = None


def make_mock_bbox_data(cfg: MockDataScenario, seed: int = 0) -> np.ndarray:
    """
    Create deterministic mock bbox data.
    Shape:
        Bounding boxes (x1, y1, x2, y2, confidence)
        Shape: (S, C, F, 1, 5)
    """
    rng = np.random.default_rng(seed)
    H, W = 720, 1280
    S, C, F = cfg.num_subjects, cfg.num_cameras, cfg.num_frames

    # Create random but valid bounding boxes
    x1 = rng.uniform(W * 0.2, W * 0.4, size=(S, C, F))
    y1 = rng.uniform(H * 0.2, H * 0.4, size=(S, C, F))
    x2 = x1 + rng.uniform(60, 160, size=(S, C, F))
    y2 = y1 + rng.uniform(60, 160, size=(S, C, F))

    x2 = np.clip(x2, 0, W)
    y2 = np.clip(y2, 0, H)

    conf = rng.uniform(0.7, 1.0, size=(S, C, F))

    data = np.empty((S, C, F, 1, 5), dtype=np.float32)
    data[..., 0, 0] = x1
    data[..., 0, 1] = y1
    data[..., 0, 2] = x2
    data[..., 0, 3] = y2
    data[..., 0, 4] = conf

    # Apply missing entries as NaNs
    if cfg.missing:
        for s, c, f in cfg.missing:
            if 0 <= s < S and 0 <= c < C and 0 <= f < F:
                data[s, c, f, 0, :] = np.nan
    return data


SCENARIOS = [
    MockDataScenario(num_subjects=1, num_cameras=1, num_frames=5),
    MockDataScenario(num_subjects=2, num_cameras=1, num_frames=6, missing=[(0, 0, 1), (0, 0, 3)]),
    MockDataScenario(num_subjects=3, num_cameras=2, num_frames=5),
    MockDataScenario(num_subjects=3, num_cameras=4, num_frames=8, missing=[(1, 2, 2), (1, 2, 5)]),
]


@pytest.fixture(params=SCENARIOS, name="mock_output_data")
def mock_output_data_fixture(request):
    """
    Pytest fixture returning synthetic mock output data.
    """
    cfg: MockDataScenario = request.param
    return make_mock_bbox_data(cfg)
