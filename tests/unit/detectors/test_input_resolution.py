import pytest

from nicetoolbox.configs.schemas.detectors_instances_configs import BaseAlgorithmConfig, DetectorInputConfig
from nicetoolbox.detectors.detector_inputs import BaseDetectorInput, NpzDetectorInput, load_detector_inputs
from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL
from tests.unit.data.conftest import make_npz

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeIO:
    """Resolves every (component, algorithm) to the same result folder."""

    def __init__(self, folder):
        self.folder = folder

    def get_detector_output_folder(self, _component, _algorithm, _token):
        return self.folder


class FakeContext:
    """Returns a stub upstream config for any algorithm in the run."""

    def __init__(self, configs):
        self._configs = configs

    def get_detector_config(self, algorithm_name):
        return self._configs[algorithm_name]


def _upstream_config() -> BaseAlgorithmConfig:
    return BaseAlgorithmConfig(algorithm_type="body_joints_stub")


def _make_pose_npz(folder, algorithm="hrnetw48", key="2d_filtered"):
    """Write an upstream pose NPZ with a (…, x, y, confidence) data axis."""
    return make_npz(
        folder / f"{algorithm}.npz",
        key=key,
        data=("coordinate_x", "coordinate_y", "confidence_score"),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestResolveRequired:
    def test_required_present_loads_array(self, tmp_path):
        _make_pose_npz(tmp_path)
        declared = [NpzDetectorInput("body_joints", "pose", schema=VECTOR_2D_CONF_PER_LABEL)]
        cfg = {"pose": DetectorInputConfig(component="body_joints", algorithm="hrnetw48", npz_key="2d_filtered")}

        resolved = load_detector_inputs(declared, cfg, FakeIO(tmp_path), FakeContext({"hrnetw48": _upstream_config()}))

        assert resolved["pose"].array is not None
        assert resolved["pose"].array.axes.data == ["coordinate_x", "coordinate_y", "confidence_score"]

    def test_upstream_config_populated(self, tmp_path):
        _make_pose_npz(tmp_path)
        upstream = _upstream_config()
        declared = [NpzDetectorInput("body_joints", "pose")]
        cfg = {"pose": DetectorInputConfig(component="body_joints", algorithm="hrnetw48", npz_key="2d_filtered")}

        resolved = load_detector_inputs(declared, cfg, FakeIO(tmp_path), FakeContext({"hrnetw48": upstream}))

        assert resolved["pose"].upstream_config is upstream
        assert resolved["pose"].algorithm == "hrnetw48"
        assert resolved["pose"].npz_key == "2d_filtered"


# ---------------------------------------------------------------------------
# Optional inputs
# ---------------------------------------------------------------------------


class TestOptional:
    def test_optional_not_configured_is_omitted(self, tmp_path):
        # `optional` means the input BLOCK may be omitted from the config: it is simply left out
        # of the resolved map (no entry, no raise).
        declared = [NpzDetectorInput("validity_mask", "mask", optional=True)]

        resolved = load_detector_inputs(declared, {}, FakeIO(tmp_path), FakeContext({}))

        assert "mask" not in resolved

    def test_optional_configured_but_missing_file_raises(self, tmp_path):
        # Once configured, an optional input must resolve fully: a missing NPZ is an error.
        declared = [NpzDetectorInput("validity_mask", "mask", optional=True)]
        cfg = {"mask": DetectorInputConfig(component="validity_mask", algorithm="masker", npz_key="mask")}

        with pytest.raises(FileNotFoundError):
            load_detector_inputs(declared, cfg, FakeIO(tmp_path), FakeContext({"masker": _upstream_config()}))

    def test_optional_configured_but_missing_key_raises(self, tmp_path):
        # Once configured, an optional input must resolve fully: a missing npz_key is an error.
        _make_pose_npz(tmp_path, key="2d_filtered")
        declared = [NpzDetectorInput("body_joints", "pose", optional=True)]
        cfg = {"pose": DetectorInputConfig(component="body_joints", algorithm="hrnetw48", npz_key="does_not_exist")}

        with pytest.raises(ValueError, match="does_not_exist"):
            load_detector_inputs(declared, cfg, FakeIO(tmp_path), FakeContext({"hrnetw48": _upstream_config()}))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_required_missing_file_raises(self, tmp_path):
        declared = [NpzDetectorInput("body_joints", "pose")]
        cfg = {"pose": DetectorInputConfig(component="body_joints", algorithm="hrnetw48", npz_key="2d_filtered")}

        with pytest.raises(FileNotFoundError):
            load_detector_inputs(declared, cfg, FakeIO(tmp_path), FakeContext({"hrnetw48": _upstream_config()}))

    def test_required_missing_key_raises(self, tmp_path):
        _make_pose_npz(tmp_path, key="2d_filtered")
        declared = [NpzDetectorInput("body_joints", "pose")]
        cfg = {"pose": DetectorInputConfig(component="body_joints", algorithm="hrnetw48", npz_key="does_not_exist")}

        with pytest.raises(ValueError, match="does_not_exist"):
            load_detector_inputs(declared, cfg, FakeIO(tmp_path), FakeContext({"hrnetw48": _upstream_config()}))

    def test_schema_mismatch_raises(self, tmp_path):
        # File has a 2-label data axis, but schema demands x/y/confidence.
        make_npz(tmp_path / "hrnetw48.npz", key="2d_filtered", data=("coordinate_x", "coordinate_y"))
        declared = [NpzDetectorInput("body_joints", "pose", schema=VECTOR_2D_CONF_PER_LABEL)]
        cfg = {"pose": DetectorInputConfig(component="body_joints", algorithm="hrnetw48", npz_key="2d_filtered")}

        with pytest.raises(ValueError, match="schema validation"):
            load_detector_inputs(declared, cfg, FakeIO(tmp_path), FakeContext({"hrnetw48": _upstream_config()}))

    def test_declared_input_missing_from_config_raises(self, tmp_path):
        declared = [NpzDetectorInput("body_joints", "pose")]
        with pytest.raises(KeyError, match="pose"):
            load_detector_inputs(declared, {}, FakeIO(tmp_path), FakeContext({}))


# ---------------------------------------------------------------------------
# Non-npz inputs
# ---------------------------------------------------------------------------


class TestNonNpz:
    def test_base_input_resolves_without_array(self, tmp_path):
        # A plain BaseDetectorInput (no npz_key/schema) resolves config only, no array load.
        declared = [BaseDetectorInput("audio_transcription", "transcript")]
        cfg = {"transcript": DetectorInputConfig(component="audio_transcription", algorithm="whisperx")}

        resolved = load_detector_inputs(declared, cfg, FakeIO(tmp_path), FakeContext({"whisperx": _upstream_config()}))

        assert resolved["transcript"].array is None
        assert resolved["transcript"].npz_key is None
        assert resolved["transcript"].algorithm == "whisperx"
