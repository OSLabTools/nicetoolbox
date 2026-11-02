import json

import pytest

from nicetoolbox.configs.schemas.detectors_instances_configs import BaseAlgorithmConfig, DetectorInputConfig
from nicetoolbox.detectors.detector_inputs import (
    BaseDetectorInput,
    JsonDetectorInput,
    NpzDetectorInput,
    load_detector_inputs,
)
from nicetoolbox_core.data.array_schema import VECTOR_2D_CONF_PER_LABEL
from nicetoolbox_core.data.json_schema import AudioDiarization
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


# ---------------------------------------------------------------------------
# JSON inputs
# ---------------------------------------------------------------------------


def _write_regions_json(folder, algorithm="audio_subject_detector", speaker="person_left"):
    """Write an upstream audio_diarization JSON with one segment on one track."""
    payload = {
        "meta": {
            "component": "audio_diarization",
            "algorithm": algorithm,
            "subsequence_start": 0.0,
            "subsequence_length": 10.0,
        },
        "tracks": {"left_mic": {"segments": [{"speaker": speaker, "start": 0.0, "end": 10.0}]}},
    }
    (folder / f"{algorithm}.json").write_text(json.dumps(payload))


class TestJson:
    _cfg = {"subjects": DetectorInputConfig(component="audio_diarization", algorithm="audio_subject_detector")}
    _context = FakeContext({"audio_subject_detector": _upstream_config()})

    def test_json_input_parsed_into_schema(self, tmp_path):
        _write_regions_json(tmp_path)
        declared = [JsonDetectorInput("audio_diarization", "subjects", schema=AudioDiarization)]

        resolved = load_detector_inputs(declared, self._cfg, FakeIO(tmp_path), self._context)

        regions = resolved["subjects"].json
        assert isinstance(regions, AudioDiarization)
        assert regions.tracks["left_mic"].segments[0].speaker == "person_left"
        assert resolved["subjects"].array is None
        assert resolved["subjects"].file_path == tmp_path / "audio_subject_detector.json"

    def test_json_input_without_schema_is_raw_dict(self, tmp_path):
        _write_regions_json(tmp_path)
        declared = [JsonDetectorInput("audio_diarization", "subjects")]

        resolved = load_detector_inputs(declared, self._cfg, FakeIO(tmp_path), self._context)

        assert resolved["subjects"].json["tracks"]["left_mic"]["segments"][0]["speaker"] == "person_left"

    def test_json_input_missing_file_raises(self, tmp_path):
        declared = [JsonDetectorInput("audio_diarization", "subjects", schema=AudioDiarization)]
        with pytest.raises(FileNotFoundError, match="audio_subject_detector.json"):
            load_detector_inputs(declared, self._cfg, FakeIO(tmp_path), self._context)

    def test_json_input_schema_mismatch_raises(self, tmp_path):
        (tmp_path / "audio_subject_detector.json").write_text(json.dumps({"tracks": {}}))  # no meta
        declared = [JsonDetectorInput("audio_diarization", "subjects", schema=AudioDiarization)]
        with pytest.raises(ValueError, match="schema validation"):
            load_detector_inputs(declared, self._cfg, FakeIO(tmp_path), self._context)
