# collection of all method and feature detectors configurations
# new detectors should be added and registered here

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from nicetoolbox_core.input_recipes import InputRecipes

from ..models.models_registry import ModelsRegistry

# registry for detectors
DETECTORS_REGISTRY = ModelsRegistry()
detector_config = DETECTORS_REGISTRY.register


# =============================================================================
# Base Algorithm Instance Config
# =============================================================================


class DetectorInputConfig(BaseModel):
    """
    Input dependency from another detector. Support both npz and non-npz based components.
    """

    component: str
    algorithm: str
    npz_key: Optional[str] = None  # None for non-npz component


class BaseAlgorithmConfig(BaseModel):
    """
    Shared base for every algorithm config. The TOML key under [algorithms.*]
    is a user-defined instance name (stored in `_instance_name`); the schema
    class is selected via the `algorithm_type` field.
    """

    algorithm_type: str
    template: Optional[str] = None
    inputs: Dict[str, DetectorInputConfig] = Field(default_factory=dict)

    _instance_name: str = PrivateAttr()


# =============================================================================
# Common Runtime Fields
# =============================================================================


class BaseDetectorRuntime(BaseModel):
    """
    Common runtime fields for ALL detectors (method and feature).

    These fields are computed during detector initialization from IO, Data,
    and SequenceRuntimeConfig - NOT from static config files.
    """

    model_config = ConfigDict(extra="forbid")

    # Output paths
    result_folders: Dict[str, str]
    out_folders: Dict[str, str] = Field(default_factory=dict)
    viz_folders: Dict[str, str] = Field(default_factory=dict)

    # Algorithm identity (the user-defined instance name from TOML)
    algorithm: str

    # Run config flag
    visualize: bool

    # Data context
    subjects_descr: List[str]


class MethodDetectorRuntime(BaseDetectorRuntime):
    """
    Runtime fields specific to method detectors.
    Extends BaseDetectorRuntime with subprocess and inference requirements.
    """

    # Root folder of NICE Toolbox installation (cwd)
    # Used to find submodules paths
    nicetoolbox_root: str

    # Logging (needed for subprocess)
    log_file: str
    log_level: str

    # Data context for inference
    calibration: Optional[Dict[str, Any]] = None
    cam_sees_subjects: Dict[str, List[int]]

    # Input recipes for dataloaders in subprocess
    input_recipes: InputRecipes


class FeatureDetectorRuntime(BaseDetectorRuntime):
    """Runtime fields specific to feature detectors."""

    # TODO: deprecate?
    ...


class TriangulationConfig(BaseModel):
    """Stereo triangulation settings for detectors that lift 3d from two views."""

    triangulate: bool
    triangulation_cameras: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_cameras(self):
        if not self.triangulate:
            return self
        if "*" in self.triangulation_cameras:
            raise ValueError("triangulation.triangulation_cameras must list explicit camera names; '*' is not allowed.")
        if len(self.triangulation_cameras) != 2:
            raise ValueError(
                f"triangulation.triangulation_cameras must list exactly 2 cameras when triangulate is true, "
                f"got {len(self.triangulation_cameras)}: {self.triangulation_cameras}."
            )
        if self.triangulation_cameras[0] == self.triangulation_cameras[1]:
            raise ValueError(
                f"triangulation.triangulation_cameras must be two distinct cameras, "
                f"got {self.triangulation_cameras}."
            )
        return self


# ================================================
#                 METHOD DETECTORS
# ================================================


@detector_config("mmpose_2d")
class MMPoseAlgorithmConfig(BaseAlgorithmConfig):
    """
    Static config for MMPose 2D pose estimators.
    Previously six stacked decorators (hrnetw48, vitpose, vitpose_huge,
    rtmpose_l_aic, rtmpose_l_wholebody, rtmpose_m_mpii) — now a single
    schema; the specific model is picked by `pose_config` + `keypoint_mapping`,
    and the produced components are declared via `components`.
    """

    camera_names: str | list[str]
    triangulation: TriangulationConfig
    env_name: str
    save_detector_images: bool
    save_detector_predictions: bool
    device: str
    filtered: bool
    window_length: int
    polyorder: int
    visualize: bool

    pose_config: str
    keypoint_mapping: str
    min_detection_confidence: float
    # Which components this instance produces (e.g. ["body_joints", "hand_joints", "face_landmarks"]
    # for wholebody, ["body_joints"] for body-only models).
    components: List[str]
    required_assets: Dict[str, str] = Field(default_factory=dict)
    # Optional dependency edges for topological sort (same shape as feature detectors).
    input_detector_names: Optional[List[List[str]]] = None

    # Nested runtime config class - extends base with MMPose-specific fields
    class RuntimeConfig(MethodDetectorRuntime):
        """MMPose-specific runtime fields."""

        prediction_folders: Dict[str, str]
        image_folders: Dict[str, str]
        keypoints_indices: Dict[str, List[int]]
        keypoints_description: Dict[str, List[str]]
        # Video / subprocess (explicit in run_config.toml — no guessed defaults in inference scripts).
        fps: int


@detector_config("motionbert")
class MotionbertAlgorithmConfig(BaseAlgorithmConfig):
    """Static config for MotionBERT 3D lifting (2D NPZ input); kept separate from 2D MMPose algorithms."""

    camera_names: str | list[str]
    env_name: str
    save_detector_images: bool
    save_detector_predictions: bool
    device: str
    filtered: bool
    window_length: int
    polyorder: int
    visualize: bool

    keypoint_mapping: str
    min_detection_confidence: float
    # 3D lifter weights only; 2D detector assets are merged from input_detector_names at init.
    required_assets: Dict[str, str] = Field(default_factory=dict)
    # Must include exactly one upstream body_joints producer (NPZ path, 2D pose assets, etc.).
    input_detector_names: List[List[str]]
    # Optional MMPose 3D frame-export layout (serialized to run_config; read with strict keys in subprocess).
    mmpose_3d_nice_multi_layout: bool = True
    pelvis_sep_scale: float = 1.0

    class RuntimeConfig(MMPoseAlgorithmConfig.RuntimeConfig):
        """Runtime fields for MotionBERT (NPZ path, merged 2D+3D assets, derived 2D lifter inputs)."""

        pose_config: str
        motionbert_2d_pose_det_dataset: str
        motionbert_2d_coco_body_indices: List[int]
        motionbert_2d_keypoints_npz: str
        required_assets: Dict[str, str]


@detector_config("spiga")
class SpigaConfig(BaseAlgorithmConfig):
    env_name: str
    camera_names: str | list[str]
    log_frame_idx_interval: int
    batch_size: int
    visualize: bool
    spiga_dataset: str

    required_assets: Dict[str, str] = Field(default_factory=dict)

    class RuntimeConfig(MethodDetectorRuntime):
        """SPIGA-specific runtime fields."""

        face_landmarks_description: List[str]


@detector_config("py_feat")
class PyFeatConfig(BaseAlgorithmConfig):
    camera_names: str | list[str]
    env_name: str
    log_frame_idx_interval: int
    batch_size: int
    visualize: bool
    required_assets: Dict[str, str] = Field(default_factory=dict)


@detector_config("eth_xgaze")
class EthXGazeConfig(BaseAlgorithmConfig):
    camera_names: str | list[str]
    env_name: str
    filtered: bool
    window_length: int
    polyorder: int
    visualize: bool
    visualize_native: bool
    required_assets: Dict[str, str] = Field(default_factory=dict)


@detector_config("whisperx")
class WhisperXConfig(BaseAlgorithmConfig):
    env_name: str
    visualize: bool
    track_names: str | list[str]
    fallback_camera: Optional[str] = None

    model_size: str
    compute_type: str
    batch_size: int
    language: Optional[str]
    vad_onset: float
    vad_offset: float
    alignment_model_name: str
    hf_weights_cache_dir: str = Field(
        default="<assets>/whisperx",
        description="Directory for WhisperX / Hugging Face weight caches (created at inference if missing).",
    )
    required_assets: Dict[str, str] = Field(default_factory=dict)


@detector_config("sam_3d_body")
class Sam3dBodyConfig(BaseAlgorithmConfig):
    """SAM 3D Body (Hugging Face weights; set hugging_face_token in machine_specific_paths.toml)."""

    camera_names: str | list[str]
    triangulation: TriangulationConfig
    env_name: str
    device: str
    visualize: bool

    hf_repo_id: str = "facebook/sam-3d-body-dinov3"
    sam3d_repo_path: str = ""
    detector_name: str = "vitdet"
    segmentor_name: str = "sam2"
    fov_name: str = "moge2"
    inference_type: str = "body"
    save_vertices: bool = True
    visualize_mesh: bool = True
    visualize_mesh_interactive: bool = False
    interactive_mesh_frame_stride: int = 1
    interactive_mesh_camera_index: int = 0
    interactive_mesh_prefer_world: bool = True
    interactive_mesh_subject_spacing: Optional[float] = None
    interactive_mesh_line_width: int = 1
    # Keep interactive HTML browser-sized (uncapped wire × frames can exceed ~500 MB).
    interactive_mesh_max_edges: int = 12_000
    interactive_mesh_max_frames: int = 48
    interactive_mesh_plotlyjs_cdn: bool = False
    bbox_sort_left_to_right: bool = True
    temporal_smooth: bool = True
    smooth_window_length: int = 7
    smooth_polyorder: int = 2
    world_align_keypoints_3d: bool = True
    cross_view_consistency: bool = True
    triangulation_min_detection_confidence: float = 0.6
    keypoint_mapping: str = "sam_3d_body_mhr"
    required_assets: Dict[str, str] = Field(default_factory=dict)


@detector_config("crisper_whisper")
class CrisperWhisperConfig(BaseAlgorithmConfig):
    track_names: str | list[str]
    env_name: str
    visualize: bool
    fallback_camera: Optional[str] = None

    batch_size: int
    chunk_length_s: float
    stride_length_s: float
    vad_onset: float
    vad_offset: float
    hf_weights_cache_dir: str
    required_assets: Dict[str, str] = Field(default_factory=dict)


# === Add Method detectors HERE ===


# ================================================
#                 FEATURE DETECTORS
# ================================================


@detector_config("gaze_distance_2d")
@detector_config("gaze_distance_3d")
class GazeDistanceConfig(BaseAlgorithmConfig):
    used_keypoints: List[str]
    threshold_look_at: float
    visualize: bool


@detector_config("velocity_body_2d")
@detector_config("velocity_body_3d")
class VelocityConfig(BaseAlgorithmConfig):
    visualize: bool


@detector_config("body_distance_2d")
@detector_config("body_distance_3d")
class BodyDistanceConfig(BaseAlgorithmConfig):
    used_keypoints: List[str]
    visualize: bool


@detector_config("gaze_fusion")
class GazeFusionConfig(BaseAlgorithmConfig):
    fusion_method: str  # "weighted_average" | "mean" | "select_view"
    # Required when fusion_method == "select_view": maps each subject to the camera
    # whose per-view gaze estimate should be adopted as that subject's world gaze.
    subject_view_map: Dict[str, str] = Field(default_factory=dict)
    filtered: bool
    window_length: int
    polyorder: int
    visualize: bool

    @model_validator(mode="after")
    def _check_select_view(self):
        if self.fusion_method == "select_view" and not self.subject_view_map:
            raise ValueError("fusion_method='select_view' requires a non-empty subject_view_map (subject -> camera).")
        return self


# === Add Feature detectors HERE ===


@detector_config("eye_closure_ear")
class EyeClosureEarConfig(BaseAlgorithmConfig):
    visualize: bool
    camera_names: str | list[str]


@detector_config("eye_closure_threshold")
class EyeClosureThresholdConfig(BaseAlgorithmConfig):
    threshold: float
    visualize: bool
    camera_names: List[str]
    min_duration: float = 0.0
    max_duration: Optional[float] = None
