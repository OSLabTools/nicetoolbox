# collection of all method and feature detectors configurations
# new detectors should be added and registred here

from typing import List

from pydantic import BaseModel, Field

from ..models.models_registry import ModelsRegistry

# registries for detectors and frameworks
DETECTORS_REGISTRY = ModelsRegistry()
FRAMEWORKS_REGISTRY = ModelsRegistry()
detector_config = DETECTORS_REGISTRY.register
framework_config = FRAMEWORKS_REGISTRY.register

# ================================================
#                 METHOD DETECTORS
# ================================================


@framework_config("mmpose")
class FrameworksMMPoseConfig(BaseModel):
    input_data_format: str
    camera_names: List[str]
    env_name: str
    multi_person: bool
    save_images: bool
    resolution: List[int]
    device: str
    filtered: bool
    window_length: int
    polyorder: int
    # python identifier cannot start with a number (using alias)
    results_3d: bool = Field(alias="3d_results")


@detector_config("hrnetw48")
@detector_config("vitpose")
class MMPoseAlgorithmConfig(FrameworksMMPoseConfig):
    framework: str
    pose_config: str
    pose_checkpoint: str
    detection_config: str
    detection_checkpoint: str
    keypoint_mapping: str
    min_detection_confidence: float


@detector_config("py_feat")
class PyFeatConfig(BaseModel):
    input_data_format: str
    camera_names: List[str]
    env_name: str
    log_frame_idx_interval: int
    batch_size: int


@detector_config("multiview_eth_xgaze")
class MultiViewETHXGazeConfig(BaseModel):
    input_data_format: str
    camera_names: List[str]
    env_name: str
    shape_predictor_filename: str
    face_model_filename: str
    pretrained_model_filename: str
    face_detector_filename: str
    log_frame_idx_interval: int
    filtered: bool
    window_length: int
    polyorder: int


@detector_config("spiga")
class SpigaConfig(BaseModel):
    input_data_format: str
    camera_names: List[str]
    env_name: str
    log_frame_idx_interval: int
    batch_size: int
    # model_config is reserved variable in pydantic model
    model_configuration: str = Field(alias="model_config")


# ================================================
#                 FEATURE DETECTORS
# ================================================


@detector_config("gaze_distance")
class GazeDistanceConfig(BaseModel):
    input_detector_names: List[List[str]]
    keypoint_mapping: str
    threshold_look_at: float


@detector_config("velocity_body")
class VelocityConfig(BaseModel):
    input_detector_names: List[List[str]]


@detector_config("body_angle")
class BodyAngleConfig(BaseModel):
    input_detector_names: List[List[str]]
    used_keypoints: List[List[str]]


@detector_config("body_distance")
class BodyDistanceConfig(BaseModel):
    input_detector_names: List[List[str]]
    used_keypoints: List[str]


@detector_config("gaze_fusion")
class GazeFusionConfig(BaseModel):
    input_detector_names: List[List[str]]
    fusion_method: str
    filtered: bool
    window_length: int
    polyorder: int
    ensemble_enabled: bool
