from typing import Dict, List, Optional

from pydantic import BaseModel


class CocoWholebodyKeypointsIndex(BaseModel):
    body: Dict[str, int]
    foot: Dict[str, int]
    face: Dict[str, List[int]]
    hand: Dict[str, List[int]]


class CocoWholebodyBodypartIndex(BaseModel):
    head: List[int]
    upper_body: List[int]
    lower_body: List[int]


class CocoWholebodyConnections(BaseModel):
    body_joints: List[List[str]]
    hand_joints: List[List[str]]
    face_landmarks: List


class CocoWholebody(BaseModel):
    keypoints_index: CocoWholebodyKeypointsIndex
    bodypart_index: CocoWholebodyBodypartIndex
    connections: CocoWholebodyConnections


class Human36mKeypointsIndex(BaseModel):
    body: Dict[str, int]
    foot: Dict[str, int]
    face: Dict[str, List[int]]
    hand: Dict[str, List[int]]


class Human36mConnections(BaseModel):
    body_joints: List[List[str]]
    hand_joints: List
    face_landmarks: List


class Human36mBodypartIndex(BaseModel):
    head: List[int]
    upper_body: List[int]
    lower_body: List[int]


class Human36m(BaseModel):
    keypoints_index: Human36mKeypointsIndex
    bodypart_index: Human36mBodypartIndex
    connections: Human36mConnections


class MpiiKeypointsIndex(BaseModel):
    body: Dict[str, int]


class MpiiConnections(BaseModel):
    body_joints: List[List[str]]


class MpiiBodypartIndex(BaseModel):
    head: List[int]
    upper_body: List[int]
    lower_body: List[int]


class Mpii(BaseModel):
    keypoints_index: MpiiKeypointsIndex
    bodypart_index: MpiiBodypartIndex
    connections: MpiiConnections


class Sam3dBodyMhrKeypointsIndex(BaseModel):
    body: Dict[str, int]
    foot: Dict[str, int]
    hand: Dict[str, int]
    extra: Dict[str, int]


class Sam3dBodyMhrBodypartIndex(BaseModel):
    head: List[int]
    upper_body: List[int]
    lower_body: List[int]


class Sam3dBodyMhrConnections(BaseModel):
    body_joints: List[List[str]]
    hand_joints: List[List[str]]
    face_landmarks: List[List[str]]


class Sam3dBodyMhr(BaseModel):
    keypoints_index: Sam3dBodyMhrKeypointsIndex
    bodypart_index: Sam3dBodyMhrBodypartIndex
    connections: Sam3dBodyMhrConnections


class Sam3dBodyMhrEvaluation(Sam3dBodyMhr):
    """Optional evaluation-time MHR↔semantic mapping; duplicate or specialize vs sam_3d_body_mhr in TOML."""

    pass


class HumanPose(BaseModel):
    coco_wholebody: CocoWholebody
    human36m: Human36m
    mpii: Mpii
    sam_3d_body_mhr: Sam3dBodyMhr
    sam_3d_body_mhr_evaluation: Optional[Sam3dBodyMhrEvaluation] = None
    bodypart_names: Dict[str, List[str]]
    bone_dict: Dict[str, List[str]]
    joint_diameter_size: Dict[str, float]


class SpigaKeypointsIndex(BaseModel):
    face: Dict[str, List[int]]


class Spiga(BaseModel):
    keypoints_index: SpigaKeypointsIndex


class HeadOrientation(BaseModel):
    spiga: Spiga


# Top-level keys in predictions_mapping.toml
class PredictionsMappingConfig(BaseModel):
    human_pose: HumanPose
    head_orientation: HeadOrientation
