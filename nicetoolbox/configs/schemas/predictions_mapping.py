from typing import Dict, List

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


class Human36mConnections(BaseModel):
    body_joints: List[List[str]]
    hand_joints: List
    face_landmarks: List


class Human36m(BaseModel):
    connections: Human36mConnections
    keypoints_index: Dict[str, int]


class HumanPose(BaseModel):
    coco_wholebody: CocoWholebody
    human36m: Human36m
    bodypart_names: Dict[str, List[str]]
    bone_dict: Dict[str, List[str]]
    joint_diameter_size: Dict[str, float]


class SpigaKeypointsIndex(BaseModel):
    face: Dict[str, List[int]]


class Spiga(BaseModel):
    keypoints_index: SpigaKeypointsIndex


class HeadOrientation(BaseModel):
    spiga: Spiga


class Order(BaseModel):
    keypoints: List[str]
    bones: List[str]
    microactions: List[str]


class Microactions(BaseModel):
    daya: Dict[str, int]


# Top-level keys in predictions_mapping.toml
class PredictionsMappingConfig(BaseModel):
    human_pose: HumanPose
    head_orientation: HeadOrientation
    order: Order
    microactions: Microactions
