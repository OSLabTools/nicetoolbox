"""
Run SPIGA head orientation detection on the data.
"""

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
from insightface.app import FaceAnalysis

from nicetoolbox_core.dataloader import ImagePathsByFrameIndexLoader
from nicetoolbox_core.entrypoint import run_inference_entrypoint

# --- Add submodule path ---
top_level_dir = Path(__file__).resolve().parents[4]
sys.path.append(str(top_level_dir) + "/submodules/SPIGA")

# --- Now import SPIGA submodule modules ---
from spiga.demo.visualize.plotter import Plotter  # noqa: E402
from spiga.inference.config import ModelConfig  # noqa: E402
from spiga.inference.framework import SPIGAFramework  # noqa: E402

# Enable cuDNN optimization (for PyTorch CNNs)
torch.backends.cudnn.benchmark = True

# Constants
NUM_FACIAL_LANDMARKS = 98


@run_inference_entrypoint
def spiga_inference(config: dict) -> None:
    """
    Run SPIGA head orientation detection on the provided data.

    1) Setup logging and validate hardware for ONNXRuntime.
    2) Initialize `ImagePathsByFrameIndexLoader` to yield all frame paths per
         frame index.
    3) Pre-allocate standardized NumPy arrays (Subject, Camera, Frame, Data).
    4) For each frame index:
        a. For each camera:
            i.   Load image.
            ii.  Detect faces with InsightFace.
            iii. Run SPIGA inference given image and detected bboxes.
            iv.  Store head pose, landmarks, and bbox results.
            v.   Optionally visualize and save annotated frames.
    5) Save results as compressed .npz file.

    Args:
        config (dict): Configuration dictionary.
    """

    logging.info("Running SPIGA head orientation detection!")

    # (1) Access config parameters

    camera_names = config["camera_names"]
    subjects_descr = config["subjects_descr"]
    cam_sees_subjects = config["cam_sees_subjects"]

    # (2) Prepare data loader
    dataloader = ImagePathsByFrameIndexLoader(config=config, expected_cameras=camera_names)
    n_frames = len(dataloader)
    n_cams = len(camera_names)

    # (3) Detect if GPU is available for ONNXRuntime
    available_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logging.info("Using GPU (CUDAExecutionProvider) for InsightFace.")
    else:
        providers = ["CPUExecutionProvider"]
        logging.info("Using CPU for InsightFace.")

    # (4) Initialize face detector and SPIGA model
    face_detector = FaceAnalysis(name="buffalo_l", providers=providers)
    face_detector.prepare(ctx_id=0)

    spiga_model = SPIGAFramework(ModelConfig(config.get("model_config", "wflw")))
    plotter = Plotter()

    # (5) Prepare results storage
    headpose_vectors = np.zeros((len(subjects_descr), n_cams, n_frames, 6))
    bbox_vectors = np.zeros((len(subjects_descr), n_cams, n_frames, 4))  # [x0, y0, bw, bh]
    landmarks_vectors = np.zeros(
        (
            len(subjects_descr),
            n_cams,
            n_frames,
            NUM_FACIAL_LANDMARKS,
            2,
        )  # coordinate_x & y
    )

    # (6) Inference loop
    frame_indices = []
    for frame_idx, (real_frame_idx, frame_paths_per_camera) in enumerate(dataloader):
        frame_indices.append(f"{real_frame_idx:09d}")

        for camera_name, image_path in frame_paths_per_camera.items():
            cam_i = camera_names.index(camera_name)

            # (A) Load images per camera
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                logging.warning(f"Could not read image at {image_path}")
                continue

            # (B) Detect faces with InsightFace
            faces = face_detector.get(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            if not faces:
                logging.warning(f"No faces found in frame {real_frame_idx}, camera {camera_name}")
                continue

            # (C) Sort faces by x-coordinate to maintain consistent ordering (TODO!)
            faces_sorted = sorted(faces, key=lambda face: face.bbox[0])
            bboxes = []
            for face in faces_sorted:
                x1, y1, x2, y2 = face.bbox
                bboxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

            # (D) Run SPIGA given the image and detected bboxes
            features = spiga_model.inference(image_bgr, bboxes)
            canvas = image_bgr.copy()

            for i, bbox in enumerate(bboxes):
                x0, y0, bw, bh = bbox
                euler_yzx = np.array(features["headpose"][i][:3])
                translation_vector = np.array(features["headpose"][i][3:])

                subj_map = cam_sees_subjects.get(camera_name, [])
                subj_idx = subj_map[i] if i < len(subj_map) else i

                if subj_idx >= len(subjects_descr):
                    logging.warning(f"Subject index {subj_idx} out of bounds")
                    continue

                headpose_vectors[subj_idx, cam_i, frame_idx, :] = features["headpose"][i]

                landmarks_vectors[subj_idx, cam_i, frame_idx, :, :] = np.array(features["landmarks"][i])
                bbox_vectors[subj_idx, cam_i, frame_idx, :] = bbox

                # Draw overlay
                if config.get("visualize", True):
                    landmarks = np.array(features["landmarks"][i])
                    canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
                    canvas = plotter.hpose.draw_headpose(
                        canvas,
                        [x0, y0, x0 + bw, y0 + bh],
                        euler_yzx,
                        translation_vector,
                        euler=True,
                    )

            # Save annotated frame
            if config.get("visualize", True):
                save_dir = os.path.join(config["out_folder"], camera_name)
                os.makedirs(save_dir, exist_ok=True)
                start_frame = int(config.get("video_start", 0))
                save_path = os.path.join(save_dir, f"{start_frame + frame_idx:09d}.jpg")
                cv2.imwrite(save_path, canvas)

        if (frame_idx != 0) & (frame_idx % config["log_frame_idx_interval"] == 0):
            logging.info(f"Processed frame {frame_idx} / {n_frames}")

    if not config.get("visualize", True):
        logging.info("Visualization turned off")

    # Save .npz
    out = {
        "headpose": headpose_vectors,
        "face_bbox": bbox_vectors,
        "face_landmark_2d": landmarks_vectors,
        "data_description": {
            "headpose": {
                "axis0": subjects_descr,
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": [
                    "euler_angle_1",
                    "euler_angle_2",
                    "euler_angle_3",
                    "translation_1",
                    "translation_2",
                    "translation_3",
                ],
            },
            "face_bbox": {
                "axis0": subjects_descr,
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": ["x0", "y0", "width", "height"],
            },
            "face_landmark_2d": {
                "axis0": subjects_descr,
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": config["face_landmarks_description"],
                "axis4": ["coordinate_x", "coordinate_y"],
            },
        },
    }
    result_path = os.path.join(config["result_folders"]["head_orientation"], f"{config['algorithm']}.npz")
    logging.info(f"Saving SPIGA result to {result_path}")
    np.savez_compressed(result_path, **out)
    logging.info("SPIGA result saved successfully.")
