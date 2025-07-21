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
import toml
import torch
from insightface.app import FaceAnalysis

# --- Add submodule path ---
top_level_dir = Path(__file__).resolve().parents[4]
sys.path.append(str(top_level_dir) + "/submodules/SPIGA")

# --- Now import SPIGA submodule modules ---
from spiga.demo.visualize.plotter import Plotter  # noqa: E402
from spiga.inference.config import ModelConfig  # noqa: E402
from spiga.inference.framework import SPIGAFramework  # noqa: E402

# Enable cuDNN optimization (for PyTorch CNNs)
torch.backends.cudnn.benchmark = True


def main(config: dict) -> None:
    logging.basicConfig(
        filename=config["log_file"],
        level=config["log_level"],
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s",
    )
    logging.info("Running SPIGA head orientation detection!")

    frames_list = config.get("frames_list")
    camera_names = config["camera_names"]
    subjects_descr = config["subjects_descr"]
    cam_sees_subjects = config["cam_sees_subjects"]
    camera_order = config["camera_order"]
    n_frames = len(frames_list)
    n_cams = len(camera_names)

    # Detect if GPU is available for ONNXRuntime
    available_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logging.info("Using GPU (CUDAExecutionProvider) for InsightFace.")
    else:
        providers = ["CPUExecutionProvider"]
        logging.info("Using CPU for InsightFace.")

    # Initialize face detector and SPIGA model
    face_detector = FaceAnalysis(name="buffalo_l", providers=providers)
    face_detector.prepare(ctx_id=0)

    spiga_model = SPIGAFramework(ModelConfig(config.get("model_config", "wflw")))
    plotter = Plotter()

    headpose_vectors = np.zeros(
        (len(subjects_descr), n_cams, n_frames, 6)
    )  # [start_x, start_y, end_x, end_y, confidence]
    bbox_vectors = np.zeros(
        (len(subjects_descr), n_cams, n_frames, 4)
    )  # [x0, y0, bw, bh]
    frame_indices = []

    for frame_idx, frame_paths in enumerate(frames_list):
        frame_name = os.path.splitext(os.path.basename(frame_paths[0]))[0]
        frame_indices.append(frame_name)

        for cam_i, image_path in enumerate(frame_paths):
            if cam_i >= len(frame_paths):
                continue

            cam_name = camera_order[cam_i]
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                logging.warning(f"Could not read image at {image_path}")
                continue

            faces = face_detector.get(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            if not faces:
                logging.info(f"No faces found in frame {frame_idx}, camera {cam_name}")
                continue
            faces_sorted = sorted(faces, key=lambda face: face.bbox[0])
            bboxes = []
            for face in faces_sorted:
                x1, y1, x2, y2 = face.bbox
                bboxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

            # Run SPIGA on the image and bboxes
            features = spiga_model.inference(image_bgr, bboxes)
            canvas = image_bgr.copy()

            for i, bbox in enumerate(bboxes):
                x0, y0, bw, bh = bbox
                euler_yzx = np.array(features["headpose"][i][:3])
                translation_vector = np.array(features["headpose"][i][3:])

                subj_map = cam_sees_subjects.get(cam_name, [])
                subj_idx = subj_map[i] if i < len(subj_map) else i

                if subj_idx >= len(subjects_descr):
                    logging.warning(f"Subject index {subj_idx} out of bounds")
                    continue

                headpose_vectors[subj_idx, cam_i, frame_idx, :] = features["headpose"][
                    i
                ]
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
                save_dir = os.path.join(config["out_folder"], cam_name)
                os.makedirs(save_dir, exist_ok=True)
                start_frame = int(config.get("video_start", 0))
                save_path = os.path.join(save_dir, f"{start_frame + frame_idx:05d}.jpg")
                cv2.imwrite(save_path, canvas)

        if (frame_idx != 0) & (frame_idx % config["log_frame_idx_interval"] == 0):
            logging.info(f"Processed frame {frame_idx} / {n_frames}")

    if not config.get("visualize", True):
        logging.info("Visualization turned off")

    # Save .npz
    out = {
        "headpose": headpose_vectors,
        "face_bbox": bbox_vectors,
        "data_description": {
            "headpose": {
                "axis0": subjects_descr,
                "axis1": camera_order,
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
                "axis1": camera_order,
                "axis2": frame_indices,
                "axis3": ["x0", "y0", "width", "height"],
            },
        },
    }
    result_path = os.path.join(
        config["result_folders"]["head_orientation"], f"{config['algorithm']}.npz"
    )
    logging.info(f"Saving SPIGA result to {result_path}")
    np.savez_compressed(result_path, **out)
    logging.info("SPIGA result saved successfully.")


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
        config = toml.load(config_path)
        main(config)
        sys.exit(0)
    except Exception as e:
        logging.critical(f"Script crashed -> {e}", exc_info=True)
        sys.exit(1)
