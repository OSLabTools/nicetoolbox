import logging
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from spiga.demo.visualize.plotter import Plotter
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

from nicetoolbox_core.data.loaded_array import load_array_from_path, select_array
from nicetoolbox_core.entrypoint import run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

# Constants
RAW_INFERENCE_PICKLE_NAME = "spiga_inference_raw.pkl"


def _visulize(image_bgr, features, bboxes, plotter, out_folder, camera_name, real_frame_idx):
    canvas = image_bgr.copy()
    for i, (x0, y0, bw, bh) in enumerate(bboxes):
        headpose = np.array(features["headpose"][i])
        canvas = plotter.landmarks.draw_landmarks(canvas, np.array(features["landmarks"][i]))
        canvas = plotter.hpose.draw_headpose(
            canvas,
            [x0, y0, x0 + bw, y0 + bh],
            headpose[:3],  # euler angles
            headpose[3:],  # translation
            euler=True,
        )

    out_dir = os.path.join(out_folder, "images", camera_name)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{real_frame_idx:09d}.jpg"), canvas)


@run_inference_entrypoint
def spiga_inference(config: dict) -> None:
    # Enable cuDNN optimization (for PyTorch CNNs)
    torch.backends.cudnn.benchmark = True

    logging.info("Running SPIGA head orientation detection!")

    # (1) Access config parameters
    camera_names = config["camera_names"]
    visualize_native = config["visualize_native"]
    out_folder = config["out_folders"]["head_orientation"]

    # (2) Prepare data loader
    dataloader = ImagePathsByFrameIndexLoader(config=config, expected_cameras=camera_names)

    # (3) Load the upstream face boxes: (subjects, cameras, frames, x0/y0/x1/y1/confidence).
    # NaN marks a subject slot with no detected face in that camera-frame.
    face_bbox_npz = Path(config["face_bbox_npz"])
    face_bbox = load_array_from_path(face_bbox_npz, config["face_bbox_npz_key"])
    # Only select cameras that we need
    face_bbox = select_array(face_bbox, cameras=camera_names)
    bboxes_data = face_bbox.data
    # Taken from the array's own axis, so a subject name always matches the row it labels.
    bbox_subjects = face_bbox.axes.subjects

    # (4) Initialize SPIGA model
    spiga_config = ModelConfig("wflw", load_model_url=False)
    model_weights_file = Path(config["required_assets"]["model_weights_path"])
    model_weights_folder = model_weights_file.parent
    spiga_config.model_weights_path = model_weights_folder

    spiga_model = SPIGAFramework(spiga_config)
    plotter = Plotter()

    # (6) Inference loop
    per_frame_outputs = []
    for frame_idx, (real_frame_idx, frame_paths_per_camera) in enumerate(dataloader):  # for each frame
        detections_per_camera = {}  # store detections per camera
        for camera_name, image_path in frame_paths_per_camera.items():  # for each camera
            cam_idx = camera_names.index(camera_name)

            # (A) Load images per camera
            image_bgr = cv2.imread(image_path)

            # (B) Collect this camera's face boxes, one detection per subject that has one.
            detections = []
            for subj_idx, subject_name in enumerate(bbox_subjects):
                # we skip all bbox that has any nan (subject not detected)
                box = bboxes_data[subj_idx, cam_idx, frame_idx]
                if np.isnan(box[:4]).any():
                    continue
                # convert it to the spiga [x, y, w, h] notation
                x0, y0, x1, y1 = (float(value) for value in box[:4])
                w, h = x1 - x0, y1 - y0
                # save them to the lost
                detections.append({"subject": subject_name, "bbox_xywh": [x0, y0, w, h]})

            if not detections:
                logging.debug(f"No upstream faces for frame {real_frame_idx}, camera {camera_name}")
                continue

            # (C) Run SPIGA given the image and those bboxes
            # spiga need flat boudning boxes list, so we repackage it
            bboxes = [detection["bbox_xywh"] for detection in detections]
            # spiga inference
            features = spiga_model.inference(image_bgr, bboxes)

            # (D) Store the raw per-face results as-is; post_inference structures them.
            # indexing allow us to keep identity matching
            for i, detection in enumerate(detections):
                # headpose is euler angles + translation
                # TODO: expose the translation. It is the mean face model's origin - which IS the nose tip,
                # index 54 in mean_face_3D_98.txt - expressed in the bbox-crop camera, so it is not metric and
                # not a world position. Only the native visualization can use it. For the toolbox the ray origin
                # should be the triangulated 3d nose instead: same point on the face, but actually measured.
                detection["head_rotation"] = features["headpose"][i][:3]
                detection["landmarks_2d"] = features["landmarks"][i]
            detections_per_camera[camera_name] = detections

            # (E) Optionally save SPIGA's own overlay
            if visualize_native:
                _visulize(
                    image_bgr=image_bgr,
                    features=features,
                    bboxes=bboxes,
                    plotter=plotter,
                    out_folder=out_folder,
                    camera_name=camera_name,
                    real_frame_idx=real_frame_idx,
                )

        per_frame_outputs.append(detections_per_camera)

    # (7) Save the raw pack
    raw_path = os.path.join(out_folder, RAW_INFERENCE_PICKLE_NAME)
    logging.info(f"Saving SPIGA raw inference output to {raw_path}")
    with open(raw_path, "wb") as raw_file:
        pickle.dump(per_frame_outputs, raw_file)
    logging.info("SPIGA raw inference output saved successfully.")
