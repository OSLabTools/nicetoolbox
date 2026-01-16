"""
Run Py-FEAT on the data using full-camera processing and dynamic batching.
"""

import logging
import os

import numpy as np
import torch
from feat import Detector

from nicetoolbox_core.dataloader import ImagePathsByCameraLoader
from nicetoolbox_core.entrypoint import run_inference_entrypoint

# Constants for output structure
NUM_EMOTIONS = 7
NUM_AUS = 20
NUM_POSES = 3


def get_safe_batch_size(target_batch_size, reserved_vram_gb=4.0, mb_per_image=50):
    """
    Checks if the config target_batch_size is feasible for the current GPU.
    Assumes the maximum size of scaled input image to be 50 MB.
    Assumes pyfeat model sizer loaded into VRAM to be 4GB (Conservative).
    """
    if not torch.cuda.is_available():
        return 1

    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_gb = total_memory - reserved_vram_gb
        actual_limit = int((available_gb * 1024) / mb_per_image) if available_gb > 0 else 1

        if target_batch_size <= actual_limit:
            return target_batch_size

        powers_of_2 = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        for p in powers_of_2:
            if p <= actual_limit:
                logging.warning(f"Target batch_size {target_batch_size} is unsafe for this GPU. " f"Adjusting to: {p}")
                return p
        return 1
    except Exception:
        return min(target_batch_size, 8)


@run_inference_entrypoint
def pyfeat_inference(config: dict) -> None:
    """
    Orchestrate the full Py-FEAT detection pipeline using Batched Inference.

    1) Setup logging and validate hardware (VRAM) for batch size optimization.
    2) Initialize `ImagePathsByCameraLoader` to yield all frame paths per camera.
    3) Pre-allocate standardized 4D NumPy arrays (Subject, Camera, Frame, Data).
    4) For each camera view:
        a. Retrieve all image paths at once.
        b. Pass full list to `detector.detect_image` for internal batching.
        c. Validate results (raise RuntimeError if empty).
        d. Sort detections spatially (X-coord) and map to Subjects.
        e. Populate the 4D arrays.
    5) Save compressed results to `<result_folders>/.npz`.

    Args:
        config (dict): Configuration dictionary.
    """

    logging.info("Starting Py-Feat Inference Pipeline (Camera-Level Processing).")

    camera_names = config["camera_names"]
    subjects_descr = config["subjects_descr"]
    cam_sees_subjects = config["cam_sees_subjects"]

    # Safety Check for Batch Size
    config_batch = int(config.get("batch_size", 1))
    inference_batch_size = get_safe_batch_size(config_batch)
    logging.info(f"Initialized with internal batch size: {inference_batch_size}")

    # Initialize Dataloader (Yields all paths for a camera at once)
    dataloader = ImagePathsByCameraLoader(config, camera_names)

    # Calculate dimensions based on the dataloader's range logic
    n_frames = len(dataloader)
    n_cams = len(camera_names)
    n_subjects = len(subjects_descr)

    detector = Detector(
        face_model="retinaface",
        au_model="xgb",
        emotion_model="resmasknet",
        facepose_model="img2pose",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Initialize result arrays
    emotion_vectors = np.zeros((n_subjects, n_cams, n_frames, NUM_EMOTIONS))
    bbox_vectors = np.zeros((n_subjects, n_cams, n_frames, 4))
    au_vectors = np.zeros((n_subjects, n_cams, n_frames, NUM_AUS))
    pose_vectors = np.zeros((n_subjects, n_cams, n_frames, NUM_POSES))

    frame_indices = [None] * n_frames

    # Process Camera by Camera
    for cam_idx, (camera_name, all_frame_paths) in enumerate(dataloader):
        logging.info(f"Processing Camera: {camera_name} ({len(all_frame_paths)} frames)")

        # Create a lookup map: FilePath -> (Array Index 0..N, Real Frame Index)
        # We need this because Py-Feat returns a DataFrame that might not be
        # perfectly ordered
        frame_range_indices = range(dataloader.start, dataloader.end, dataloader.step)
        path_map = {path: (i, real_idx) for i, (path, real_idx) in enumerate(zip(all_frame_paths, frame_range_indices))}

        try:
            # Pass ALL paths to detector. Py-Feat handles the batching internally.
            results = detector.detect_image(all_frame_paths, batch_size=inference_batch_size)
        except Exception as e:
            logging.critical(f"Inference failed for camera {camera_name}: {e}")
            raise e

        # Crash if results are empty
        if results is None or results.empty:
            error_msg = f"ERROR: No faces detected in ANY frame for camera {camera_name}."
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        # Process results: Group by 'input' (file path) to handle each frame
        for img_path, frame_data in results.groupby("input"):
            # Retrieve indices from our lookup map
            if img_path not in path_map:
                continue

            array_idx, real_idx = path_map[img_path]

            # Populate global frame index list once per frame
            if frame_indices[array_idx] is None:
                frame_indices[array_idx] = f"{real_idx:09d}"

            # We sort faces by X-coordinate to ensure consistent Subject ID assignment
            # (Left -> Right) across the video sequence without using a tracker.
            frame_data = frame_data.sort_values(by="FaceRectX")

            subj_map = cam_sees_subjects.get(camera_name, [])

            for face_idx, (_, row) in enumerate(frame_data.iterrows()):
                subj_idx = subj_map[face_idx] if face_idx < len(subj_map) else None
                if subj_idx is not None and subj_idx < n_subjects:
                    bbox_vectors[subj_idx, cam_idx, array_idx, :] = [
                        row["FaceRectX"],
                        row["FaceRectY"],
                        row["FaceRectWidth"],
                        row["FaceRectHeight"],
                    ]
                    emotion_vectors[subj_idx, cam_idx, array_idx, :] = row.emotions.values
                    au_vectors[subj_idx, cam_idx, array_idx, :] = row.aus.values
                    pose_vectors[subj_idx, cam_idx, array_idx, :] = row.poses.values

        # Clear GPU memory between cameras to prevent fragmentation
        torch.cuda.empty_cache()

    # Save Results
    out = {
        "faceboxes": bbox_vectors,
        "emotions": emotion_vectors,
        "aus": au_vectors,
        "poses": pose_vectors,
        "data_description": {
            "faceboxes": {
                "axis0": subjects_descr,
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": ["x", "y", "w", "h"],
            },
            "emotions": {
                "axis0": subjects_descr,
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": [
                    "anger",
                    "disgust",
                    "fear",
                    "happiness",
                    "sadness",
                    "surprise",
                    "neutral",
                ],
            },
            "aus": {
                "axis0": subjects_descr,
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": [
                    "AU01",
                    "AU02",
                    "AU04",
                    "AU05",
                    "AU06",
                    "AU07",
                    "AU09",
                    "AU10",
                    "AU11",
                    "AU12",
                    "AU14",
                    "AU15",
                    "AU17",
                    "AU20",
                    "AU23",
                    "AU24",
                    "AU25",
                    "AU26",
                    "AU28",
                    "AU43",
                ],
            },
            "poses": {
                "axis0": subjects_descr,
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": ["Pitch", "Roll", "Yaw"],
            },
        },
    }

    result_path = os.path.join(config["result_folders"]["emotion_individual"], f"{config['algorithm']}.npz")
    np.savez_compressed(result_path, **out)
    logging.info("Py-Feat processing complete.")
