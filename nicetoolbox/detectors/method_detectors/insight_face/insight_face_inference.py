# ! DON'T AUTOSORT - torch need to be imported first to enable onnx-runtime GPU
import torch  # noqa: F401, I001
import onnxruntime
import insightface  # noqa: F401
from insightface.app import FaceAnalysis

import logging
import os
import pickle

import cv2

from nicetoolbox_core.entrypoint import run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

RAW_INFERENCE_PICKLE_NAME = "insight_face_inference_raw.pkl"

# TODO: this detector runs detection only. Add support for landmarks (2d/3d),
# gender/age (probably don't need this) and face embedding (w600k_r50) models.
ALLOWED_MODULES = ["detection"]


def _visualize_frame(app, out_folder, camera_name, image, faces, real_frame_idx):
    out_dir = os.path.join(out_folder, "images", camera_name)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{real_frame_idx:09d}.jpg"), app.draw_on(image, faces))


def _cuda_providers():
    available = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider is not available to onnxruntime. Available providers: {available}.")
    # HEURISTIC instead of the default EXHAUSTIVE: we change resolution each time and kill GPU by it
    return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]


@run_inference_entrypoint
def insight_face_inference(config: dict) -> None:
    logging.info("RUNNING face detection 'InsightFace' (raw native save)!")

    # (1) Access config parameters
    out_folder = config["out_folders"]["face_bounding_box"]
    visualize_native = config["visualize_native"]

    # (2) Prepare data loader
    camera_names = config["camera_names"]
    dataloader = ImagePathsByFrameIndexLoader(config=config, expected_cameras=camera_names)

    # (3) Initialize the detector.
    # insightface resolves weights as <root>/models/<name>, and skips the download when present.
    providers = _cuda_providers()
    model_pack = config["model_pack"]
    model_root = config["model_root"]
    app = FaceAnalysis(name=model_pack, root=model_root, allowed_modules=ALLOWED_MODULES, providers=providers)

    # Set insigh face hyperparams
    det_thresh = float(config["det_thresh"])
    det_size = [tuple(pair) for pair in config["det_size"]]
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

    bound_providers = app.models["detection"].session.get_providers()
    logging.info(f"InsightFace detection session providers: {bound_providers}, det_size: {det_size}")
    if "CUDAExecutionProvider" not in bound_providers:
        raise RuntimeError(f"InsightFace detection session is not running on CUDA. Providers: {bound_providers}.")

    # (4) Run InsighFace for each frame
    per_frame_outputs = []
    for real_frame_idx, frame_paths_per_camera in dataloader:
        frame_bundle = {}
        for camera_name, frame_path in frame_paths_per_camera.items():
            image = cv2.imread(frame_path)
            faces = app.get(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            detections = []
            for face in faces:
                detections.append(
                    {
                        "bbox": face.bbox.tolist(),
                        "det_score": float(face.det_score),
                        "keypoints_2d": face.kps.tolist(),
                    }
                )
            frame_bundle[camera_name] = detections

            # Optional native visualization
            if visualize_native:
                _visualize_frame(app, out_folder, camera_name, image, faces, real_frame_idx)

        per_frame_outputs.append(frame_bundle)

    # (5) Save results to the raw pickle
    save_file_name = os.path.join(out_folder, RAW_INFERENCE_PICKLE_NAME)
    with open(save_file_name, "wb") as raw_file:
        pickle.dump(per_frame_outputs, raw_file)
    logging.info(f"Face detection 'InsightFace' raw pack COMPLETED! Wrote '{save_file_name}'.\n")
