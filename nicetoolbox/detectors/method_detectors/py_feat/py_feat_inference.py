"""
Run Py-Feat (Detectorv2) on the data, camera by camera.

Detectorv2 predicts action units, emotions, valence/arousal, gaze, a 478-point
face mesh, head pose and blendshapes in a single forward pass per face crop.
All cameras are concatenated into one Fex table and saved as parquet.
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from feat.multitask import MESH_COLUMNS_V2, VA_COLUMNS_V2
from huggingface_hub import hf_hub_download

from nicetoolbox_core.entrypoint import run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByCameraLoader

RAW_INFERENCE_PARQUET_NAME = "py_feat_inference_raw.parquet"
# Fex column groupings are python attributes, so parquet drops them.
# They are embedded in the file's schema metadata under this key instead.
FEATURE_COLUMNS_METADATA_KEY = b"feature_columns"


def create_pyfeat_detector(assets_dir: str):
    # ! pyfeat doesn't have any proper way to overload HF_HOME for downloaded models
    # we are doing this ugly monkey patch to redirect face detector model to the assets
    import feat.utils
    from feat.multitask.inference import HF_REPO, HF_WEIGHTS_FILE

    # ! fallback_filename and cache_dir are part of py-feat's call signature but unused here:
    # the current weights file exists, and cache_dir is what we are overriding.
    def patched(repo_id, filename, fallback_filename, cache_dir):  # noqa: ARG001
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=assets_dir,
            local_files_only=True,
        )

    feat.utils.hf_hub_download_with_fallback = patched

    from feat import Detectorv2  # imported after the patch so it picks it up

    multitask_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_WEIGHTS_FILE,
        cache_dir=assets_dir,
        local_files_only=True,
    )
    return Detectorv2(
        device="cuda",
        multitask_weights=multitask_path,
        identity_model=None,  # TODO: add support for identity extraction
    )


def save_native_visualization(fex, out_folder: str, camera_name: str, barplots: bool) -> None:
    # Disable matplotlib interactive visualization
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_folder) / "images" / camera_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate frame by frame, rather than standard pyfeat all frames together.
    fex_by_frames = fex.groupby("frame")
    for pyfeat_idx, single_frame in fex_by_frames:
        # ! Gaze direction visualization is broken
        # ! Pose rotation is also broken? Inversed roll?
        # TODO: support other visualization modes? (see https://py-feat.org/basic_tutorials/Plotting)
        single_frame_figure = single_frame.plot_detections(
            faces="landmarks",
            faceboxes=True,
            poses=False,
            gazes=False,
            au_barplot=barplots,
            emotion_barplot=barplots,
        )

        single_frame_figure = single_frame_figure[0]
        single_frame_figure.savefig(out_dir / f"{int(pyfeat_idx):09d}.jpg", dpi=100, bbox_inches="tight")
        plt.close(single_frame_figure)

    logging.info(f"Native visualization: wrote figures for camera {camera_name} to {out_dir}.")


@run_inference_entrypoint
def py_feat_inference(config: dict) -> None:
    """
    Orchestrate the Py-Feat detection pipeline.

    1) Build Detectorv2 against the pre-downloaded asset weights.
    2) Iterate camera by camera, passing all frame paths of a view at once.
    3) Save each camera's Fex table as a builtins-only pickle.

    Args:
        config (dict): Configuration dictionary.
    """
    logging.info("Starting Py-Feat Inference Pipeline.")

    camera_names = config["camera_names"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    face_detection_threshold = config["face_detection_threshold"]
    save_face_mesh = config["save_face_mesh"]
    visualize_native = config["visualize_native"]
    visualize_native_barplots = config["visualize_native_barplots"]
    assets_dir = config["hf_weights_cache_dir"]
    out_folder = config["out_folders"]["emotion_individual"]
    # Optional (width, height) to resize frames before detection. None = native resolution.
    output_size = config["output_size"] or None

    dataloader = ImagePathsByCameraLoader(config, camera_names)

    # (1) Build the detector, point HF cache to assets dir
    detector = create_pyfeat_detector(assets_dir)

    per_camera, columns = [], {}
    for camera_name, frame_paths in dataloader:
        logging.info(f"Processing camera {camera_name} ({len(frame_paths)} frames).")

        fex = detector.detect(
            frame_paths,
            output_size=output_size,
            batch_size=batch_size,
            num_workers=num_workers,
            face_detection_threshold=face_detection_threshold,
        )

        # Which columns form each feature block. Identical for every camera, and the
        # only place these groupings exist - Fex keeps them as python attributes.
        columns = {
            "faceboxes": fex.facebox_columns,  # [X, Y, W, H, Confidence]
            "valence_arousal": list(VA_COLUMNS_V2),  # Valence / Arousal scores [-1, 1]
            "emotions": fex.emotion_columns,  # 7 Emotions scores
            "aus": fex.au_columns,  # Subset of FACS activation units
            "poses": fex.facepose_columns,  # rot + pos of the head
            "gazes": fex.gaze_columns,  # Gaze direction vector + angle
            "landmarks": fex.landmark_columns,  # 2d iBug 68 points
            "blendshapes": fex.blendshape_columns,
            # TODO: read and save identities
        }
        if save_face_mesh:
            columns["face_mesh"] = list(MESH_COLUMNS_V2)  # 478 mediapipe points, x/y/z blocks
        columns = {name: [str(c) for c in cols] for name, cols in columns.items()}

        # extend pandas dataframe to store also camera name
        camera_table = pd.DataFrame(fex).assign(camera=camera_name)
        if not save_face_mesh:
            camera_table = camera_table.drop(columns=list(MESH_COLUMNS_V2))
        per_camera.append(camera_table)

        if visualize_native:
            save_native_visualization(fex, out_folder, camera_name, visualize_native_barplots)

        torch.cuda.empty_cache()

    # convert pandas table to parquet
    table = pa.Table.from_pandas(pd.concat(per_camera, ignore_index=True))
    # as a small bonus we put a json with column-feature relationship
    feature_columns = json.dumps(columns).encode()
    metadata = {**(table.schema.metadata or {}), FEATURE_COLUMNS_METADATA_KEY: feature_columns}

    result_path = os.path.join(out_folder, RAW_INFERENCE_PARQUET_NAME)
    pq.write_table(table.replace_schema_metadata(metadata), result_path)

    logging.info(f"Py-Feat processing complete. Results saved to {result_path}.")
