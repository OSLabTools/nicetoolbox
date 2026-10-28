import logging
import os
import pickle
import shutil
from typing import Dict, List

from mmdet.apis import DetInferencer

from nicetoolbox_core.entrypoint import SubprocessResult, run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByCameraLoader

RAW_INFERENCE_PICKLE_NAME = "mmdet_inference_raw.pkl"

# TODO: not hardcode it, drive through configs
PERSON_LABEL = 0  # COCO class 0 == person


@run_inference_entrypoint
def main(config: dict) -> None:
    logging.info("RUNNING person detection 'mmdet' (raw native save)!")

    out_folder = config["algorithm_out_folder"]
    native_viz_folder = config["native_viz_folder"]
    camera_names = config["camera_names"]
    required_assets = config["required_assets"]
    batch_size = config["batch_size"]
    save_raw_json_predictions = config["save_raw_json_predictions"]
    visualize_native = config["visualize_native"]

    dataloader = ImagePathsByCameraLoader(config=config, expected_cameras=camera_names)

    # there is two ways to get config file: as alias from local installation
    # or as file from assets system. At least one need to be provided.
    model_ref = required_assets.get("det_config") or config["detection_model_config"]
    if model_ref is None:
        raise ValueError(
            "mmdet: need either detection_model_config (mmdet metafile alias) or "
            "required_assets.det_config (path to a .py config file)."
        )

    inferencer = DetInferencer(
        model=model_ref,
        weights=required_assets["det_checkpoint"],
        device="cuda:0",
        show_progress=False,
    )

    # for each camera, send batches to the frame
    per_camera_outputs: Dict[str, List[list]] = {}
    for camera_name, frame_paths in dataloader.with_progress(desc="MMDet"):
        camera_out_dir = os.path.join(out_folder, camera_name)

        # run batched inference
        result: Dict[str, list] = inferencer(
            frame_paths,
            batch_size=batch_size,
            return_datasamples=False,
            no_save_vis=not visualize_native,
            no_save_pred=not save_raw_json_predictions,
            out_dir=camera_out_dir,
            # ! default mmdet thr, applied only for visualization optimization
            # pkl predictoons are kept as is and not discarded
            pred_score_thr=0.3,
        )

        # to match toolbox convention, we need to move hardcoded mmdet vis folder
        # to standard native visualization folder (a bit ugly, but should be fine)
        if visualize_native:
            mmdet_vis_dir = os.path.join(camera_out_dir, "vis")
            target_vis_dir = os.path.join(native_viz_folder, camera_name)
            if os.path.isdir(target_vis_dir):
                shutil.rmtree(target_vis_dir)
            shutil.move(mmdet_vis_dir, target_vis_dir)

        # collect per-frame detections for this camera
        camera_frames = []
        for prediction in result["predictions"]:
            detections = []
            for bbox, label, score in zip(prediction["bboxes"], prediction["labels"], prediction["scores"]):
                if int(label) != PERSON_LABEL:
                    continue
                detections.append({"bbox": [float(v) for v in bbox], "score": float(score)})
            camera_frames.append(detections)
        per_camera_outputs[camera_name] = camera_frames

    # save to raw pickle
    save_file_name = os.path.join(out_folder, RAW_INFERENCE_PICKLE_NAME)
    with open(save_file_name, "wb") as raw_file:
        pickle.dump(per_camera_outputs, raw_file)
    logging.info(f"Person detection 'mmdet' raw pack COMPLETED! Wrote '{save_file_name}'.\n")

    return SubprocessResult.from_artifacts(raw_output=save_file_name)
