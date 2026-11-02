import logging
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from nicetoolbox_core.data.loaded_array import load_array_from_path, select_array
from nicetoolbox_core.entrypoint import SubprocessResult, run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

# ! keep this order
# TODO: find a better way to inject non pip-installable modules?
# import sam3d modules from cloned repository
SAM3D_VENV_ROOT = Path(sys.prefix)
SAM3D_REPO_DIR = SAM3D_VENV_ROOT / "src" / "sam-3d-body"
sys.path.insert(0, str(SAM3D_REPO_DIR))
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf  # noqa: E402
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info  # noqa: E402
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer  # noqa: E402

RAW_INFERENCE_PICKLE_NAME = "sam3d_inference_raw.pkl"


def _intrinsics_tensor(calibration, camera_name, device):
    # load calibration, get inrinsics
    cam_calib = (calibration or {}).get(camera_name) or {}
    if "intrinsic_matrix" not in cam_calib:
        logging.warning(
            f"No intrinsics for camera '{camera_name}', SAM 3D Body uses its default focal length "
            "(image diagonal). Body placement (cam_t, depth) will be inaccurate."
        )
        return None
    K = np.asarray(cam_calib["intrinsic_matrix"], dtype=np.float32).reshape(3, 3)
    return torch.as_tensor(K, device=device).unsqueeze(0)


@run_inference_entrypoint
def main(config: dict) -> None:
    logging.info("RUNNING SAM 3D Body inference!")

    device = "cuda:0"
    out_folder = config["algorithm_out_folder"]
    camera_names = config["camera_names"]
    calibration = config.get("calibration")
    inference_type = config["inference_type"]
    save_vertices = config["save_vertices"]
    repo_id = config["hf_repo_id"]
    visualize_native = config["visualize_native"]
    native_viz_folder = config["native_viz_folder"]

    # force both HF and torch to work in offline mode
    cache_dir = config["cache_dir"]
    os.environ["TORCH_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"

    # load SAM3D model
    model, model_cfg = load_sam_3d_body_hf(repo_id)
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,  # TODO: support detectron2 as a separate detector?
        human_segmentor=None,  # TODO: support SAM2 segmentation masks?
        fov_estimator=None,  # TODO: support MoGe2 when intrinsics not available?
    )

    # native 2d skeleton visualizer
    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(mhr70_pose_info)

    # nice bboxes: (subjects, cameras, frames, 5) [x1, y1, x2, y2, conf], only the cameras we need
    bboxes = load_array_from_path(config["person_bbox_npz"], config["person_bbox_npz_key"])
    bboxes_all = select_array(bboxes, cameras=camera_names).data

    # try to load intrinsics from NICE calibration (if available)
    # if not, set them None; built once per camera (gpu tensors)
    cam_ints = {}
    for camera_name in camera_names:
        cam_ints[camera_name] = _intrinsics_tensor(calibration, camera_name, device)

    dataloader = ImagePathsByFrameIndexLoader(config=config, expected_cameras=camera_names)
    per_frame_outputs = []
    iterator = dataloader.with_progress(desc="SAM 3D Body")
    for frame_idx, (real_frame_idx, frame_paths_per_camera) in enumerate(iterator):  # for each frame
        persons_per_camera = {}  # store persons per camera
        for camera_name, frame_path in frame_paths_per_camera.items():  # for each camera
            cam_idx = camera_names.index(camera_name)
            cam_int = cam_ints[camera_name]

            # skip non-present subjects
            frame_bboxes = bboxes_all[:, cam_idx, frame_idx, :4]
            valid_mask = ~np.isnan(frame_bboxes).any(axis=1)
            if not valid_mask.any():
                logging.debug(f"No person boxes for frame {real_frame_idx}, camera {camera_name}")
                outputs = []
            else:
                valid_subject_indices = np.where(valid_mask)[0]

                # TODO: implement cross-frames batching?
                # run model inference
                outputs = estimator.process_one_image(
                    frame_path,
                    bboxes=frame_bboxes[valid_mask].astype(np.float32),
                    masks=None,  # TODO: support masks
                    cam_int=cam_int,
                    inference_type=inference_type,  # TODO: support hand mode? need hands bboxes
                )

            # native results
            persons = []
            for pi, output in enumerate(outputs):
                subject_idx = int(valid_subject_indices[pi])
                person = {
                    "subject_idx": subject_idx,
                    "keypoints_2d": np.asarray(output["pred_keypoints_2d"], np.float32),  # (70, 2) pixels
                    "keypoints_3d": np.asarray(output["pred_keypoints_3d"], np.float32),  # (70, 3) body-centred
                    "cam_t": np.asarray(output["pred_cam_t"], np.float32),  # (3,) body position in camera frame
                    "mhr_model_params": np.asarray(output["mhr_model_params"], np.float32),  # (204,) MHR pose + scales
                    "shape_params": np.asarray(output["shape_params"], np.float32),  # (45,) MHR identity
                }
                if save_vertices:  # (18439, 3) body-centred
                    person["vertices"] = np.asarray(output["pred_vertices"], np.float32)
                persons.append(person)

            persons_per_camera[camera_name] = persons

            # run native 2d visualization
            if visualize_native:
                image = cv2.imread(frame_path)
                for output in outputs:
                    # expects (K, 3) keypoints with a confidence column
                    fake_conf = np.ones((len(output["pred_keypoints_2d"]), 1))
                    kpts = np.concatenate([output["pred_keypoints_2d"], fake_conf], 1)
                    image = visualizer.draw_skeleton(image, kpts)
                out_dir = os.path.join(native_viz_folder, camera_name)
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, f"{real_frame_idx:09d}.jpg"), image)

            # TODO: native 3d visualization?
            # original repo uses opengl and windows manager, but they are hard to maintain
            # especially cross platform and headless systems

        per_frame_outputs.append(persons_per_camera)

    # frames: list over frames of {camera: persons}; a camera without persons is absent
    raw_output = {"frames": per_frame_outputs}
    if save_vertices:  # (36874, 3), shared by all meshes
        raw_output["faces"] = np.asarray(estimator.faces, dtype=np.int32)

    # save results to pkl
    save_file_name = os.path.join(out_folder, RAW_INFERENCE_PICKLE_NAME)
    with open(save_file_name, "wb") as raw_file:
        pickle.dump(raw_output, raw_file)
    logging.info(f"SAM 3D Body raw pack COMPLETED! Wrote '{save_file_name}'.\n")

    return SubprocessResult.from_artifacts(raw_output=save_file_name)
