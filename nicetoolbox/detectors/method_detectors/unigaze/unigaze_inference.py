"""
Run UniGaze inference on the provided video frames.
"""

import logging
import os
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
import torchvision.transforms as transforms
from unigaze.loader import MODEL_INDEX, build_unigaze_model

from nicetoolbox_core.entrypoint import run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

# The 6 selected landmarks from the generic 3D face model
FACE_MODEL_6 = np.array(
    [
        [-46.5094986, -38.32709503, 36.41600418],
        [-17.76072121, -32.58519745, 29.07615662],
        [18.04391098, -30.95682335, 29.0629673],
        [44.73758698, -34.10787201, 36.73243713],
        [-10.61166477, 12.95834923, 21.62276459],
        [11.28962994, 13.86424446, 21.83790016],
    ]
)
###
# Functions for cropping, undistorting and normalizing face for gaze inference
###


def pitchyaw_to_vector(pitchyaws):
    """Convert given yaw and pitch angles to unit gaze vectors."""
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((pitchyaws.shape[0], 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to pitch and yaw angles."""
    n = vectors.shape[0]
    vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(n, 1)
    out = np.empty((n, 2))
    out[:, 0] = np.arcsin(vectors[:, 1])  # pitch
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # yaw
    return out


def get_face_center_by_nose(hR, ht, face_model_6):
    """Computes the face center location based on the 3D landmarks."""
    Fc = np.dot(hR, face_model_6.T) + ht  # (3, 6)
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape(3, 1)
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape(3, 1)
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape(3, 1)
    return face_center


def estimateHeadPose(landmarks, face_model_6, camera, distortion, iterate=True):
    """Solves the Perspective-n-Point (PnP) problem to estimate head pose."""
    _, rvec, tvec = cv2.solvePnP(face_model_6, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    if iterate:
        _, rvec, tvec = cv2.solvePnP(face_model_6, landmarks, camera, distortion, rvec, tvec, True)
    return rvec, tvec


def normalize_face_image(img, focal_norm, distance_norm, roi_size, center, hr, cam):
    """Normalizes the face image and landmarks using projective transformation."""
    center = center.reshape(3, 1)
    hR = cv2.Rodrigues(hr)[0]

    distance = np.linalg.norm(center)
    z_scale = distance_norm / distance

    cam_norm = np.array(
        [
            [focal_norm, 0, roi_size[0] / 2],
            [0, focal_norm, roi_size[1] / 2],
            [0, 0, 1.0],
        ]
    )

    S = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ]
    )

    hRx = hR[:, 0]
    forward = (center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))
    img_warped = cv2.warpPerspective(img, W, roi_size)

    hR_norm = np.dot(R, hR)
    return img_warped, R, hR_norm


def denormalize_predicted_gaze(gaze_yaw_pitch, R_inv):
    """Transforms normalized gaze predictions back to camera coordinate system."""
    pred_gaze_cancel_nor = pitchyaw_to_vector(gaze_yaw_pitch.reshape(1, 2)).reshape(3, 1)
    pred_gaze_cancel_nor = np.matmul(R_inv, pred_gaze_cancel_nor.reshape(3, 1))
    pred_gaze_cancel_nor = pred_gaze_cancel_nor / np.linalg.norm(pred_gaze_cancel_nor)
    pred_yaw_pitch_cancel_nor = vector_to_pitchyaw(pred_gaze_cancel_nor.reshape(1, 3))
    return pred_gaze_cancel_nor, pred_yaw_pitch_cancel_nor


def get_image_transform():
    """Returns PyTorch transform for ImageNet normalization."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_cam_para_studio(content, cam):
    """Extracts intrinsic and extrinsic camera parameters from calibration data."""
    cam_matrix = content[cam].get("intrinsic_matrix", None)
    cam_matrix = np.vstack(cam_matrix) if cam_matrix is not None else None
    cam_distortion = content[cam].get("distortions", None)
    cam_distortion = np.hstack(cam_distortion) if cam_distortion is not None else None
    cam_rotation = content[cam].get("rotation_matrix", None)
    cam_extrinsic = content[cam].get("extrinsics_matrix", None)

    if not isinstance(cam_rotation, np.ndarray) and cam_rotation is not None:
        cam_rotation = np.array(cam_rotation)
    return cam_matrix, cam_distortion, cam_rotation, cam_extrinsic


def _visualize_camera_frame(config, camera_name, image, gaze_rays_2d, real_frame_idx):
    """Draw one camera's gaze arrows on its frame (native visualization only).

    Mirrors UniGaze's own `predict_gaze_video.py`: each arrow runs from the 3D face center
    projected into the image to the projected ray tip, so the origin matches the point the
    gaze vector is anchored to during normalization (rather than a 2D landmark centroid).
    """
    image = image.copy()

    for gaze_ray_2d in gaze_rays_2d:
        if np.isnan(gaze_ray_2d).any():
            continue

        start_point = tuple(gaze_ray_2d[0].astype(int))
        end_point = tuple(gaze_ray_2d[1].astype(int))
        cv2.arrowedLine(image, start_point, end_point, (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.2)

    # Match the method-detector convention: detector_output/images/<camera_name>/.
    out_dir = os.path.join(config["out_folder"], "images", camera_name)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{real_frame_idx:09d}.jpg"), image)


###
# Main inference function for unigaze
###


@run_inference_entrypoint
def unigaze_inference(config):
    """
    Run UniGaze gaze detection on the provided video frames.
    """
    logging.info("RUNNING gaze detection 'UniGaze'!")

    # (1) Access config parameters
    camera_names = config["camera_names"]
    calibration = config["calibration"]
    # Patch TORCH_HOME OS env variable
    # face_alignment s3fd and 2DFAN4 models are used implicitly by torch
    # they should be already downloaded by the asset manager
    face_alignment_cache_dir = config["face_alignment_cache_dir"]
    os.environ["TORCH_HOME"] = face_alignment_cache_dir

    # Get model name from the file name
    unigaze_model_path = config["required_assets"]["unigaze_model"]
    model_name = Path(unigaze_model_path).stem
    if model_name not in MODEL_INDEX:
        raise ValueError(
            f"Unknown UniGaze model '{model_name}' derived from required asset "
            f"'{unigaze_model_path}'. Expected the filename to be one of: "
            f"{sorted(MODEL_INDEX)}."
        )

    # (2) Prepare data loader
    dataloader = ImagePathsByFrameIndexLoader(config=config, expected_cameras=camera_names)

    # (3) Initialize UniGaze and Face Alignment models
    device = torch.device("cuda")
    logging.info(f"Loading UniGaze model '{model_name}' on device '{device}'...")
    model = build_unigaze_model(model_name)
    model.load_unigaze_weights(unigaze_model_path)
    model.to(device)
    model.eval()

    logging.info("Initializing face alignment (2D landmark detector)...")
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=str(device))
    except Exception as e:
        logging.warning(f"FaceAlignment initialization failed: {e}")
        raise e

    image_torch_transform = get_image_transform()

    # (4) Prepare per-frame outputs list
    per_frame_outputs = []

    # settings for virtual camera to normalize face images for gaze estimation
    focal_norm = 960
    distance_norm = 600
    roi_size = (224, 224)
    facePts = FACE_MODEL_6.reshape(6, 1, 3)

    # (5) Inference loop
    for real_frame_idx, frame_paths_per_camera in dataloader:
        frame_bundle = {}

        for camera_name in camera_names:
            frame_file = frame_paths_per_camera[camera_name]
            image = cv2.imread(frame_file)

            # Run face alignment
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preds = fa.get_landmarks(image_rgb)

            if preds is None or len(preds) == 0:
                continue

            detections = []
            gaze_rays_2d = []  # projected (start, end) point pairs, used only for native visualization
            # Iterate over the detected faces
            for landmarks_in_original in preds:
                # Calculate bounding box for normalization
                x_min_val = landmarks_in_original[:, 0].min()
                x_max_val = landmarks_in_original[:, 0].max()
                y_min_val = landmarks_in_original[:, 1].min()
                y_max_val = landmarks_in_original[:, 1].max()

                bbox_center = (int((x_min_val + x_max_val) // 2), int((y_min_val + y_max_val) // 2))
                bbox_width = x_max_val - x_min_val
                bbox_height = y_max_val - y_min_val

                # Scale bounding box for face model and landmark cropping, leaving a margin around the face
                scale_factor = 2.0
                x_min = max(0, bbox_center[0] - int(bbox_width * scale_factor // 2))
                x_max = min(image.shape[1], bbox_center[0] + int(bbox_width * scale_factor // 2))
                y_min = max(0, bbox_center[1] - int(bbox_height * scale_factor // 2))
                y_max = min(image.shape[0], bbox_center[1] + int(bbox_height * scale_factor // 2))

                image_face = image[y_min:y_max, x_min:x_max]
                if image_face.size == 0:
                    continue

                landmarks_cropped = landmarks_in_original - np.array([x_min, y_min])
                landmarks_cropped_sub = landmarks_cropped[[36, 39, 42, 45, 31, 35], :]
                landmarks_cropped_sub = landmarks_cropped_sub.astype(float).reshape(6, 1, 2)

                # Get real calibrated camera parameters and shift principal point for face crop
                cam_matrix_orig, cam_distortion, _, _ = get_cam_para_studio(calibration, camera_name)
                cam_matrix = cam_matrix_orig.copy()
                cam_matrix[0, 2] -= x_min
                cam_matrix[1, 2] -= y_min
                cam_distor = cam_distortion if cam_distortion is not None else np.zeros((1, 5))

                hr, ht = estimateHeadPose(landmarks_cropped_sub, facePts, cam_matrix, cam_distor)
                hR = cv2.Rodrigues(hr)[0]
                face_center_camera_cord = get_face_center_by_nose(hR, ht, FACE_MODEL_6)

                img_normalized, R, hR_norm = normalize_face_image(
                    image_face,
                    focal_norm,
                    distance_norm,
                    roi_size,
                    face_center_camera_cord,
                    hr,
                    cam_matrix,
                )

                hr_norm = np.array([np.arcsin(hR_norm[1, 2]), np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
                if np.linalg.norm(hr_norm) > 80 * np.pi / 180:
                    continue

                # Prepare input tensor
                input_var = img_normalized[:, :, [2, 1, 0]]  # BGR to RGB
                input_var = image_torch_transform(input_var)
                input_var = input_var.float().to(device).unsqueeze(0)

                with torch.no_grad():
                    ret = model(input_var)

                pred_gaze = ret["pred_gaze"][0].cpu().numpy()

                # inverse is same as transpose for rotation matrices
                R_inv = R.T
                pred_gaze_camera, _ = denormalize_predicted_gaze(pred_gaze, R_inv)
                pred_gaze_camera = pred_gaze_camera.reshape(3)

                # Project the 3D gaze ray back to 2D, as UniGaze's own demo does
                vec_length = pred_gaze_camera.reshape(3, 1) * -112 * 1.5
                gaze_ray = np.concatenate(
                    (face_center_camera_cord.reshape(1, 3), (face_center_camera_cord + vec_length).reshape(1, 3)),
                    axis=0,
                )
                zero_vec = np.zeros((3, 1), dtype=float)
                projected, _ = cv2.projectPoints(gaze_ray, zero_vec, zero_vec, cam_matrix, cam_distor)
                projected = projected.reshape(2, 2) + np.array([x_min, y_min])

                detections.append(
                    {
                        "gaze_cam": pred_gaze_camera,
                        "landmarks_2d": landmarks_in_original,
                        "gaze_origin_2d": projected[0],
                    }
                )
                gaze_rays_2d.append(projected)

            frame_bundle[camera_name] = detections
            if config["visualize_native"]:
                _visualize_camera_frame(config, camera_name, image, gaze_rays_2d, real_frame_idx)

        per_frame_outputs.append(frame_bundle)

    # Save raw outputs as compressed .npz file
    out_dict = {
        "per_frame_outputs": np.asarray(per_frame_outputs, dtype=object),
    }

    save_file_name = os.path.join(config["out_folders"]["gaze_individual"], "unigaze_inference_raw.npz")
    np.savez_compressed(save_file_name, **out_dict)

    logging.info("Gaze detection 'UniGaze' COMPLETED!\n")
