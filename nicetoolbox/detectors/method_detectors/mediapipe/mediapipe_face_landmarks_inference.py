import logging
import os
import pickle

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from nicetoolbox_core.entrypoint import run_inference_entrypoint
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

RAW_INFERENCE_PICKLE_NAME = "mediapipe_face_landmarks_inference_raw.pkl"


# TODO: legacy, might be deprecated with newer mediapipe version
def _visualize_frame(image_bgr, detection_result, native_viz_folder, camera_name, real_frame_idx):
    annotated = image_bgr.copy()
    for face_landmarks in detection_result.face_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_landmarks])
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    out_dir = os.path.join(native_viz_folder, camera_name)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{real_frame_idx:09d}.jpg"), annotated)


@run_inference_entrypoint
def mediapipe_face_landmarks_inference(config: dict) -> None:
    logging.info("RUNNING face landmarker 'MediaPipe' (raw native save)!")

    # (1) Access config parameters
    out_folder = config["algorithm_out_folder"]
    native_viz_folder = config["native_viz_folder"]
    visualize_native = config["visualize_native"]
    model_path = config["required_assets"]["model_path"]

    # (2) Prepare data loader
    camera_names = config["camera_names"]
    dataloader = ImagePathsByFrameIndexLoader(config=config, expected_cameras=camera_names)

    # (3) Determine maximum number of faces to detect, we need some kind of number for that because default is 1
    num_faces = config["num_faces"]

    # (4) Initialize MediaPipe Face Landmarker detectors (one per camera stream)
    min_face_detection_confidence = config["min_face_detection_confidence"]
    min_face_presence_confidence = config["min_face_presence_confidence"]
    min_face_tracking_confidence = config["min_face_tracking_confidence"]

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=num_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_face_tracking_confidence,
    )
    detectors = {cam: vision.FaceLandmarker.create_from_options(options) for cam in camera_names}

    # Fps is needed to compute the timestamp for each frame, which is required by the MediaPipe API
    fps = config["input_recipes"]["video_input_recipe"]["fps"]

    # (5) Process frames
    per_frame_outputs = []
    for frame_idx, frame_paths_per_camera in dataloader:
        timestamp_ms = int(frame_idx * 1000.0 / fps)
        frame_bundle = {}
        for camera_name, frame_path in frame_paths_per_camera.items():
            image = cv2.imread(frame_path)
            height, width = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            detection_result = detectors[camera_name].detect_for_video(mp_image, timestamp_ms)

            detections = []
            if detection_result.face_landmarks:
                for idx, landmarks in enumerate(detection_result.face_landmarks):
                    keypoints = [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks]

                    blendshapes = {}
                    if detection_result.face_blendshapes:
                        blendshapes = {c.category_name: c.score for c in detection_result.face_blendshapes[idx]}
                    transformation_matrix = detection_result.facial_transformation_matrixes[idx]

                    detections.append(
                        {
                            "keypoints": keypoints,
                            "blendshapes": blendshapes,
                            "transformation_matrix": transformation_matrix,
                        }
                    )
            frame_bundle[camera_name] = detections

            if visualize_native:
                _visualize_frame(image, detection_result, native_viz_folder, camera_name, frame_idx)

        per_frame_outputs.append(frame_bundle)

    # (6) Save results to the raw pickle
    save_file_name = os.path.join(out_folder, RAW_INFERENCE_PICKLE_NAME)
    with open(save_file_name, "wb") as raw_file:
        pickle.dump(per_frame_outputs, raw_file)
    logging.info(f"Face landmarker 'MediaPipe' raw pack COMPLETED! Wrote '{save_file_name}'.\n")
