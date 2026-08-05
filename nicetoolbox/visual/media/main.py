"""
Main module for initializing and running the visualizer.
"""

import argparse
import os
from pathlib import Path

import cv2

from ...utils import visual_utils as vis_utils
from ...utils.system import check_long_path_support
from .. import config_handler as vis_cfg
from ..in_out import IO
from .components import (
    BodyJointsComponent,
    EmotionIndividualComponent,
    EyeClosedStateComponent,
    EyeClosureComponent,
    FaceLandmarksComponent,
    GazeFusionComponent,
    GazeInteractionComponent,
    HandJointsComponent,
    HeadOrientationComponent,
    KinematicsComponent,
    ProximityComponent,
)
from .viewer import Viewer


def main(project_folder_path: Path, machine_specifics_file: Path, visualizer_config_file: Path):
    """
    Main function to run the visualizer.

    This function sets up the configuration, initializes the input/output handlers,
    loads calibration data, and initializes the viewer for visualizing the components.

    Args:
        project_folder_path (Path): Path to the project folder containing nice_project.toml.
        machine_specifics_file (Path): Path to machine_specific_paths.toml.
        visualizer_config_file (Path): Path to visualizer_config.toml, may contain placeholders.
    """

    # SYSTEM CHECK
    check_long_path_support()

    # CONFIGURATION - IO
    config_handler = vis_cfg.Configuration(project_folder_path, machine_specifics_file, visualizer_config_file)
    visualizer_config = config_handler.get_updated_visualizer_config()

    # IO
    io = IO(visualizer_config)
    nice_tool_input_folder = io.get_component_nice_tool_input_folder(
        visualizer_config["video"], visualizer_config["io"]["dataset_name"]
    )

    # load calibration for the video
    calibration_file = io.get_calibration_file(visualizer_config["video"])
    calib = None
    if calibration_file:
        calib = vis_utils.load_calibration(
            calibration_file,
            visualizer_config["video"],
            camera_names=config_handler.get_camera_names(),
        )
    if not calib:
        print(
            "WARNING: User did not provide a valid calibration file. "
            "Visualization of 3d pose estimation, and gaze results "
            "requires calibration data."
        )

    # INITIALIZE VIEWER
    viewer = Viewer(visualizer_config)

    # CHECK CONFIGURATION
    all_cameras = config_handler.get_camera_names()
    config_handler.check_config()

    # LOAD COMPONENTS DATA
    components_list = visualizer_config["media"]["visualize"]["components"]
    components = []
    for component in components_list:
        if component not in os.listdir(io.get_experiment_video_folder()):
            print(
                f"WARNING: {component} Component is not found in video output. "
                f"It will not be visualized.\n To avoid this warning, consider "
                f"removing '{component}' from the components list in the "
                "visualizer_config.toml file"
            )
            continue
        components.append(component)

    if "body_joints" in components:
        body_joints_component = BodyJointsComponent(visualizer_config, io, viewer, "body_joints")
        eyes_middle_2d_data, eyes_middle_3d_data = body_joints_component.calculate_middle_eyes()
    else:
        body_joints_component = None
        eyes_middle_2d_data = None
        eyes_middle_3d_data = None

    hand_joints_component = (
        HandJointsComponent(visualizer_config, io, viewer, "hand_joints") if "hand_joints" in components else None
    )
    face_landmarks_component = (
        FaceLandmarksComponent(visualizer_config, io, viewer, "face_landmarks")
        if "face_landmarks" in components
        else None
    )
    gaze_interaction_component = (
        GazeInteractionComponent(visualizer_config, io, viewer, "gaze_interaction")
        if "gaze_interaction" in components
        else None
    )

    look_at_data_tuples = (
        gaze_interaction_component.get_lookat_data() if "gaze_interaction" in components else None
    )  # returns list of (data, data_labels), one per gaze_distance instance

    gaze_fusion_component = (
        GazeFusionComponent(
            visualizer_config,
            io,
            viewer,
            "gaze_multiview",
            calib,
            eyes_middle_3d_data,
            look_at_data_tuples,
        )
        if "gaze_multiview" in components
        else None
    )

    emotion_ind_component = (
        EmotionIndividualComponent(visualizer_config, io, viewer, "emotion_individual")
        if "emotion_individual" in components
        else None
    )

    head_orientation_component = (
        HeadOrientationComponent(visualizer_config, io, viewer, "head_orientation")
        if "head_orientation" in components
        else None
    )

    proximity_component = (
        ProximityComponent(
            visualizer_config,
            io,
            viewer,
            "proximity",
            eyes_middle_3d_data,
            eyes_middle_2d_data,
        )
        if "proximity" in components
        else None
    )

    kinematics_component = (
        KinematicsComponent(visualizer_config, io, viewer, "kinematics") if "kinematics" in components else None
    )

    eye_closed_state_component = (
        EyeClosedStateComponent(visualizer_config, io, viewer, "eye_closed_state")
        if "eye_closed_state" in components
        else None
    )

    # returns list of (closed_state, eye_labels), one per eye_closure_threshold instance
    closed_state_tuples = (
        eye_closed_state_component.get_closed_state_data() if eye_closed_state_component is not None else None
    )
    state_camera_names = eye_closed_state_component.camera_names if eye_closed_state_component is not None else None

    eye_closure_component = (
        EyeClosureComponent(
            visualizer_config,
            io,
            viewer,
            "eye_closure_score",
            closed_state_tuples,
            state_camera_names,
        )
        if "eye_closure_score" in components
        else None
    )

    instances = [
        body_joints_component,
        hand_joints_component,
        face_landmarks_component,
        gaze_fusion_component,
        emotion_ind_component,
        proximity_component,
        kinematics_component,
        head_orientation_component,
        eye_closure_component,
        eye_closed_state_component,
    ]

    # VISUALIZATION
    # initialize rerun visualizer
    viewer.spawn()
    for camera in all_cameras:
        # to get image width and height
        example_image_path = os.path.join(nice_tool_input_folder, camera, "frames", f"{1:09}.png").replace("\\", "/")
        example_image = cv2.cvtColor(cv2.imread(example_image_path), cv2.COLOR_BGR2RGB)
        h, w = example_image.shape[:2]  # ← grab size from actual frame
        image_size = (w, h)
        entity_path_cams = viewer.get_camera_pos_entity_path(camera)
        camera_calib = calib[camera] if calib else None
        viewer.log_camera(camera_calib, entity_path_cams, image_size)
    frame_idx = viewer.get_start_frame()
    end_frame = viewer.get_end_frame()
    while True:
        if end_frame != -1 and frame_idx > end_frame:
            break
        viewer.go_to_timestamp(frame_idx)
        frame_no = viewer.get_video_start() + frame_idx
        image_name = f"{frame_no:09}.png"
        for camera in all_cameras:
            # log camera into 3d canvas
            image_path = os.path.join(nice_tool_input_folder, camera, "frames", image_name).replace("\\", "/")
            # TODO: we will stop if we will just stop when there is no frame exist
            # is it a good idea? probably not, but we can fix it latter
            if not os.path.exists(image_path):
                return
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            entity_path_imgs = viewer.get_images_entity_path(camera)
            viewer.log_image(image, entity_path_imgs, img_quality=75)

        for instance in instances:
            if instance is not None:
                instance.visualize(frame_idx)

        frame_idx += viewer.get_step()


def entry_point():
    """Entry point for running NICE toolbox rerun visualizations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_folder_path",
        default=Path("."),
        type=Path,
        required=False,
        help="Path to the NICE Toolbox project folder containing nice_project.toml config",
    )
    parser.add_argument(
        "--machine_specifics",
        default=Path("machine_specific_paths.toml"),
        type=Path,
        required=False,
        help="Path to machine_specific_paths.toml config",
    )
    parser.add_argument(
        "--visual_config",
        default=Path("<configs_folder_path>/visualizer_config.toml"),
        type=Path,
        required=False,
        help="Path to visualizer_config.toml, supports placeholders",
    )
    args = parser.parse_args()

    main(args.project_folder_path, args.machine_specifics, args.visual_config)


if __name__ == "__main__":
    entry_point()
