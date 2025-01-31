"""
Main module for initializing and running the visualizer.
"""

import argparse
import os

import cv2

from ...utils import visual_utils as vis_utils
from .. import config_handler as vis_cfg
from ..in_out import IO
from .components import (
    BodyJointsComponent,
    FaceLandmarksComponent,
    GazeIndividualComponent,
    GazeInteractionComponent,
    HandJointsComponent,
    KinematicsComponent,
    ProximityComponent,
)
from .viewer import Viewer


def main(visualizer_config_file, machine_specifics_file):
    """
    Main function to run the visualizer.

    This function sets up the configuration, initializes the input/output handlers,
    loads calibration data, and initializes the viewer for visualizing the components.
    """

    # CONFIGURATION - IO
    config_handler = vis_cfg.Configuration(
        visualizer_config_file, machine_specifics_file
    )
    visualizer_config = config_handler.get_updated_visualizer_config()

    # IO
    io = IO(visualizer_config)
    nice_tool_input_folder = io.get_component_nice_tool_input_folder(
        visualizer_config["video"], visualizer_config["media"]["dataset_name"]
    )

    # load calibration for the video
    calibration_file = io.get_calibration_file(visualizer_config["video"])
    calib = vis_utils.load_calibration(
        calibration_file,
        visualizer_config["video"],
        camera_names=config_handler.get_camera_names(),
    )

    # INITIALIZE VIEWER
    viewer = Viewer(visualizer_config)

    # CHECK CONFIGURATION
    all_cameras = list(calib.keys())
    config_handler.check_config()
    for cam in all_cameras:
        config_handler.check_calibration(calib, cam)
    viewer.check_multiview()
    visualizer_config = config_handler.check_and_update_canvas()

    # LOAD COMPONENTS DATA
    components = visualizer_config["media"]["visualize"]["components"]
    for component in components:
        if component not in os.listdir(io.get_experiment_video_folder()):
            print(
                f"WARNING: {component} Component is not found in video output. "
                f"It will not be visualized.\n To avoid this warning, consider "
                f"removing '{component}' from the components list in the "
                "visualizer_config.toml file"
            )
            components.remove(component)  # noqa: B909

    if "body_joints" in components:
        body_joints_component = BodyJointsComponent(
            visualizer_config, io, viewer, "body_joints"
        )
        eyes_middle_2d_data = body_joints_component.calculate_middle_eyes(dimension=2)
        eyes_middle_3d_data = (
            body_joints_component.calculate_middle_eyes(dimension=3)
            if visualizer_config["media"]["multi_view"] is True
            else (None, None)
        )
    else:
        body_joints_component = None
        eyes_middle_2d_data = None, None
        eyes_middle_3d_data = None, None

    hand_joints_component = (
        HandJointsComponent(visualizer_config, io, viewer, "hand_joints")
        if "hand_joints" in components
        else None
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

    look_at_data_tuple = (
        gaze_interaction_component.get_lookat_data()
        if "gaze_interaction" in components
        else None
    )  # returns (data, data_labels)

    gaze_ind_component = (
        GazeIndividualComponent(
            visualizer_config,
            io,
            viewer,
            "gaze_individual",
            calib,
            eyes_middle_3d_data,
            look_at_data_tuple,
        )
        if "gaze_individual" in components
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
        KinematicsComponent(visualizer_config, io, viewer, "kinematics")
        if "kinematics" in components
        else None
    )

    instances = [
        body_joints_component,
        hand_joints_component,
        face_landmarks_component,
        gaze_ind_component,
        proximity_component,
        kinematics_component,
    ]

    # VISUALIZATION
    # initialize rerun visualizer
    viewer.spawn()
    for camera in all_cameras:
        if viewer.get_is_camera_position():
            entity_path_cams = viewer.get_camera_pos_entity_path(camera)
            viewer.log_camera(calib[camera], entity_path_cams)
    for frame_idx in range(
        viewer.get_start_frame(), viewer.get_end_frame(), viewer.get_step()
    ):
        viewer.go_to_timestamp(frame_idx)
        frame_no = (
            viewer.get_video_start()
            + frame_idx
            + config_handler.get_dataset_starting_index()
        )
        image_name = f"{frame_no:05}.png"
        for camera in all_cameras:
            # log camera into 3d canvas
            image_path = os.path.join(
                nice_tool_input_folder, camera, "frames", image_name
            ).replace("\\", "/")
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            entity_path_imgs = viewer.get_images_entity_path(camera)
            viewer.log_image(image, entity_path_imgs, img_quality=75)

        for instance in instances:
            if instance is not None:
                instance.visualize(frame_idx)


def entry_point():
    """Entry point for running NICE toolbox rerun visualizations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visual_config",
        default="configs/visualizer_config.toml",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--machine_specifics",
        default="machine_specific_paths.toml",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    main(args.visual_config, args.machine_specifics)
