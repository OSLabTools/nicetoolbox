import os
import sys
import cv2

sys.path.append(os.getcwd())

# internal imports
import visual.configs.config_handler as vis_cfg
from visual.in_out import IO
import utils.visual_utils as vis_utils
from visual.media.viewer import Viewer
from visual.media.components import GazeIndividualComponent, BodyJointsComponent, HandJointsComponent, \
    FaceLandmarksComponent, GazeInteractionComponent, ProximityComponent, KinematicsComponent

def main():
    ############# CONFIGURATION - IO #############
    visualizer_config_file = "visual/configs/visualizer_config.toml"
    machine_specifics_file = "machine_specific_paths.toml"
    config_handler = vis_cfg.Configuration(visualizer_config_file, machine_specifics_file)
    visualizer_config = config_handler.get_updated_visualizer_config()

    # IO
    io = IO(config_handler.get_io_config(add_exp=True))
    io.initialization(visualizer_config)
    nice_tool_input_folder = io.get_component_nice_tool_input_folder(visualizer_config['video'], visualizer_config['media']['dataset_name'])

    ### load calibration for the video
    calibration_file = io.get_calibration_file(visualizer_config['video'])
    calib = vis_utils.load_calibration(calibration_file, visualizer_config['video'], camera_names='all')

    ############## INITIALIZE VIEWER ############
    viewer = Viewer(visualizer_config)

    ############## CHECK CONFIGURATION ###########
    all_cameras = list(calib.keys())
    config_handler.check_config()
    for cam in all_cameras:
        config_handler.check_calibration(calib, cam)
    viewer.check_multiview()

    ############# LOAD COMPONENTS DATA #############
    components = visualizer_config['media']['visualize']['components']

    if 'body_joints' in components:
        body_joints_component = BodyJointsComponent(visualizer_config, io, viewer, "body_joints")
        eyes_middle_2d_data = body_joints_component.calculate_middle_eyes(dimension=2)
        eyes_middle_3d_data = body_joints_component.calculate_middle_eyes(dimension=3) if \
        visualizer_config['media']['multi_view'] == True else (None, None)
    else:
        body_joints_component = None
        eyes_middle_2d_data = None, None
        eyes_middle_3d_data = None, None

    hand_joints_component = HandJointsComponent(visualizer_config, io, viewer,
                                                "hand_joints") if 'hand_joints' in components else None
    face_landmarks_component = FaceLandmarksComponent(visualizer_config, io, viewer,
                                                      "face_landmarks") if 'face_landmarks' in components else None
    gaze_interaction_component = GazeInteractionComponent(visualizer_config, io, viewer,
                                                          "gaze_interaction") if 'gaze_interaction' in components else None

    look_at_data_tuple = gaze_interaction_component.get_lookat_data() if 'gaze_interaction' in components else None  # returns (data, data_labels)

    gaze_ind_component = GazeIndividualComponent(visualizer_config, io, viewer, "gaze_individual",
                                                 calib, eyes_middle_3d_data,
                                                 look_at_data_tuple) if 'gaze_individual' in components else None
    proximity_component = ProximityComponent(visualizer_config, io, viewer, "proximity",
                                             eyes_middle_3d_data,
                                             eyes_middle_2d_data) if 'proximity' in components else None
    kinematics_component = KinematicsComponent(visualizer_config, io, viewer,
                                               "kinematics") if 'kinematics' in components else None

    instances = [body_joints_component, hand_joints_component, face_landmarks_component,
                 gaze_ind_component, proximity_component, kinematics_component]


    ############## VISUALIZATION ###########
    # # initialize rerun visualizer
    viewer.spawn()

    for frame_idx in range(viewer.get_start_frame(), viewer.get_end_frame(), viewer.get_step()):
        viewer.go_to_timestamp(frame_idx)
        frame_no = viewer.get_video_start() + frame_idx
        image_name = f"{frame_no:05}.png"  ##TODO check in general cases
        for camera in all_cameras:
            # log camera into 3d canvas
            image_path = os.path.join(nice_tool_input_folder, camera, "frames",
                                      image_name).replace("\\", "/")
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if viewer.get_is_camera_position():
                entity_path_cams = viewer.get_camera_pos_entity_path(camera)
                viewer.log_camera(calib[camera], entity_path_cams)

            entity_path_imgs = viewer.get_images_entity_path(camera)
            viewer.log_image(image, entity_path_imgs, img_quality=75)

        for instance in instances:
            if instance is not None:
                instance.visualize(frame_idx)

if __name__ == "__main__":
    main()