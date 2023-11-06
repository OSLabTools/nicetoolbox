import os
import sys
import cv2
import numpy as np
import re

sys.path.append("./third_party/xgaze_3cams/xgaze_3cams")
sys.path.append("./third_party")
print(sys.path)
from gaze_estimator import GazeEstimator
from utils import vector_to_pitchyaw, draw_gaze, get_cam_para_studio
from landmarks import get_landmarks_1

from file_handling import load_config, save_to_hdf5


def main(config):
    """ Run inference of the method on the pre-loaded image
    """

    print('\n\n\t RUNNING xgaze_3cams!')

    # check that the data was created
    assert config['frames_list'] is not None, \
        f"{config['name']}: Please specify 'input_data_format: frames' in " \
        f"config. Currently it is {config['input_data_format']}."

    gaze_estimator = GazeEstimator(
            config['face_model_filename'],
            config['pretrained_model_filename'])

    # re-organize input data
    n_cams = len(config['camera_names'])
    n_frames = len(config['frames_list']) // n_cams
    frames_list = np.array(sorted(config['frames_list'])
                           ).reshape(n_cams, n_frames).T
    frames_list = [list(l) for l in frames_list]

    # convert camera_names to camera_ids
    camera_ids = []
    for cam_name in config['camera_names']:
        cam_id = [int(x) for x in re.findall(r'\d+', cam_name)]
        assert len(cam_id) == 1, f"ERROR: invalid camera_name '{cam_name}'!"
        camera_ids.append(cam_id[0])

    results_sub1, results_sub2 = [], []
    ## read images
    for frame_files, frame_id in zip(frames_list, config['frame_indices_list']):
        images = []  # all images for that frame
        file_paths = []  # all image file path for that frame
        sub_1_landmarks = []  # all landmarks for that frame
        sub_2_landmarks = []  # all landmarks for that frame

        # get the landmarks
        for frame_file, cam_id in zip(frame_files, camera_ids):
            image = cv2.imread(frame_file)
            images.append(image)

            file_paths.append(frame_file)
            landmark_1, landmark_2 = get_landmarks_1(
                    image,
                    cam_id,
                    config['shape_predictor_filename'],
                    config['face_detector_filename']
            )
            sub_1_landmarks.append(landmark_1)
            sub_2_landmarks.append(landmark_2)

        # OK, now let's do gaze estimation
        for sub_id in range(1, 3):  # loop it for each subject
            gaze_world = []  # the final gaze direction in the world coordinate system
            if sub_id == 1:
                landmarks = sub_1_landmarks
            else:
                landmarks = sub_2_landmarks
            for cam_id in camera_ids:
                if landmarks[cam_id - 1] is None:
                    continue
                image = images[cam_id - 1]
                cam_matrix, cam_distor, cam_rotation = get_cam_para_studio(
                        config['calibration'], cam_id, image)
                pred_gaze = gaze_estimator.gaze_estimation(
                        image,
                        landmarks[cam_id - 1],
                        cam_matrix,
                        cam_distor
                )

                # convert the gaze direction to cam4, the world coordinate system
                pred_gaze_cam4 = np.dot(np.linalg.inv(cam_rotation),
                                        pred_gaze)  # to the cam4 coordinate system
                pred_gaze_cam4 = pred_gaze_cam4.reshape((1, 3))
                pred_gaze_cam4 = pred_gaze_cam4 / np.linalg.norm(
                        pred_gaze_cam4)
                gaze_world.append(pred_gaze_cam4.reshape((1, 3)))
            # get the final gaze direction by averaging them
            gaze_world = np.asarray(gaze_world).reshape((-1, 3))
            gaze_world = np.mean(gaze_world, axis=0).reshape((-1, 3))
            if sub_id == 1:
                gaze_world_sub_1 = gaze_world
                results_sub1.append(gaze_world.flatten())
            else:
                gaze_world_sub_2 = gaze_world
                results_sub2.append(gaze_world.flatten())

        # visualization on each image, project the gaze back to each cam to draw the gaze direction
        is_debug = False
        for cam_id in camera_ids:
            image = images[cam_id - 1]
            cam_matrix, cam_distor, cam_rotation = get_cam_para_studio(
                    config['calibration'], cam_id, image)
            for sub_id in range(1, 3):
                if sub_id == 1:
                    landmarks = sub_1_landmarks
                    gaze_world = gaze_world_sub_1
                else:
                    landmarks = sub_2_landmarks
                    gaze_world = gaze_world_sub_2
                if landmarks[cam_id - 1] is None:
                    continue

                # convert the gaze to current camera coordinate system
                gaze_cam = np.dot(cam_rotation, gaze_world.T)
                draw_gaze_dir = vector_to_pitchyaw(gaze_cam).reshape(-1)
                face_center = np.mean(landmarks[cam_id - 1], axis=0)
                draw_gaze(image,
                          draw_gaze_dir,
                          thickness=2,
                          color=(0, 0, 255),
                          position=face_center.astype(int)
                          )
            if is_debug:
                cv2.imshow("img_show", image)
                cv2.waitKey(0)

            # save the image with gaze direction
            _, file_name = os.path.split(file_paths[cam_id - 1])
            save_file_name = os.path.join(
                    config['out_folder'], f'cam{cam_id}_{file_name}')
            cv2.imwrite(save_file_name, image)

    save_to_hdf5(
            [np.array(results_sub1), np.array(results_sub2)],
            ['personL', 'personR'],
            os.path.join(config['result_folder'], f"{config['behavior']}.hdf5")
    )


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = load_config(config_path)
    main(config)
