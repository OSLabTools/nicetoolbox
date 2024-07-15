import os
import sys
import cv2
import numpy as np
import re
import logging

sys.path.append(os.getcwd())
import utils.config as cf

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgaze_3cams'))
from gaze_estimator import GazeEstimator
from xgaze_utils import vector_to_pitchyaw, draw_gaze, get_cam_para_studio
import landmarks as lm


# TODO: Split into smaller functions: e.g. load_data, run_inference, visualize_gaze_direction, save_results
def main(config, debug=False):
    """ 
    Run xgaze_3cams gaze detection on the provided data.
    
    The function uses the 'xgaze_3cams' library to estimate gaze vectors for each camera.
    The estimated gaze vectors are then converted to pitch and yaw angles using a simple
    linear transformation. The resulting angles are saved in a .npz file with the following
    structure:
        - 3d: Numpy array of shape (n_frames, n_subjects, 3)
        - data_description: A dictionary containing the description of the data.
    
    Args:
        config (dict): The configuration dictionary containing parameters for gaze detection.
        debug (bool, optional): A flag indicating whether to print debug information.
            Defaults to False.
    """

    # initialize logger
    logging.basicConfig(filename=config['log_file'], level=config['log_level'],
                        format='%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s')
    logging.info("RUNNING gaze detection 'xgaze_3cams'!")

    # check that the data was created
    assert config['frames_list'] is not None, \
        f"{config['name']}: Please specify 'input_data_format: frames' in " \
        f"config. Currently it is {config['input_data_format']}."

    # get valid camera_names
    camera_names = sorted([n for n in config['camera_names'] if n != ''])
    n_cams = len(camera_names)

    # re-organize input data
    n_subjects = len(config['subjects_descr'])
    if len(np.array(config['frames_list']).shape) == 2:
        assert np.array(config['frames_list']).shape[1] == n_cams, \
            "config['frames_list' has unknown shape!"
        frames_list = np.array(config['frames_list']).flatten()
    else:
        frames_list = config['frames_list']
    n_frames = len(frames_list) // n_cams
    frames_list = np.array(sorted(frames_list)).reshape(n_cams, n_frames).T
    frames_list = [list(l) for l in frames_list]

    # start gaze detection
    logging.info('Load gaze estimator and start detection.')
    gaze_estimator = GazeEstimator(
            config['face_model_filename'],
            config['pretrained_model_filename'])
    
    face_detector = lm.get_face_detector(config['shape_predictor_filename'],
                                      config['face_detector_filename'])

    frame_indices = []
    results = np.zeros((n_frames, n_subjects, 3))
    results_2d = np.zeros((n_frames, n_cams, n_subjects, 2))
    landmarks_2d = np.zeros((n_frames, n_cams, n_subjects, 6, 2))
    # landmarks_3d = np.zeros((n_frames, n_cams, n_subjects, 6, 3))

    for frame_i, frame_files in enumerate(frames_list):
        images = []  # (n_cams, h, w, 3)

        frame_name = [os.path.basename(file).strip('.png').strip('.jpg').strip('.jpeg') for file in frame_files]
        if len (set(frame_name)) != 1:
            logging.warning(f"Found multiple frame names '{frame_name}' for frame index {frame_i}.")
        else:
            frame_indices.append(frame_name[0])

        # get the landmarks
        for cam_i, frame_file in enumerate(frame_files):
            image = cv2.imread(frame_file)
            images.append(image)

            landmark_predictions, score = lm.get_landmarks(
                    image,
                    face_detector,
                    debug
            )
            if landmark_predictions is not None and landmark_predictions.shape[0] != n_subjects:
                scores = score.mean(axis=1)
                max_value_indices = sorted(scores.argsort()[-n_subjects:])
                landmark_predictions = landmark_predictions[max_value_indices]

            landmarks_2d[frame_i, cam_i] = landmark_predictions
            # # debugging
            # for point in landmark_predictions.reshape(-1, 3):
            #     cv2.circle(image, point[:2].astype(int), 3, (0,0,255), -1)
            # cv2.imwrite('folder-path/image.png', image)

            # # calculate 3D landmarks from 2.5D predictions
            # cam_matrix, cam_distor, cam_rotation, cam_translation = get_cam_para_studio(
            #     config['calibration'], camera_names[cam_i], image)
            # # z * (u, v, 1)
            # landmark_predictions[..., :2] *= landmark_predictions[..., 2:]
            # # camera coords
            # landmark_camera = np.matmul(landmark_predictions, np.linalg.inv(cam_matrix).T)
            # # world coords
            # landmarks_3d[frame_i, cam_i] = np.matmul((landmark_camera - cam_translation.reshape(1, 1, 3)), np.linalg.inv(cam_rotation).T)
            
        # OK, now let's do gaze estimation - per subject
        for sub_id in range(n_subjects):  # loop it for each subject
            gaze_world = []  # the final gaze direction in the world coordinate system

            assert len(images) == len(camera_names), "Error!"

            # loop over the cameras and predict the gaze for each in 2d
            for image, landmarks, cam_name in zip(images, landmarks_2d[frame_i], camera_names):

                # is this subject visible in the camera? and were landmarks predicted?
                subjects_by_cam = config['cam_sees_subjects'][cam_name]
                if (landmarks is None) or (sub_id not in subjects_by_cam) or (len(landmarks) != len(subjects_by_cam)):
                    continue
                # where to find this subject 'sub_id' in camera 'cam_name'
                sub_cam_id = subjects_by_cam.index(sub_id)

                # DEBUGGING
                # for p in landmarks[0]:
                #     image[p.astype(int)[1], p.astype(int)[0]] = [0, 0, 255]
                
                cam_matrix, cam_distor, cam_rotation, _ = get_cam_para_studio(
                        config['calibration'], cam_name, image)
                pred_gaze = gaze_estimator.gaze_estimation(
                        image,
                        landmarks[sub_cam_id],
                        cam_matrix,
                        cam_distor
                )

                # convert the gaze direction to the world coordinate system
                pred_gaze_world = np.dot(np.linalg.inv(cam_rotation), pred_gaze).reshape((1, 3))
                pred_gaze_world = pred_gaze_world / np.linalg.norm(pred_gaze_world)
                gaze_world.append(pred_gaze_world.reshape((1, 3)))

            # get the final gaze direction by averaging them
            gaze_world = np.asarray(gaze_world).reshape((-1, 3))
            gaze_world = np.mean(gaze_world, axis=0).reshape((-1, 3))
            results[frame_i][sub_id] = gaze_world.flatten()

        if len(camera_names) == 1:
            gaze_camera = np.matmul(cam_rotation, results[frame_i].T).T
            # vector to pitchyaw
            pitchyaw = np.empty((n_subjects, 2))
            gaze_norm = gaze_camera / np.linalg.norm(gaze_camera, axis=1, keepdims=True)
            pitchyaw[:, 0] = np.arcsin(-1 * gaze_norm[:, 1])  # theta
            pitchyaw[:, 1] = np.arctan2(-1 * gaze_norm[:, 0], -1 * gaze_norm[:, 2])  # phi

            length_ratio=5.0
            length = image.shape[1] / length_ratio
            dx = (-length * np.sin(pitchyaw[:, 1]) * np.cos(pitchyaw[:, 0])).astype(int)
            dy = (-length * np.sin(pitchyaw[:, 0])).astype(int)

            # shape (n_subjects, 2)
            results_2d[frame_i, 0] = np.stack((dx, dy), axis=-1)

        if config["visualize"] and (results[frame_i] == results[frame_i]).all():
            # visualization on each image, project the gaze back to each cam to draw the gaze direction
            for image, landmarks, cam_name, file_path in (
                    zip(images, landmarks_2d[frame_i], camera_names, frames_list[frame_i])):
                
                if (landmarks  == landmarks).all() and (len(landmarks) == len(config['cam_sees_subjects'][cam_name])): 
                    cam_matrix, cam_distor, cam_rotation, _ = get_cam_para_studio(
                            config['calibration'], cam_name, image)

                    for sub_id in range(n_subjects):
                        if sub_id not in config['cam_sees_subjects'][cam_name]:
                            continue

                        # where to find this subject 'sub_id' in camera 'cam_name'
                        sub_cam_id = config['cam_sees_subjects'][cam_name].index(sub_id)

                        # convert the gaze to current camera coordinate system
                        gaze_cam = np.dot(cam_rotation, results[frame_i][sub_id].T)
                        draw_gaze_dir = vector_to_pitchyaw(gaze_cam).reshape(-1)
                        face_center = np.mean(landmarks[sub_cam_id], axis=0)
                        draw_gaze(image,
                                draw_gaze_dir,
                                thickness=2,
                                color=(0, 0, 255),
                                position=face_center.astype(int)
                                )
                    if debug:
                        cv2.imshow("img_show", image)
                        cv2.waitKey(0)

                # save the image with gaze direction
                _, file_name = os.path.split(file_path)
                save_file_name = os.path.join(
                        config['out_folder'], f'{cam_name}_{file_name}')
                cv2.imwrite(save_file_name, image)

        if frame_i % config['log_frame_idx_interval'] == 0:
            logging.info(f"Finished frame {frame_i} / {n_frames}.")

    if not config["visualize"]:
        logging.info("Visualization of images turned off.")

    #  save as npz file
    out_dict = {
        '3d': results[None].transpose(2, 0, 1, 3),
        'landmarks_2d': landmarks_2d.transpose(2, 1, 0, 3, 4),
        # 'landmarks_3d': landmarks_3d.transpose(2, 1, 0, 3, 4),
        'data_description': {
            '3d': dict(
                axis0=config["subjects_descr"],
                axis1=None,
                axis2=frame_indices,
                axis3=['coordinate_x', 'coordinate_y', 'coordinate_z']
            ),
            'landmarks_2d': dict(
                axis0=config["subjects_descr"],
                axis1=camera_names,
                axis2=frame_indices,
                axis3=['right_eye_0', 'right_eye_1', 'left_eye_0', 'left_eye_1', 'mouth_0', "mouth_1"],
                axis4=['coordinate_u', 'coordinate_v']
            ),
            # 'landmarks_3d': dict(
            #     axis0=config["subjects_descr"],
            #     axis1=None,
            #     axis2=frame_indices,
            #     axis3='face_landmarks',
            #     axis4=['coordinate_x', 'coordinate_y', 'coordinate_z']
            # ),
        }
    }
    if len(camera_names) == 1:
        out_dict.update({'2d': results_2d.transpose(2, 1, 0, 3)})
        out_dict['data_description'].update({
            '2d': dict(
                axis0=config["subjects_descr"],
                axis1=camera_names,
                axis2=frame_indices,
                axis3=['coordinate_x', 'coordinate_y'],
            )})

    save_file_name = os.path.join(config["result_folders"]['gaze_individual'], f"{config['algorithm']}.npz")
    np.savez_compressed(save_file_name, **out_dict)

    logging.info("Gaze detection 'xgaze_3cams' COMPLETED!\n")


if __name__ == '__main__':
    config_path = sys.argv[1]
    # config_path = '/is/sg2/cschmitt/pis/experiments/20240710/mpi_inf_3dhp_S1_s20_l20/gaze_individual/xgaze_3cams/run_config.toml'
    config = cf.load_config(config_path)
    main(config)
