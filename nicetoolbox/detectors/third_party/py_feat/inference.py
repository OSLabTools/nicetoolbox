"""
Run Py-FEAT on the data.
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feat import Detector

top_level_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(top_level_dir))
# Internal and third party imports
from utils import filehandling as fh  # noqa: E402


def main(config):
    """
    Run Py-feat on the provided data.

    This function uses 'py-feat' library to estimate emotion range for each camera.
    The resulting frame to emotion intensity dataframe is saved in a .npz file of the
    following structure:
        - TODO

    Args:
        config(dict): Configuration dictionary with parameters for
            emotion intensity detection
        debug(bool, optional): Flag indicating whether to print debug information.
            Defaults to False
    """

    # Initialise logger
    logging.basicConfig(
        filename=config["log_file"],
        level=config["log_level"],
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s",
    )
    logging.info("Running emotion intensity detection 'py-feat'!")

    # Check if data was created
    assert config["frames_list"] is not None, (
        f"{config['name']}: Please specify 'input_data_format': frames' in "
        f"config. Currently it is {config['input_data_format']}"
    )

    # Get valid camera names
    camera_names = sorted([n for n in config["camera_names"] if n != ""])
    n_cams = len(camera_names)

    # Re-organise input data
    n_subjects = len(config["subjects_descr"])
    if len(np.array(config["frames_list"]).shape) == 2:
        assert np.array(config["frames_list"]).shape[1] == n_cams, (
            "Number of cameras do not match! Please check the camera's definitions "
            "in detectors/configs/datset_properties.toml."
        )
        frames_list = np.array(config["frames_list"]).flatten()
    else:
        frames_list = config["frames_list"]
    n_frames = len(frames_list) // n_cams
    frames_list = np.array(sorted(frames_list)).reshape(n_cams, n_frames).T
    frames_list = [list(l) for l in frames_list]  # noqa: E741

    # Start Emotion Intensity detection
    logging.info("Load py-feat and start detection.")
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        facepose_model="img2pose",
    )

    frames_list_centre = [str(imglist[0]) for imglist in frames_list]
    frames_list_left = [str(imglist[1]) for imglist in frames_list]
    frames_list_right = [str(imglist[2]) for imglist in frames_list]
    frames_list_top = [str(imglist[3]) for imglist in frames_list]

    img_centre_output = detector.detect_image(
        frames_list_centre, batch_size=int(config["batch_size"])
    )
    img_left_output = detector.detect_image(
        frames_list_left, batch_size=int(config["batch_size"])
    )
    img_right_output = detector.detect_image(
        frames_list_right, batch_size=int(config["batch_size"])
    )
    # Switching Person_0 with Person_1 as for nice toolbox persons are counted
    # from left(0) to right(1) and pyfeat labels person in the right image as Person_0
    img_right_output["Identity"] = img_right_output["Identity"].replace(
        "Person_0", "Person_1"
    )
    img_top_output = detector.detect_image(
        frames_list_top, batch_size=int(config["batch_size"])
    )

    all_identities = pd.concat(
        [
            img_centre_output["Identity"],
            img_left_output["Identity"],
            img_right_output["Identity"],
            img_top_output["Identity"],
        ]
    ).unique()
    identity_to_idx = {identity: idx for idx, identity in enumerate(all_identities)}
    n_subjects = len(identity_to_idx)

    frame_indices = []
    faceboxes = np.zeros((n_frames, n_cams, n_subjects, 5))
    aus = np.zeros((n_frames, n_cams, n_subjects, 20))
    emotions = np.zeros((n_frames, n_cams, n_subjects, 7))
    # Poses only contains Pitch, Roll, and Yaw
    # Documentation suggested additional X,Y,Z
    # so only 3 components in following ndarray
    poses = np.zeros((n_frames, n_cams, n_subjects, 3))

    outputs = [img_centre_output, img_left_output, img_right_output, img_top_output]

    # store faceboxes, action units, emotions, poses into ndarrays
    for frame_i, frame_files in enumerate(frames_list):
        frame_name = [
            os.path.basename(file).strip(".png").strip(".jpg").strip(".jpeg")
            for file in frame_files
        ]
        if len(set(frame_name)) != 1:
            logging.warning(
                f"Found multiple frame names '{frame_name}' for frame index {frame_i}."
            )
        else:
            frame_indices.append(frame_name[0])

        for cam_i, output in enumerate(outputs):
            # Filter detections for the current frame
            frame_data = output[output["frame"] == frame_i]
            facebox_data = frame_data.faceboxes.to_numpy()
            aus_data = frame_data.aus.to_numpy()
            emotions_data = frame_data.emotions.to_numpy()
            poses_data = frame_data.poses.to_numpy()

            # Assign facebox data to the corresponding subject index
            for rel_idx, (_index, row) in enumerate(frame_data.iterrows()):
                subject_idx = identity_to_idx[row["Identity"]]
                faceboxes[frame_i, cam_i, subject_idx, :] = facebox_data[rel_idx, :]
                aus[frame_i, cam_i, subject_idx, :] = aus_data[rel_idx, :]
                emotions[frame_i, cam_i, subject_idx, :] = emotions_data[rel_idx, :]
                poses[frame_i, cam_i, subject_idx, :] = poses_data[rel_idx, :]

        # If visualize is true and the output numpy arrays have no NaNs
        if (
            config["visualize"]
            and (faceboxes[frame_i] == faceboxes[frame_i]).all()
            and (aus[frame_i] == aus[frame_i]).all()
            and (emotions[frame_i] == emotions[frame_i]).all()
            and (poses[frame_i] == poses[frame_i]).all()
        ):
            for output, cam_name in zip(outputs, camera_names):
                frame_data = output[output["frame"] == frame_i]
                frame_file_path = frame_data["input"].to_list()
                # Visualization for frames
                fig = frame_data.plot_detections()
                # Add legend for multiple subject detection cases
                if frame_data["Identity"].nunique() > 1:
                    fig[0].axes[1].legend(
                        config["subjects_descr"], loc="upper right", fontsize=8
                    )
                    fig[0].axes[2].legend(
                        config["subjects_descr"], loc="upper right", fontsize=8
                    )
                # save the image
                _, file_name = os.path.split(frame_file_path[0])
                save_file_name = os.path.join(
                    config["out_folder"], f"{cam_name}_{file_name}"
                )
                fig[0].savefig(str(save_file_name))
                plt.close()

        if frame_i % config["log_frame_idx_interval"] == 0:
            logging.info(f"Finished frame {frame_i} / {n_frames}.")

    if not config["visualize"]:
        logging.info("Visualization of images turned off.")

    # save as npz files
    out_dict = {
        "faceboxes": np.array(faceboxes, dtype=float).transpose(2, 1, 0, 3),
        "aus": np.array(aus, dtype=float).transpose(2, 1, 0, 3),
        "emotions": np.array(emotions, dtype=float).transpose(2, 1, 0, 3),
        "poses": np.array(poses, dtype=float).transpose(2, 1, 0, 3),
        "data_description": {
            "faceboxes": dict(
                axis0=config["subjects_descr"],
                axis1=camera_names,
                axis2=frame_indices,
                axis3=[
                    "FaceRectX",
                    "FaceRectY",
                    "FaceRectWidth",
                    "FaceRectHeight",
                    "FaceScore",
                ],
            ),
            "aus": dict(
                axis0=config["subjects_descr"],
                axis1=camera_names,
                axis2=frame_indices,
                axis3=[
                    "AU01",
                    "AU02",
                    "AU04",
                    "AU05",
                    "AU06",
                    "AU07",
                    "AU09",
                    "AU10",
                    "AU11",
                    "AU12",
                    "AU14",
                    "AU15",
                    "AU17",
                    "AU20",
                    "AU23",
                    "AU24",
                    "AU25",
                    "AU26",
                    "AU28",
                    "AU43",
                ],
            ),
            "emotions": dict(
                axis0=config["subjects_descr"],
                axis1=camera_names,
                axis2=frame_indices,
                axis3=[
                    "anger",
                    "disgust",
                    "fear",
                    "happiness",
                    "sadness",
                    "surprise",
                    "neutral",
                ],
            ),
            "poses": dict(
                axis0=config["subjects_descr"],
                axis1=camera_names,
                axis2=frame_indices,
                axis3=["Pitch", "Roll", "Yaw"],
            ),
        },
    }

    save_file_name = os.path.join(
        config["result_folders"]["emotion_individual"], f"{config['algorithm']}.npz"
    )
    np.savez_compressed(save_file_name, **out_dict)

    logging.info("'py_feat' COMPLETED!\n")


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = fh.load_config(config_path)
    main(config)
