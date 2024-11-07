import json
import os

import numpy as np


def get_calibration_dict(matrix_dict):

    calibration_dict = {}
    for camera_name, camera_dict in matrix_dict.items():

        calibration_dict.update(
            {
                camera_name: dict(
                    camera_name=camera_dict["name"].tolist(),
                    image_size=camera_dict["size"].tolist(),
                    intrinsic_matrix=camera_dict["K"],
                    distortions=camera_dict["d"].squeeze(),
                    rotation_matrix=camera_dict["R"],
                    rvec=camera_dict["rvec"],
                    translation=camera_dict["t"].squeeze(),
                    extrinsics_matrix=camera_dict["Rt"],
                    projection_matrix=camera_dict["P"],
                )
            }
        )
    return calibration_dict


def default(obj):
    # serialize numpy array for saving to json
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError("Unknown type:", type(obj))


def save_calibration_npz_json(entries, matrix_dict):

    calibration = {}
    # convert matrix_dict
    for session_name, session_data in matrix_dict.items():
        for sequence_name, sequence_data in session_data.items():
            name = "__".join(
                [word for word in [session_name, sequence_name] if word != "None"]
            )
            calibration[name] = get_calibration_dict(sequence_data)

    # save to npz
    out_file_npz = os.path.join(entries["output_directory"].get(), "calibrations.npz")
    np.savez_compressed(out_file_npz, **calibration)

    # save to json
    out_file_json = os.path.join(entries["output_directory"].get(), "calibrations.json")
    with open(out_file_json, "w") as file:
        json.dump(calibration, file, default=default)

    entries["message"].set(f"Saved successfully to '{out_file_npz}'.")
