import cv2
import numpy as np

from . import constants as const


def nested_entries2matrix(entries):
    matrix_dict = {}

    for session_name, session_data in entries.items():
        matrix_dict[session_name] = {}

        for sequence_name, sequence_data in session_data.items():
            matrix_dict[session_name][sequence_name] = {}

            for camera_name, camera_data in sequence_data.items():
                matrix_dict[session_name][sequence_name][camera_name] = entries2matrix(
                    camera_data
                )

    return matrix_dict


def entries2matrix(entries):
    matrix_dict = {}
    for name, vars in entries.items():
        # only convert matrices, no strings
        if not isinstance(vars, np.ndarray):
            continue

        mat = np.empty_like(vars)
        # create numpy array from variable values
        for i, row in enumerate(vars):
            mat[i] = np.array([var.get() for var in row])

        if name == "name":
            matrix_dict[name] = mat.astype(str)
        elif name == "size":
            matrix_dict[name] = mat.astype(int)
        else:
            matrix_dict[name] = mat.astype(float)
    return matrix_dict


def matrix2entries(matrix_dict, entries):
    matrix_keys = list(matrix_dict.keys())

    for name, vars in entries.items():
        # check for variable being present in the given matrix dictionary
        synonyms = [syns for syns in const.matrix_name_synonyms if name in syns][0]
        matrix_key = [key for key in matrix_keys if key in synonyms]
        if matrix_key == []:
            entries["message"].set(f"'{name}' not found in loaded calibration file.")
            continue

        # get matrix and turn into numpy array
        matrix = np.array(matrix_dict[matrix_key[0]])

        # hack for 4-dimensional distortions
        if matrix_key[0] == "distortions" and len(matrix) == 4:
            matrix = np.array(matrix_dict[matrix_key[0]] + [0.0])
        if name == "R" and matrix.shape == (3, 4):
            t_dist = (
                matrix[:, 3].flatten() - np.array(matrix_dict["translation"]).flatten()
            )
            if np.linalg.norm(t_dist) > 0.01:
                return (
                    "Loaded rotation matrix of shape 3x4 does not align with the "
                    "loaded translation vector."
                )
            matrix = matrix[:, :3]

        try:
            matrix = matrix.reshape(vars.shape)
        except ValueError:
            return (
                f"Shape mismatch! Loaded '{matrix_key}' and variable '{name}' "
                "do not match."
            )

        for i, row in enumerate(vars):
            for j, item in enumerate(row):
                item.set(matrix[i, j])

    return entries


def fill_matrix_dict(matrix_dict):
    if all([name in list(matrix_dict.keys()) for name in ["K", "R", "d", "t"]]):
        K = matrix_dict["K"]
        R = matrix_dict["R"]
        t = matrix_dict["t"]

        matrix_dict["mtx"] = K
        matrix_dict["rvec"] = cv2.Rodrigues(R)[0]
        matrix_dict["tvec"] = t
        matrix_dict["dist"] = matrix_dict["d"]

    elif all(
        [name in list(matrix_dict.keys()) for name in ["mtx", "dist", "rvec", "tvec"]]
    ):
        K = matrix_dict["mtx"]
        R = cv2.Rodrigues(matrix_dict["rvec"])[0]
        t = matrix_dict["tvec"]

        matrix_dict["K"] = K
        matrix_dict["R"] = R
        matrix_dict["t"] = t
        matrix_dict["d"] = matrix_dict["dist"]

    else:
        return None

    Rt = np.concatenate((R, t), axis=1)
    matrix_dict["Rt"] = Rt
    matrix_dict["P"] = create_projection_matrix(K, Rt)

    return matrix_dict


def fill_nested_matrix_dict(entries):
    matrix_dict = {}

    for session_name, session_data in entries.items():
        matrix_dict[session_name] = {}

        for sequence_name, sequence_data in session_data.items():
            matrix_dict[session_name][sequence_name] = {}

            for camera_name, camera_data in sequence_data.items():
                matrix_dict[session_name][sequence_name][camera_name] = (
                    fill_matrix_dict(camera_data)
                )

    return matrix_dict


def create_projection_matrix(int_matrix, ext_matrix):
    """
    Create a projection matrix from intrinsic and extrinsic matrices.

    Args:
        int_matrix (np.array): The intrinsic matrix (3x3).
        ext_matrix (np.array): The extrinsic matrix (3x4).

    Returns:
        np.array: The projection matrix (3x4).
    """
    projection_matrix = np.matmul(int_matrix, ext_matrix)
    return projection_matrix
