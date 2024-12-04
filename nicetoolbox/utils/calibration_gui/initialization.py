import tkinter as tk

import numpy as np

from .constants import all_fields


def init_entries_3x3():
    # declaring float variables for storing a 3x3 matrix
    var_11, var_12, var_13 = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
    var_21, var_22, var_23 = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
    var_31, var_32, var_33 = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()

    entries = np.array(
        [[var_11, var_21, var_31], [var_12, var_22, var_32], [var_13, var_23, var_33]]
    ).T
    return entries


def init_entries_3x1():
    # declaring float variables for storing a 3x1 matrix
    var_11, var_21, var_31 = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()

    entries = np.array([var_11, var_21, var_31]).reshape(3, 1)
    return entries


def init_entries_1x3():
    # declaring float variables for storing a 3x1 matrix
    var_11, var_21, var_31 = tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()

    entries = np.array([var_11, var_21, var_31]).reshape(1, 3)
    return entries


def init_entries_1x5():
    # declaring float variables for storing a 3x1 matrix
    var_1, var_2, var_3, var_4, var_5 = (
        tk.DoubleVar(),
        tk.DoubleVar(),
        tk.DoubleVar(),
        tk.DoubleVar(),
        tk.DoubleVar(),
    )

    entries = np.array([[var_1, var_2, var_3, var_4, var_5]]).reshape(1, 5)
    return entries


def init_entries_1x2():
    # declaring float variables for storing a 3x1 matrix
    var_1, var_2 = tk.DoubleVar(), tk.DoubleVar()

    entries = np.array([[var_1, var_2]]).reshape(1, 2)
    return entries


def init_entries_int():
    # declaring int variable
    var = tk.IntVar()
    return np.array([var]).reshape(1, 1)


def init_entries_str():
    # declaring string variable
    var = tk.StringVar()
    return np.array([var]).reshape(1, 1)


def get_fields(entries):
    if "chosen_matrices" not in entries:
        entries["message"].set("Please select a calibration format before loading.")
        return None

    # get the subset of all_fileds that is chosen
    fields = all_fields.copy()
    for field in list(fields.keys()):
        if field not in entries["chosen_matrices"]:
            del fields[field]

    return fields


def init_data_variables(fields):
    data_entries = {}

    # create tkinter variables for the selected camera matrix format
    for _field_name, field in fields.items():
        for variable in field["variables"]:
            if variable["type"] == "3x3":
                data_entries[variable["name"]] = init_entries_3x3()
            elif variable["type"] == "3x1":
                data_entries[variable["name"]] = init_entries_3x1()
            elif variable["type"] == "1x3":
                data_entries[variable["name"]] = init_entries_1x3()
            elif variable["type"] == "1x5":
                data_entries[variable["name"]] = init_entries_1x5()
            elif variable["type"] == "1x2":
                data_entries[variable["name"]] = init_entries_1x2()
            elif variable["type"] == "int":
                data_entries[variable["name"]] = init_entries_int()
            elif variable["type"] == "str":
                data_entries[variable["name"]] = init_entries_str()
            else:
                raise NotImplementedError(variable["type"])

    # set default values for rotation
    if "R" in data_entries:
        data_entries["R"][0, 0].set(1.0)
        data_entries["R"][1, 1].set(1.0)
        data_entries["R"][2, 2].set(1.0)

    return data_entries


def init_calibration_format(entries, chosen_format):
    # which camera matrix formats are requested?
    match chosen_format.get():
        case "Camera Matrices":
            chosen_matrices = ["K", "Rt", "d", "cams"]
        case "OpenCV":
            chosen_matrices = ["mtx", "rtvec", "dist", "cams"]
        case _:
            entries["message"].set(
                f"Format '{chosen_format.get()}' is not yet implemented."
            )
            return

    entries["chosen_format"] = chosen_format
    entries["chosen_matrices"] = chosen_matrices


def init_io_variables(entries):
    # add config entries
    entries["input_file"] = tk.StringVar()
    entries["input_directory"] = tk.StringVar()
    entries["output_directory"] = tk.StringVar()
    entries["session_names"] = tk.StringVar()
    entries["session_names"].set("session1, session2, ...")
    entries["sequence_names"] = tk.StringVar()
    entries["sequence_names"].set("sequence1, sequence2, ... or None")
    entries["camera_names"] = tk.StringVar()
    entries["camera_names"].set("camera1, camera2, ...")
    entries["message"] = tk.StringVar()
    entries["message"].set(" ")
    entries["chosen_format"] = tk.StringVar()

    return entries
