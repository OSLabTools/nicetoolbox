import os
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename

import initialization as init
import matrix_utils as mut
import numpy as np
import utils.filehandling as fh
from constants import px, py


def clear_frame(frame):
    for widgets in frame.winfo_children():
        widgets.destroy()


def matrix2form(frame, matrix_vars):
    row = tk.Frame(frame, bd=10)
    row.pack(side=tk.TOP, fill=tk.X, padx=px, pady=py)

    for var_name, vars in matrix_vars.items():
        # label on the left
        left = tk.Frame(row)
        left.pack(side=tk.LEFT, padx=px, pady=py)
        label = tk.Label(left, width=8, text=var_name + " = ", anchor="w")
        label.pack(side=tk.LEFT, fill=tk.Y)

        # matrix on the right
        for column_list in vars.T:
            right = tk.Frame(row)
            right.pack(side=tk.LEFT, padx=px, pady=py)
            for var in column_list:
                entry = tk.Entry(right, textvariable=var, width=15)
                entry.pack(side=tk.TOP, fill=tk.X)


def select_dataset_directory(entries):
    directory = askdirectory()
    entries["input_directory"].set(directory)
    entries["message"].set(f"Set input directory to {directory}.")


def select_calibration_file(entries):
    filepath = askopenfilename()
    entries["input_file"].set(filepath)
    entries["message"].set(f"Set calibration file to {filepath}.")


def walk_directory(directory):
    folders = [f.path for f in os.scandir(directory) if f.is_dir()]
    for folder in folders:
        if "oslab" in folder:
            folders.remove(folder)
    return folders


def find_video_files(directory):
    formats = ["avi", "mp4", "mov"]
    file_names = [f.name for f in os.scandir(directory) if f.is_file()]

    video_files = []
    for file_name in file_names:
        ending = file_name.split(".")[-1]
        name = (".").join(file_name.split(".")[:-1])
        if ending in formats:
            video_files.append(name)

    return video_files


def load_dataset_directory(frame, entries):

    if entries["input_directory"].get() == "":
        entries["message"].set("Please select a dataset directory path before loading.")
        return

    # get fields
    fields = init.get_fields(entries)
    if fields is None:
        return

    # delete current matrix variables as the folder structure of sessions and cameras might change
    if "data" in list(entries.keys()):
        del entries["data"]
    entries["data"] = {}

    directory = entries["input_directory"].get()

    # SESSIONS
    for session_folder in walk_directory(directory):
        session_name = os.path.basename(session_folder)
        entries["data"][session_name] = {}

        sequence_folders = (
            walk_directory(session_folder)
            if walk_directory(session_folder) != []
            else ["None"]
        )

        # SEQUENCES
        for sequence_folder in sequence_folders:
            sequence_name = os.path.basename(sequence_folder)
            entries["data"][session_name][sequence_name] = {}

            if sequence_name == "None" or walk_directory(sequence_folder) == []:
                camera_names = (
                    find_video_files(sequence_folder)
                    if sequence_folder != "None"
                    else find_video_files(session_folder)
                )
            else:
                camera_names = [
                    os.path.basename(cam) for cam in walk_directory(sequence_folder)
                ]

            # CAMERAS
            for camera_name in camera_names:
                # initialize the matrix variables again
                camera_entries = init.init_data_variables(fields)
                camera_entries["name"][0][0].set(camera_name)

                # add the matrix variables to the entries dict
                entries["data"][session_name][sequence_name][
                    camera_name
                ] = camera_entries

    # make the form for the calibration loaded
    make_content_form(frame, entries, fields)

    # done
    entries["message"].set(f"Directory path loaded successfully.")


def load_new_file(frame, entries):

    # get fields
    fields = init.get_fields(entries)
    if fields is None:
        return

    # delete current matrix variables as the folder structure of sessions and cameras might change
    if "data" in list(entries.keys()):
        del entries["data"]
    entries["data"] = {}

    for session in entries["session_names"].get().split(","):
        session = session.strip()
        entries["data"][session] = {}

        for sequence in entries["sequence_names"].get().split(","):
            sequence = sequence.strip() if sequence != "" else "None"
            entries["data"][session][sequence] = {}

            for camera in entries["camera_names"].get().split(","):
                camera = camera.strip()

                # initialize the matrix variables again
                camera_entries = init.init_data_variables(fields)
                camera_entries["name"][0][0].set(camera)

                # add the matrix variables to the entries dict
                entries["data"][session][sequence][camera] = camera_entries

    # make the form for the calibration loaded
    make_content_form(frame, entries, fields)

    # done
    entries["message"].set(f"New file started.")


def load_calibration_file(frame, entries):

    # get fields
    fields = init.get_fields(entries)
    if fields is None:
        return

    # load clibration dictionary
    calibration_file = entries["input_file"].get()
    if calibration_file.endswith(".npz"):
        calibration_dict = np.load(calibration_file, allow_pickle=True)
        load_type = "npz"
    elif calibration_file.endswith(".json"):
        calibration_dict = fh.load_json_file(calibration_file)
        load_type = "json"
    elif calibration_file.endswith(".toml"):
        calibration_dict = fh.load_toml(calibration_file)
        load_type = "toml"
    else:
        entries["message"].set(
            f"Calibration file is not a.npz, .json, or .toml file and currently not supported: '{calibration_file}'"
        )
        return

    # delete current matrix variables as the folder structure of sessions and cameras might change
    if "data" in list(entries.keys()):
        del entries["data"]
    entries["data"] = {}

    for session, calib in calibration_dict.items():

        # get session and sequence names
        if "__" in session:
            session_name, sequence_name = session.split("__")
        else:
            session_name, sequence_name = session, "None"
        # create dictionary entries
        if session_name not in entries["data"].keys():
            entries["data"][session_name] = {}
        entries["data"][session_name][sequence_name] = {}

        # turn the numpy array into a dictionary
        calib_dict = calib.item() if load_type == "npz" else calib
        for camera in calib_dict.values():
            camera_name = camera["camera_name"]

            # initialize the matrix variables again
            camera_entries = init.init_data_variables(fields)

            # search for the chosen matrix format in the calibration dict loaded
            camera_entries = mut.matrix2entries(camera, camera_entries)
            if isinstance(camera_entries, str):
                entries["message"].set(camera_entries)
                return

            # add the matrix variables to the entries dict
            entries["data"][session_name][sequence_name][camera_name] = camera_entries

    # make the form for the calibration loaded
    make_content_form(frame, entries, fields)

    # done
    entries["message"].set(f"Calibration file loaded successfully.")


def make_input_form(frame, main, entries):
    io = tk.Frame(frame)
    io.pack(side=tk.TOP, padx=px, pady=py)

    # input directory
    io_dataset = tk.Frame(io)
    io_dataset.pack(side=tk.TOP, padx=px, pady=py)
    label_dataset = tk.Label(
        io_dataset, text="Dataset directory path: ", anchor="w", justify=tk.LEFT
    )
    label_dataset.pack(side=tk.TOP, fill=tk.X)
    entry_dataset = tk.Entry(
        io_dataset, textvariable=entries["input_directory"], width=90
    )
    entry_dataset.pack(side=tk.LEFT, fill=tk.Y, padx=px, pady=py)
    # create select and load buttons
    button_dataset = tk.Button(
        io_dataset,
        text="Select",
        command=(lambda e=entries: select_dataset_directory(e)),
    )
    button_dataset.pack(side=tk.LEFT, padx=px, pady=py)
    button_load_dataset = tk.Button(
        io_dataset,
        text="Load",
        command=(lambda e=(main, entries): load_dataset_directory(*e)),
    )
    button_load_dataset.pack(side=tk.LEFT, padx=px, pady=py)

    # input calibration file
    io_calibration = tk.Frame(io)
    io_calibration.pack(side=tk.TOP, padx=px, pady=py)
    label_calibration = tk.Label(
        io_calibration, text="Calibration file path: ", anchor="w", justify=tk.LEFT
    )
    label_calibration.pack(side=tk.TOP, fill=tk.X)
    entry_calibration = tk.Entry(
        io_calibration, textvariable=entries["input_file"], width=90
    )
    entry_calibration.pack(side=tk.LEFT, fill=tk.Y, padx=px, pady=py)
    # create select and load buttons
    button_calibration = tk.Button(
        io_calibration,
        text="Select",
        command=(lambda e=entries: select_calibration_file(e)),
    )
    button_calibration.pack(side=tk.LEFT, padx=px, pady=py)
    button_load_calibration = tk.Button(
        io_calibration,
        text="Load",
        command=(lambda e=(main, entries): load_calibration_file(*e)),
    )
    button_load_calibration.pack(side=tk.LEFT, padx=px, pady=py)

    # free form input
    io_free = tk.Frame(io)
    io_free.pack(side=tk.TOP, padx=px, pady=py)
    label_free = tk.Label(
        io_free, text="Start a new file: ", anchor="w", justify=tk.LEFT
    )
    label_free.pack(side=tk.TOP, fill=tk.X)
    entry_free_sess = tk.Entry(io_free, textvariable=entries["session_names"], width=32)
    entry_free_sess.pack(side=tk.LEFT, fill=tk.Y, padx=px, pady=py)
    entry_free_seq = tk.Entry(io_free, textvariable=entries["sequence_names"], width=32)
    entry_free_seq.pack(side=tk.LEFT, fill=tk.Y, padx=px, pady=py)
    entry_free_cam = tk.Entry(io_free, textvariable=entries["camera_names"], width=32)
    entry_free_cam.pack(side=tk.LEFT, fill=tk.Y, padx=px, pady=py)
    button_load_free = tk.Button(
        io_free, text="Load", command=(lambda e=(main, entries): load_new_file(*e))
    )
    button_load_free.pack(side=tk.LEFT, fill=tk.Y, padx=px, pady=py)

    return entries


def make_camera_form(tab, fields, entries):
    header = tk.Frame(tab)
    header.pack(side=tk.TOP, padx=px, pady=py)
    label = tk.Label(header, text="Intrinsic and Extrinsic Camera Matrices")
    label.pack(padx=px, pady=py)

    for field_name, field in fields.items():

        # write description
        label = tk.Label(tab, text=field["description"], anchor="w")
        label.pack(side=tk.TOP, fill=tk.X)

        # each field can be connected to multiple metrics
        matrix_names = [matrix["name"] for matrix in field["variables"]]
        # write matrix or matrices
        matrix2form(
            tab, dict((key, val) for key, val in entries.items() if key in matrix_names)
        )


def make_content_form(frame, entries, fields):

    def create_camera_tabs(tab, tab_entries, fields):
        # create camera tabs
        camera_tabs = ttk.Notebook(tab)
        camera_tabs.pack(expand=1, fill="both")

        for camera_name, camera_entries in tab_entries.items():
            camera_tab = tk.Frame(camera_tabs)
            camera_tabs.add(camera_tab, text=camera_name)

            make_camera_form(camera_tab, fields, camera_entries)

    # clear the form
    clear_frame(frame)

    # create session tabs
    session_tabs = ttk.Notebook(frame)
    session_tabs.pack(expand=1, fill="both")

    for session_name, session_entries in entries["data"].items():
        session_tab = tk.Frame(session_tabs)
        session_tabs.add(session_tab, text=session_name)

        # are there sequences?
        if (
            len(session_entries.keys()) == 1
            and list(session_entries.keys())[0] == "None"
        ):
            # no sequences
            create_camera_tabs(session_tab, session_entries["None"], fields)

        else:
            # create sequence tabs
            sequence_tabs = ttk.Notebook(session_tab)
            sequence_tabs.pack(expand=1, fill="both")

            for sequence_name, sequence_entries in session_entries.items():
                sequence_tab = tk.Frame(sequence_tabs)
                sequence_tabs.add(sequence_tab, text=sequence_name)

                create_camera_tabs(sequence_tab, sequence_entries, fields)
