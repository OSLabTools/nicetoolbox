
import os
import sys
import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory

sys.path.append(os.getcwd())
import matrix_utils as mut
import initialization as init
import make_forms as mf
import calib_io as io
from constants import px, py


def save(entries):

    if 'data' not in entries.keys():
        entries['message'].set('No data for saving. Please load or create calibration data.')
        return

    if entries['output_directory'].get() == '':
        entries['message'].set('Please select an output directory path before saving.')
        return

    # get current matrix states from entries
    matrix_dict = mut.nested_entries2matrix(entries['data'])

    # calculate all representations
    matrix_dict = mut.fill_nested_matrix_dict(matrix_dict)
    if matrix_dict is None:
        entries['message'].set('Could not calculate all representations.')
        return

    # print them to command out
    # for name, mat in matrix_dict.items():
    #     print(mut.matrix2printstring(mat, name))
    io.save_calibration_npz_json(entries, matrix_dict)


def select_calibration_format(main_frame, entries, chosen_format):

    # clear the main frame, empty form to start fresh
    mf.clear_frame(main_frame)

    if 'data' in entries.keys():
        del entries['data']

    entries['message'].set(f"Format '{chosen_format.get()}' selected.")

    # initialize calibration format
    init.init_calibration_format(entries, chosen_format)


def select_output_directory(entries):
    directory = askdirectory()
    entries['output_directory'].set(directory)
    entries['message'].set(f"Set output directory to {directory}.")


def calibration_converter():
    # create the parent window
    root = tk.Tk()
    root.title("Calibration Converter")

    # Implement different formats
    calib = tk.Frame(root)
    calib.pack(side=tk.TOP, padx=px, pady=py)
    label_calib = tk.Label(calib, text='Calibration format: (clears values entered previously)', 
                           width=118, anchor='w', justify=tk.LEFT)
    label_calib.pack(side=tk.LEFT)
    radio = tk.Frame(root)
    radio.pack(side=tk.TOP, padx=px, pady=py)

    # initialize data: create matrix dictionary
    entries = {}
    entries = init.init_io_variables(entries)
    main = tk.Frame(root)

    # radiobuttons: the user may choose only one
    for opt in ["Camera Matrices", "OpenCV", "Camera Parameters"]:
        button_radio = tk.Radiobutton(
            radio, text=opt, variable=entries['chosen_format'], value=opt, 
            command=(lambda e=(main, entries, entries['chosen_format']): select_calibration_format(*e)))
        button_radio.pack(side=tk.LEFT, padx=px)

    # define input directory and file
    entries = mf.make_input_form(root, main, entries)

    # create message display
    frame_message = tk.Frame(root)
    frame_message.pack(side=tk.TOP, padx=px, pady=py)
    label_message = tk.Label(frame_message, textvariable=entries['message'])
    label_message.pack()

    # place the main window
    main.pack(side=tk.TOP, fill=tk.BOTH, padx=px, pady=py)

    # deine the output directory path
    frame_output = tk.Frame(root)
    frame_output.pack(side=tk.TOP, padx=px, pady=py)
    label_output = tk.Label(frame_output, text='Output directory path: ', anchor='w', justify=tk.LEFT)
    label_output.pack(side=tk.TOP, fill=tk.X)
    entry_output = tk.Entry(frame_output, textvariable=entries['output_directory'], width=101)
    entry_output.pack(side=tk.LEFT, fill=tk.Y, padx=px, pady=py)
    button_output = tk.Button(frame_output, text='Select', command=(lambda e=entries: select_output_directory(e)))
    button_output.pack(side=tk.LEFT, padx=px, pady=py)

    # create quit button
    frame_quit = tk.Frame(root)
    frame_quit.pack(side=tk.BOTTOM, padx=px, pady=py)
    button_quit = tk.Button(frame_quit, text='Quit', command=root.quit)
    button_quit.pack(side=tk.RIGHT, fill=tk.X)
    # create save button
    button_submit = tk.Button(frame_quit, text='Save', command=(lambda e=entries: save(e)))
    button_submit.pack(side=tk.RIGHT, fill=tk.X, padx=px, pady=py)
    label_quit = tk.Label(frame_quit, text='   ', width=85)
    label_quit.pack(side=tk.RIGHT, fill=tk.X)

    root.mainloop()


if __name__ == '__main__':
    calibration_converter()

