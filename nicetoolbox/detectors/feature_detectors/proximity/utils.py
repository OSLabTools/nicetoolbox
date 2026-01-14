"""
Utility functions for visualizing proximity scores.
"""

import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def visualize_proximity_score(data, output_folder, keypoint, camera_names=None):
    """
    Visualizes the proximity score for a given data array and saves the plots as images.

    Images are created for each camera. If the number of keypoints is greater than 1,
    the proximity score is visualized for the center of the selected keypoints.

    Args:
        data (numpy.ndarray): The data array containing proximity scores.
        output_folder (str): The path to the output folder where the images will be
            saved.
        keypoint (str or list): The name(s) of the keypoint(s) used for calculating
            proximity scores.
        camera_names (list, optional): The names of the cameras. Defaults to None.

    Returns:
        None
    """
    for camera_idx in range(data.shape[1]):
        camera_name = camera_names[camera_idx] if camera_names is not None else "3d"
        unit = "(in pixels)" if camera_name != "3d" else "(real-world units: m/cm/mm)"
        plt.clf()
        plt.figure(figsize=(10, 5))
        # Plot the distances for the average coordinates of the selected keypoints
        # across all frames
        plt.plot(data[0, camera_idx])
        plt.xlabel("Frame Index")
        plt.ylabel(f"Proximity Score {unit}")
        if len(keypoint) == 1:
            title = f"Distance between {keypoint[0]} across individuals"
        else:
            title = f"Distance between center of selected keypoints {keypoint}"
            "across individuals"
        plt.title(title)

        # Save the plot
        plt.savefig(
            os.path.join(output_folder, f"proximity_score_{keypoint}_{camera_name}.png"),
            dpi=500,
        )


def frame_with_linegraph(frame, data, current_frame, global_min, global_max):
    """
    Combines a video frame with line graphs representing proximity scores up to the
    current frame.

    Args:
        frame (numpy.ndarray): The video frame to be combined with the line graphs.
        data (list): The list of data arrays containing proximity scores.
        current_frame (int): The current frame index.
        global_min (float): The minimum value for the y-axis of the line graphs.
        global_max (float): The maximum value for the y-axis of the line graphs.

    Returns:
        numpy.ndarray: The combined image of the frame and line graphs.
    """
    if len(data) != 1:
        logging.error("The data shape is wrong. Data should be given as into a list")

    data = data[0]
    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
    fig.patch.set_facecolor("black")

    colors = ["#98FB98", "#FFB347", "#DDA0DD", "#ADD8E6"]
    ax.plot(data[:current_frame], label="proximity", color=colors[0])

    ax.set_title("proximity", color="white")
    ax.set_facecolor("black")

    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(global_min, global_max)
    ax.set_xticks(range(data.shape[0] + 1))
    ax.tick_params(axis="both", colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.subplots_adjust(bottom=0.20)

    leg = plt.legend(loc="upper center", bbox_to_anchor=(1.0, 0.8), facecolor="black")
    for text in leg.get_texts():
        text.set_color("white")

    canvas = FigureCanvas(fig)
    canvas.draw()
    graph_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Resize the graph_img to match the width of the frame
    graph_img = cv2.resize(graph_img, (frame.shape[1], graph_img.shape[0]))

    combined_img = cv2.vconcat([frame, graph_img])
    return combined_img
