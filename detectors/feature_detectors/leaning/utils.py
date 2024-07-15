import os
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def calculate_angle_btw_three_points(data):
    """
    Calculate the angle between three points in a 3D space.

    Args:
        data (numpy.ndarray): A 4D numpy array with shape (num_subjects, num_frames, num_points,
            num_coordinates) representing the coordinates of points A, B, and C for each subject 
            and frame.

    Returns:
        numpy.ndarray: A 4D numpy array with shape (num_subjects, num_frames, num_points, 1) 
            containing the angle between the three points for each subject and frame.
    """
    # Extract the coordinates of A, B, and C for all frames
    A = data[:, :, :, 0]
    B = data[:, :, :, 1]
    C = data[:, :, :, 2]
    # Compute vectors AB and BC for all frames
    AB = B - A
    BC = C - B

    # Compute the dot product and magnitudes for all frames
    dot_product = np.sum(AB * BC, axis=-1, keepdims=True)
    mag_AB = np.linalg.norm(AB, axis=-1, keepdims=True)
    mag_BC = np.linalg.norm(BC, axis=-1, keepdims=True)

    # Compute the cosine of the angle
    cos_angle = dot_product / (mag_AB * mag_BC)

    # Clip values to be in the range [-1, 1] to avoid issues with numerical precision
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Compute the angles in radians
    angles_rad = np.arccos(cos_angle)
    # Convert to degrees
    angles_deg = np.degrees(angles_rad)
    return angles_deg

def visualize_lean_in_out_per_person(hip_angle, person_list, output_folder, camera_names=None):
    """
    Visualize the leaning angle between midpoint of shoulders, hips, and knees for each person.

    Args:
        hip_angle (numpy.ndarray): A 4D numpy array with shape (num_subjects, num_cameras, 
            num_frames, 2) representing the leaning angles (axis angle and derivative) for each 
            subject, camera, and frame.
        person_list (list): A list of strings representing the names of the persons in frame.
        output_folder (str): The path to the output folder where the plots will be saved.
        camera_names (list, optional): A list of strings representing the names of the cameras. 
            Defaults to None.

    Returns:
        None
    """
    if len(hip_angle)!=len(person_list):
        logging.error("Number of subjects and data shape mismatch!")

    for camera_idx in range(hip_angle.shape[1]):
        plt.title(f'Leaning Angle between Midpoint of Shoulders, Hips, and Knees '
                f'({person_list})')
        fig, axes = plt.subplots(2, len(person_list), figsize=(10, 8))
        axes = axes.reshape(2, len(person_list))

        for i in range(len(person_list)):
            axes[0, i].plot(hip_angle[i, camera_idx, :, 0], label=f'Leaning Angle {person_list[i]}')
            axes[1, i].plot(hip_angle[i, camera_idx, :, 1], label=f'Derivative of Leaning Angle {person_list[i]}')

        # Set labels and legends for each subplot
        axes[0, i].set_xlabel('FrameNo')
        axes[1, i].set_xlabel('FrameNo')
        axes[0, i].set_ylabel('AxisAngle')
        axes[1, i].set_ylabel('Gradient of AxisAngle')
        axes[0, i].legend()
        axes[1, i].legend()
        # axes[0, i].set_ylim(25, 120)
        # axes[1, i].set_ylim(-10,10)

        # Adjust layout
        plt.tight_layout()
        # Save the plot
        camera_name = camera_names[camera_idx] if camera_names is not None else f"camera_{camera_idx}"
        plt.savefig(os.path.join(output_folder, f'leaning_angle_graph_{camera_name}.png'), dpi=500)

def frame_with_linegraph(frame, current_frame, data, fig, canvas, axL, axR):
    """
    Combine a video frame with the plots for PersonL and PersonR up to the current frame.

    Args:
        frame (numpy.ndarray): The current video frame as a numpy array.
        current_frame (int): The index of the current frame.
        data (list): A list containing two numpy arrays, dataL and dataR, representing the 
            leaning angles for PersonL and PersonR respectively.
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        canvas (matplotlib.backends.backend_agg.FigureCanvasAgg): The matplotlib canvas object.
        axL (matplotlib.axes._subplots.AxesSubplot): The left subplot axis for PersonL.
        axR (matplotlib.axes._subplots.AxesSubplot): The right subplot axis for PersonR.

    Returns:
        numpy.ndarray: A numpy array representing the combined video frame and the plots.

    """
    colors = ['#98FB98', '#FFB347', '#DDA0DD', '#ADD8E6']
    if len(data) != 2:
        logging.error(f"The data shape is wrong. Data should be given as a list [dataL, dataR]")
    dataL, dataR = data
    axL.clear()
    axR.clear()
    axL.plot(dataL[:current_frame], label="leaning_angle", color=colors[1])
    axR.plot(dataR[:current_frame], label="leaning_angle", color=colors[1])

    # Redraw the canvas
    canvas.draw()
    graph_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    combined_img = cv2.vconcat([frame, graph_img])
    return combined_img

def create_video_canvas(num_of_frames, global_min, global_max):
    """
    Create a matplotlib figure and canvas for creating a video with two subplots for PersonL 
    and PersonR.
    
    Args:
        num_of_frames (int): The total number of frames in the video.
        global_min (float): The global minimum value for the y-axis.
        global_max (float): The global maximum value for the y-axis.

    Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        canvas (matplotlib.backends.backend_agg.FigureCanvasAgg): The matplotlib canvas object.
        axL (matplotlib.axes._subplots.AxesSubplot): The left subplot axis for PersonL.
        axR (matplotlib.axes._subplots.AxesSubplot): The right subplot axis for PersonR.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.4, 2.4))
    fig.patch.set_facecolor('black')

    for ax, title in zip([axL, axR], ['PersonL', 'PersonR']):
        ax.set_title(title, color='white')
        ax.set_facecolor('black')
    for ax in [axL, axR]:
        ax.set_xlim(0, num_of_frames)
        ax.set_ylim(global_min, global_max)
        ax.set_xticks(range(num_of_frames + 1))
        ax.tick_params(axis='both', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.subplots_adjust(bottom=0.20)
    leg = plt.legend(loc='upper center', bbox_to_anchor=(1.0, 0.8), facecolor='black')
    for text in leg.get_texts():
        text.set_color("white")
    canvas = FigureCanvas(fig)
    return fig, canvas, axL, axR

