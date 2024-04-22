import os
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def calculate_angle_btw_three_points(data):
    # Extract the coordinates of A, B, and C for all frames
    A = data[0]
    B = data[1]
    C = data[2]
    # Compute vectors AB and BC for all frames
    AB = B - A
    BC = C - B

    # Compute the dot product and magnitudes for all frames
    dot_product = np.sum(AB * BC, axis=-1)
    mag_AB = np.linalg.norm(AB, axis=-1)
    mag_BC = np.linalg.norm(BC, axis=-1)

    # Compute the cosine of the angle
    cos_angle = dot_product / (mag_AB * mag_BC)

    # Clip values to be in the range [-1, 1] to avoid issues with numerical precision
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Compute the angles in radians
    angles_rad = np.arccos(cos_angle)
    # Convert to degrees
    angles_deg = np.degrees(angles_rad)
    return angles_deg

def visualize_lean_in_out_per_person(hip_angle_person_list, person_list, output_folder):
    if len(hip_angle_person_list)!=len(person_list):
        logging.error("Number of subjects and data shape mismatch!")
    fig, axes = plt.subplots(2, len(person_list), figsize=(10, 8))
    axes = axes.reshape(2, len(person_list))
    plt.title(f'Leaning Angle between Midpoint of Shoulders, Hips, and Knees '
              f'({person_list})')

    for i in range(len(person_list)):
        axes[0, i].plot(hip_angle_person_list[i][:, 0], label=f'Leaning Angle {person_list[i]}')
        axes[1, i].plot(hip_angle_person_list[i][:, 1], label=f'Derivative of Leaning Angle {person_list[i]}')

        # Set labels and legends for each subplot
        axes[0, i].set_xlabel('FrameNo')
        axes[1, i].set_xlabel('FrameNo')
        axes[0, i].set_ylabel('AxisAngle')
        axes[1, i].set_ylabel('Gradient of AxisAngle')
        axes[0, i].legend()
        axes[1, i].legend()
        axes[0, i].set_ylim(25, 120)
        axes[1, i].set_ylim(-10,10)

    # Adjust layout
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(output_folder, 'leaning_angle_graph.png'), dpi=500)

def frame_with_linegraph(frame, current_frame, data, fig, canvas, axL, axR):
    """Combine a video frame with the plots for PersonL and PersonR up to the current frame."""
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

