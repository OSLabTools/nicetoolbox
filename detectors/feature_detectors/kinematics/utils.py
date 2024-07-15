"""
Utility functions for visualizing motion data (kinematics component).
"""

import os
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

PART_MAPPING = {
    "body": {'color': '#AEC6CF', 'size': 12, 'indices': list(range(0, 17))}, #pastel Blue
    "foot": {'color': '#00008B', 'size': 8, 'indices': list(range(17, 23))}, #dark blue
    "face": {'color': '#FFFACD', 'size': 6, 'indices': list(range(23, 92))}, #pastelyellow
    "Lhand": {'color': '#77DD77', 'size': 8, 'indices': list(range(92, 113))}, #pastel green
    "Rhand": {'color': '#B19CD9', 'size': 8, 'indices': list(range(113, 134))} #pastel lavender
}

def visualize_mean_of_motion_magnitude_by_bodypart(data, bodyparts_list, 
        global_min, global_max, output_folder, people_names=["PersonL", "PersonR"], camera_names=None) -> None:
    """
    Visualizes the mean of motion magnitude by body part across frames for multiple 
    people and cameras.

    Args:
        data (ndarray): The input data array of shape 
            (#persons, #cameras, #frames, #bodyparts(3)).
        bodyparts_list (list): The list of body parts to visualize.
        global_min (float): The global minimum value for the y-axis.
        global_max (float): The global maximum value for the y-axis.
        output_folder (str): The path to the output folder where the plots will be saved.
        people_names (list, optional): The list of names for each person. 
            Defaults to ["PersonL", "PersonR"].
        camera_names (list, optional): The list of names for each camera. 
            Defaults to None.

    Returns:
        None
    """
    # data.shape = (#persons, #cameras, #frames, #bodyparts(3))

    # Number of people (numpy arrays) in the list
    num_people = len(data)

    for camera_idx in range(data.shape[1]):
        fig, axs = plt.subplots(num_people, 1, figsize=(10, 15))

        # Ensure axs is a list in case num_people is 1
        if num_people == 1:
            axs = [axs]

        delta = (global_max - global_min) * 0.025
        # Iterate through the data list and the array of subplots to fill in data
        for i, (ax, dat) in enumerate(zip(axs, data)):
            for j, body_part in enumerate(bodyparts_list):
                ax.plot(dat[camera_idx, :, j], label=body_part)

            ax.set_title(f'Mean of Movements by Body Part Across Frames ({people_names[i]})')
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Mean of Movements')
            # ax.set_ylim(global_min - delta, global_max + delta)
            ax.legend()

        camera_name = camera_names[camera_idx] if camera_names is not None else f"camera_{camera_idx}"
        # Save the plot
        plt.savefig(os.path.join(output_folder, f'mean_of_motion_by_bodypart_{camera_name}.png'), dpi=500)


def frame_with_linegraph(frame, data, categories, current_frame, global_min, global_max):
    """
    Combines a video frame with the plots for PersonL and PersonR up to the current frame.

    This function takes a video frame and data for two people, and combines the frame with line 
    graphs of the data up to the current frame.

    Args:
        frame (numpy.ndarray): The video frame to which the line graphs will be added.
        data (list of numpy.ndarray): The data to be plotted. Each array represents data for a
            person.
        categories (list of str): The categories for the data. Each category corresponds to a 
            line on the graph.
        current_frame (int): The current frame number. Only data up to this frame will be plotted.
        global_min (float): The minimum value across all data. Used to set the y-axis limit.
        global_max (float): The maximum value across all data. Used to set the y-axis limit.

    Returns:
        combined_img (numpy.ndarray): The video frame combined with the line graphs.
    """
    if len(data) != 2:
        logging.error(f"The data shape is wrong. Data should be given as a list [dataL, dataR]")
    dataL, dataR = data

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('black')

    colors = ['#98FB98', '#FFB347', '#DDA0DD', '#ADD8E6']

    for ax, d, title in zip([axL, axR], [dataL, dataR], ['PersonL', 'PersonR']):
        for j, category in enumerate(categories):
            ax.plot(d[:current_frame + 1, j], label=category, color=colors[j])
        ax.set_title(title, color='white')
        ax.set_facecolor('black')

    columns = len(categories) // 3 + (len(categories) % 3 > 0)

    for ax in [axL, axR]:
        ax.set_xlim(0, dataL.shape[0])
        # ax.set_ylim(global_min, global_max)
        ax.set_xticks(range(dataL.shape[0] + 1))
        ax.tick_params(axis='both', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.subplots_adjust(bottom=0.20)

    leg = plt.legend(loc='upper center', bbox_to_anchor=(1.0, 0.8), ncol=columns, facecolor='black')
    for text in leg.get_texts():
        text.set_color("white")

    canvas = FigureCanvas(fig)
    canvas.draw()
    graph_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    combined_img = cv2.vconcat([frame, graph_img])
    return combined_img


def create_video_evolving_linegraphs(
        frames_data_list, data, categories, global_min, global_max,
        output_folder, file_name=None, video_fps=30.0):
    """
    Creates a video with evolving line graphs for each frame.

    This function takes a list of frames and data for two people, and creates a video where each 
    frame is combined with line graphs of the data up to that frame. The line graphs are 
    color-coded based on categories.

    Args:
        frames_data_list (list of str): The list of paths to the frames to be included in the 
            video.
        data (list of numpy.ndarray): The data to be plotted. Each array represents data for a 
            person.
        categories (list of str): The categories for the data. Each category corresponds to a 
            line on the graph.
        global_min (float): The minimum value across all data. Used to set the y-axis limit.
        global_max (float): The maximum value across all data. Used to set the y-axis limit.
        output_folder (str): The path to the folder where the video will be saved.
        file_name (str, optional): The name of the output video file. If not provided, defaults 
            to 'movement_score_on_video'.
        video_fps (float, optional): The frames per second of the output video. 
            Defaults to 30.0.

    Returns:
        None
    """
    # Get a sample image to determine video dimensions
    sample_frame = cv2.imread(frames_data_list[0])
    sample_combined_img = frame_with_linegraph(sample_frame, data, categories, 0, global_min, global_max)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4 format
    if file_name is None:
        output_path = os.path.join(output_folder, 'movement_score_on_video.mp4')
    else:
        output_path = os.path.join(output_folder, f'{file_name}.mp4')

    out = cv2.VideoWriter(output_path, fourcc, video_fps, (sample_combined_img.shape[1], sample_combined_img.shape[0]))

    for i, frame_path in enumerate(frames_data_list):
        frame = cv2.imread(frame_path)
        if i % 100 == 0:
            logging.info(f"Image ind: {i}")
        if i == 0:  # because there is no movement score for the first frame
            out.write(frame)
        else:
            combined = frame_with_linegraph(frame, data, categories, i-1, global_min, global_max)  # i-1 because movement score data starts from the 2nd frame
            out.write(combined)
    out.release()


def create_motion_heatmap(frames_data_list, data, output_folder):
    """
    Creates a motion heatmap video from a list of frames and corresponding motion data.

    This function reads a list of frames and corresponding motion data, and creates a heatmap for 
    each frame based on the motion data. The heatmaps are then overlaid on the original frames
    with a certain transparency to create a motion heatmap video. The video is saved to the 
    specified output folder.

    Args:
        frames_data_list (list of str): The list of paths to the frames to be included in the 
            video.
        data (list of numpy.ndarray): The motion data to be plotted. Each array represents motion 
            data for a frame.
        output_folder (str): The path to the folder where the video will be saved.

    Returns:
        None
    """
    # Read frames
    frames = [cv2.imread(f) for f in frames_data_list]
    # Assuming motion_data is a list where each item is a numpy array of shape (height, width, 3)
    # representing the dx, dy, dz for each point in the corresponding frame
    # This data structure might look like: [frame1_data, frame2_data, ...]
    heatmapped_frames = []

    for frame, movement in zip(frames[1:], data):

        # Normalize the motion to range [0, 1] for heatmap conversion
        normalized_movement = cv2.normalize(movement, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Convert to a heatmap (using COLORMAP_JET here, but OpenCV offers many colormaps)
        heatmap = cv2.applyColorMap(normalized_movement, cv2.COLORMAP_JET)

        # Overlay heatmap onto original frame (adjust alpha for transparency)
        alpha = 0.6
        overlaid = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        heatmapped_frames.append(overlaid)

    # Write to a video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_folder, 'movement_heatmap_on_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 1.0, (frames[0].shape[1], frames[0].shape[0]))

    for frame in heatmapped_frames:
        out.write(frame)

    out.release()
