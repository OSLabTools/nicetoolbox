import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import oslab_utils.logging_utils as log_ut

PART_MAPPING = {
    "body": {'color': '#AEC6CF', 'size': 12, 'indices': list(range(0, 17))}, #pastel Blue
    "foot": {'color': '#00008B', 'size': 8, 'indices': list(range(17, 23))}, #dark blue
    "face": {'color': '#FFFACD', 'size': 6, 'indices': list(range(23, 92))}, #pastelyellow
    "Lhand": {'color': '#77DD77', 'size': 8, 'indices': list(range(92, 113))}, #pastel green
    "Rhand": {'color': '#B19CD9', 'size': 8, 'indices': list(range(113, 134))} #pastel lavender
}

def visualize_sum_of_motion_magnitude_by_bodypart(data, bodyparts_list, output_folder):
    """
       Visualizes the sum of movements by body part across frames for each person.

    Parameters
    ----------
    data: list
    List of numpy arrays. Each array represents data for a person. Rows are frames and columns are body parts.
       """
    # Number of people (numpy arrays) in the list
    num_people = len(data)

    # Names of people for each dataset in the data list
    people_names = ["PersonL", "PersonR"]  #TODO-hardcoded

    fig, axs = plt.subplots(num_people, 1, figsize=(10, 15))

    # Ensure axs is a list in case num_people is 1
    if num_people == 1:
        axs = [axs]

    # Iterate through the data list and the array of subplots to fill in data
    for i, (ax, data) in enumerate(zip(axs, data)):
        for j, body_part in enumerate(bodyparts_list):
            ax.plot(data[:, j], label=body_part)

        ax.set_title(f'Sum of Movements by Body Part Across Frames ({people_names[i]})')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Sum of Movements')
        ax.legend()

    # Save the plot
    plt.savefig(os.path.join(output_folder, 'sum_of_motion_magnitude_by_bodypart.png'), dpi=500)


def frame_with_linegraph(frame, data, categories, current_frame):
    """Combine a video frame with the plots for PersonL and PersonR up to the current frame."""
    num_person = len(data)
    log_ut.assert_and_log(len(data) == 2, f"The data shape is wrong. Data should be given as a list [dataL, dataR]")
    dataL = data[0]
    dataR = data[1]

    fig, (axL, axR) = plt.subplots(1, num_person, figsize=(10, 4))

    # Set black background
    fig.patch.set_facecolor('black')
    axL.set_facecolor('black')
    axR.set_facecolor('black')

    colors = ['#98FB98', '#FFB347', '#DDA0DD', '#ADD8E6']
    # '#ADD8E6', '#FFB6C1', '#FFD700'
    linesL = []
    # Plot data for PersonL up to the current frame
    for j, category in enumerate(categories):
        line, = axL.plot(dataL[:current_frame + 1, j], label=category, color = colors[j])
        linesL.append(line)
    axL.set_title('PersonL', color='white')


    # Plot data for PersonR up to the current frame
    for j, category in enumerate(categories):
        axR.plot(dataR[:current_frame + 1, j], label=category, color = colors[j])
    axR.set_title('PersonR', color='white')
    axR.legend()


    # Determine global min and max for y-axis
    global_min = min(dataL.min(), dataR.min())
    global_max = max(dataL.max(), dataR.max())

    # Set the same y-axis limits for both subplots
    axL.set_ylim(global_min, global_max)
    axR.set_ylim(global_min, global_max)

    # Adjusting x axis
    for ax in [axL, axR]:
        ax.set_xlim(0, dataL.shape[0])
        ax.set_xticks(range(dataL.shape[0]+1))
        ax.tick_params(axis='both', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.subplots_adjust(bottom=0.20)

    # Placing a single legend for both graphs
    columns = len(categories) // 3 + (len(categories) % 3 > 0)
    leg  = plt.legend(linesL, categories, loc='upper center', bbox_to_anchor=(1.0, 0.8), ncol=columns,
               facecolor='black')
    for text in leg.get_texts():
        text.set_color("white")


    # Convert the Matplotlib figure to an image
    canvas = FigureCanvas(fig)
    canvas.draw()
    graph_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #graph_img = cv2.resize(graph_img, (frame.shape[1], frame.shape[0]))

    plt.close(fig)

    # Combine the frame with the graph image
    combined_img = cv2.vconcat([frame, graph_img])

    return combined_img


def create_video_evolving_linegraphs(frames_data_list, data, categories, output_folder):
    # Read frames
    frames = [cv2.imread(f) for f in frames_data_list]

    # Generate a combined image for the first frame to get dimensions
    sample_combined_img = frame_with_linegraph(frames[0], data, categories, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4 format
    output_path = os.path.join(output_folder, 'movement_score_on_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 1.0, (sample_combined_img.shape[1], sample_combined_img.shape[0]))

    # Now write the combined images to the video
    for i, frame in enumerate(frames):
        if i == 0:  #because there is no movement score for first frame
            out.write(frame)
        else:
            combined = frame_with_linegraph(frame, data, categories, i-1) # i-1 bec movement score data starts from 2nd frame
            out.write(combined)
    out.release()


def create_motion_heatmap(frames_data_list, data, output_folder):
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
