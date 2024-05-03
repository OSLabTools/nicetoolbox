import os
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def visualize_proximity_score(data, output_folder, keypoint):
    plt.clf()
    plt.figure(figsize=(10, 5))
    # Plot the distances for the average coordinates of the selected keypoints across all frames
    plt.plot(data)
    plt.xlabel('Frame Index')
    plt.ylabel('Proximity Score')
    if len(keypoint) == 1:
        title = f'Distance between {keypoint[0]} in PersonL and PersonR'
    else:
        title = f'Distance between center of selected keypoints {keypoint} in PersonL and PersonR'
    plt.title(title)
    plt.ylim(1.3, 1.9)
    # Save the plot
    plt.savefig(os.path.join(output_folder, f'proximity_score_{keypoint}.png'), dpi=500)
    plt.show()

def frame_with_linegraph(frame, data, current_frame, global_min, global_max):
    """Combine a video frame with the plots for PersonL and PersonR up to the current frame."""

    if len(data)!=1:
        logging.error("The data shape is wrong. Data should be given as into a list")

    data = data[0]
    fig, ax = plt.subplots(1, 1, figsize=(20, 4))
    fig.patch.set_facecolor('black')

    colors = ['#98FB98', '#FFB347', '#DDA0DD', '#ADD8E6']
    ax.plot(data[:current_frame], label="proximity", color=colors[0])

    ax.set_title('proximity', color='white')
    ax.set_facecolor('black')

    ax.set_xlim(0, data.shape[0])
    ax.set_ylim(global_min, global_max)
    ax.set_xticks(range(data.shape[0] + 1))
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
    canvas.draw()
    graph_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)


    # Resize the graph_img to match the width of the frame
    graph_img = cv2.resize(graph_img, (frame.shape[1], graph_img.shape[0]))

    combined_img = cv2.vconcat([frame, graph_img])
    return combined_img