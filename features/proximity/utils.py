import os
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import oslab_utils.logging_utils as log_ut

def visualize_proximity_score(data, output_folder, keypoint):


    # Plot the distances for the average coordinates of the selected keypoints across all frames
    plt.plot(data)
    plt.xlabel('Frame Index')
    plt.ylabel('Proximity Score')
    if len(keypoint) == 1:
        title = f'Distance between {keypoint[0]} in PersonL and PersonR'
    else:
        title = f'Distance between center of selected keypoints{keypoint} in PersonL and PersonR'
    plt.title(title)
    # Save the plot
    plt.savefig(os.path.join(output_folder, f'proximity_score_{keypoint}.png'), dpi=500) # ToDo correct title
    plt.show()