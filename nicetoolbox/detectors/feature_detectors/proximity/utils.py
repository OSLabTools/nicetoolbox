"""
Utility functions for visualizing proximity scores.
"""

import os

import matplotlib.pyplot as plt

from nicetoolbox_core.data.array_schema import NpzArray


def visualize_proximity_score(array: NpzArray, output_folder: str, keypoint: list[str]):
    """
    Visualizes the proximity score for a given array and saves the plots as images.

    Images are created for each camera. If the number of keypoints is greater than 1,
    the proximity score is visualized for the center of the selected keypoints.

    Args:
        array (NpzArray): the proximity scores with named axes (subjects, cameras, frames,
            labels). Camera names are read from array.axes.cameras.
        output_folder (str): The path to the output folder where the images will be saved.
        keypoint (list): The name(s) of the keypoint(s) used for calculating
            proximity scores.

    Returns:
        None
    """
    for camera_idx, camera_name in enumerate(array.axes.cameras):
        unit = "(in pixels)" if camera_name != "3d" else "(real-world units: m/cm/mm)"
        plt.clf()
        plt.figure(figsize=(10, 5))
        # Plot the distances for the average coordinates of the selected keypoints
        # across all frames (subject 0; the score is symmetric between the two subjects).
        # axis3[0] = distance; axis3[1] = confidence (drop the confidence column for the plot).
        plt.plot(array.data[0, camera_idx, :, 0])
        plt.xlabel("Frame Index")
        plt.ylabel(f"Proximity Score {unit}")
        if len(keypoint) == 1:
            title = f"Distance between {keypoint[0]} across individuals"
        else:
            title = f"Distance between center of selected keypoints {keypoint} across individuals"
        plt.title(title)

        # Save the plot
        plt.savefig(os.path.join(output_folder, f"proximity_score_{keypoint}_{camera_name}.png"), dpi=500)
