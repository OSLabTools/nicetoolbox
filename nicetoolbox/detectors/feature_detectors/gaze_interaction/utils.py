"""
Utility functions for visualizing gaze interaction data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def visualize_gaze_interaction(
    data, categories_list, output_folder, people_names=None, camera_names=None
) -> None:
    """
    Visualizes the gaze interaction results across frames for multiple
    people and cameras.

    Args:
        data (ndarray): The input data array of shape
            (distances, look_at, mutual).
        categories_list (list): The list of category names
            (i.e., ["distance_gaze_face", "gaze_look_at", "gaze_mutual"]).
        output_folder (str): The path to the output folder where the plots will be
            saved.
        people_names (list, optional): The list of names for each person.
            Defaults to None.
        camera_names (list, optional): The list of names for each camera.
            Defaults to None.

    Returns:
        None
    """
    num_people = len(data)

    for camera_idx in range(data.shape[1]):
        camera_name = camera_names[camera_idx] if camera_names is not None else "3d"
        unit = "(in pixels)" if camera_name != "3d" else "(real-world units: m/cm/mm)"
        _, axs = plt.subplots(num_people, 1, figsize=(10, 15))
        # Ensure axs is a list in case num_people is 1
        if num_people == 1:
            axs = [axs]
        gaze_distance_max = 0
        for i, (ax, dat) in enumerate(zip(axs, data)):
            # create secondary axis for boolean values
            ax2 = ax.twinx()
            for j, category in enumerate(categories_list):
                # first category is gaze distance and it is continuous
                if j == 0:
                    ax.plot(
                        dat[camera_idx, :, j],
                        label=f"{category} (left axis)",
                        color="tab:blue",
                    )
                    gaze_distance_max = np.max(dat[camera_idx, :, j])
                # other categories are boolean
                if j == 1:
                    ax2.plot(
                        dat[camera_idx, :, j],
                        label=f"{category} (right axis)",
                        linestyle="--",
                        color="tab:orange",
                    )
                elif j == 2:
                    ax2.plot(
                        dat[camera_idx, :, j],
                        label=f"{category} (right axis)",
                        linestyle=":",
                        color="tab:green",
                    )

            if people_names is None:
                people_names = ["PersonL", "PersonR"]
            ax.set_title(f"Gaze Interaction Across Frames - {people_names[i]}")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel(f"Gaze Distance to the Other Person’s Face \n {unit}")
            ax2.set_ylabel("True/False (1/0)")

            # Set specific ticks for axis
            ax.set_ylim(-0.1, gaze_distance_max + 0.2)
            yticks = np.arange(0, gaze_distance_max + 0.2, 0.2)
            ax.set_yticks(yticks)
            ax2.set_yticks([0, 1])
            ax2.set_ylim(-0.1, 1.1)
            # Add legends from both axes
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(
                lines + lines2,
                labels + labels2,
                loc="upper left",
                bbox_to_anchor=(1.03, 1),
            )

        camera_name = (
            camera_names[camera_idx] if camera_names is not None else "camera_3d"
        )
        # Save the plot
        plt.subplots_adjust(right=0.85)
        plt.savefig(
            os.path.join(output_folder, f"gaze_interaction_{camera_name}.png"),
            bbox_inches="tight",
            dpi=500,
        )
