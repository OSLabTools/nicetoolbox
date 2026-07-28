"""
Utility functions for visualizing motion data (kinematics component).
"""

import os

import matplotlib.pyplot as plt


def visualize_mean_of_motion_magnitude_by_bodypart(
    data, bodyparts_list, output_folder, people_names=None, camera_names=None
) -> None:
    """
    Visualizes the mean of motion magnitude by body part across frames for multiple
    people and cameras.

    Args:
        data (ndarray): The input data array of shape
            (#persons, #cameras, #frames, #bodyparts(3)).
        bodyparts_list (list): The list of body parts to visualize.
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
        _, axs = plt.subplots(num_people, 1, figsize=(10, 15))

        # Ensure axs is a list in case num_people is 1
        if num_people == 1:
            axs = [axs]

        # delta = (global_max - global_min) * 0.025
        # Iterate through the data list and the array of subplots to fill in data
        for i, (ax, dat) in enumerate(zip(axs, data)):
            for j, body_part in enumerate(bodyparts_list):
                ax.plot(dat[camera_idx, :, j], label=body_part)

            if people_names is None:
                people_names = ["PersonL", "PersonR"]
            ax.set_title(f"Mean of Movements by Body Part Across Frames ({people_names[i]})")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Mean of Movements")
            # ax.set_ylim(global_min - delta, global_max + delta)
            ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1))

        camera_name = camera_names[camera_idx] if camera_names is not None else "camera_3d"
        # Save the plot
        plt.subplots_adjust(right=0.85)
        plt.savefig(
            os.path.join(output_folder, f"mean_of_motion_by_bodypart_{camera_name}.png"),
            bbox_inches="tight",
            dpi=500,
        )
