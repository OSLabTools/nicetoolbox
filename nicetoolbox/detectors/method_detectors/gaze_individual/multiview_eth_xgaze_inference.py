"""
Run gaze detection on the provided data.
"""

import sys

import toml
from multiview_eth_xgaze import inference


def main(config):
    """
    Run multiview-eth-xgaze gaze detection on the provided data.

    The function uses the 'multiview-eth-xgaze gaze' library to estimate gaze vectors for each
    camera. The estimated gaze vectors are then converted to pitch and yaw angles
    using a simple linear transformation. The resulting angles are saved in a .npz
    file with the following structure:
        - 3d: Numpy array of shape (n_frames, n_subjects, 3)
        - data_description: A dictionary containing the description of the data.

    Args:
        config (dict): The configuration dictionary containing parameters for gaze
            detection.
    """
    inference.main(config)


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = toml.load(config_path)
    main(config)
