# Components

NICE Toolbox incorporates a growing set of Computer Vision algorithms to track and identify important visual components of nonverbal communication. The initial release encompasses whole-body pose estimation (body joints, hand joints, and face landmarks), gaze tracking, movement dynamics calculation (kinematics), and emotion detection for each individual. In addition, it features gaze interaction monitoring (mutual gaze) and the measurement of physical body distance (proximity) between dyads. 

This document first introduces the toolbox's output files and then details the detected components.

```{contents} Contents
:depth: 3
```




## Output files

The output of each component is saved in the corresponding component folder as an `<algorithm_name>.npz` file. Additionally, if the `save_csv` parameter is set to true in the [`./configs/detectors_run_file.toml`](../../configs/detectors_run_file.toml) file, the outputs will also be saved in the csv_files folder as separate CSV files. For both file formats, the results for different algorithms are provided separately.

### Numpy arrays
Per component, each `<algorithm>.npz` file contains several numpy arrays plus a dictionary called `data_description`.

| component | contained numpy arrays |
| - | - |
| body_joints | 2d, 2d_filtered, 2d_interpolated, bbox_2d, 3d |
| hand_joints | 2d, 2d_filtered, 2d_interpolated, bbox_2d, 3d |
| face_landmarks | 2d, 2d_filtered, 2d_interpolated, bbox_2d, 3d |
| gaze_individual | landmarks_2d, 3d, 3d_filtered, 2d_projected_from_3d_filtered, 2d_projected_from_3d |
| gaze_interaction | distance_gaze_3d, gaze_look_at_3d, gaze_mutual_3d |
| kinematics | displacement_vector_body_2d, velocity_body_2d, displacement_vector_body_3d, velocity_body_3d |
| leaning | body_angle_2d, body_angle_3d |
| proximity | body_distance_2d, body_distance_3d |
| emotion_individual | faceboxes, aus, emotions, poses |

All these numpy arrays share a common structure: the first 3 dimensions contain the subjects, cameras, and frames, the remaining dimensions vary with the respective entity.

### Data description
The `data_description` dictionary details the entries of all numpy files within on component's algorithm `.npz` file. `axis0` contains the subject descriptions, `axis1` the camera names or '3d', and `axis2` the frame numbers as a string of 5 digits. The remaining axis may take the following data:

| array name | `axis3` | `axis4` |
| - | - | - |
| 2d, 2d_filtered, 2d_interpolated | list of all joint names | coordinate_x, coordinate_y, confidence_score |
| 3d, displacement_vector_body_2d, displacement_vector_body_3d | list of all joint names | coordinate_x, coordinate_y, coordinate_z |
| 3d, 3d_filtered | coordinate_x, coordinate_y, coordinate_z | --
| 2d_projected_from_3d_filtered, 2d_projected_from_3d | coordinate_u, coordinate_v | --
| bbox_2d | full_body | top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence_score |
| landmarks_2d | list of all landmarks | coordinate_u, coordinate_v |
| distance_gaze_3d | per subject: to_face_<subject_name> | -- |
| gaze_look_at_3d | per subject: look_at_<subject_name> | -- |
| gaze_mutual_3d | per subject: with_<subject_name> | -- |
| velocity_body_2d, velocity_body_3d | list of all joint names | velocity |
| body_angle_2d, body_angle_3d | angle_deg, gradient_angle | -- |
| body_distance_2d, body_distance_3d | distance | -- |
| faceboxes | FaceRectX, FaceRectY, FaceRectWidth, FaceRectHeight, FaceScore | -- |
| aus | list of action unit IDs | -- |
| emotions | anger, disgust, fear, happiness, sadness, surprise, neutral | -- |
| poses | Pitch, Roll, Yaw | -- |


### Python code
The code snippet below shows how you can access the content of an `.npz` file in Python:

```
import numpy as np

# load the file
arr = np.load("path/to/file.npz", allow_pickle=True)

# to see all arrays and dictionaries inside
print(arr.files)

# arrays can be accessed as usual
print(arr['3d'].shape)
print(arr['3d'][:, 0])

# there is always a dictionary describing all available arrays and what to find in each their dimensions
print(arr['data_description'].item())

# array axis descriptions for array '3d':
print(arr['data_description'].item()['3d'])

```








## Body joints
Identifies and tracks the position of key body joints, (e.g., shoulders, elbows) to analyze body posture and movements. Available algorithms are *HRNet-w48* and *ViTPose*. The figure below illustrates the key body joints identified. *ViTPose* estimates full-body joints, including arms, shoulders, hips, wrists, and ankles, but excludes foot-specific joints like heels and toes. *HRNet-w48* includes these additional foot joints.

[<img src="../graphics/body_joints.png" height="400">](../graphics/body_joints.png)
[<img src="../graphics/foot_joints.png" height="200">](../graphics/foot_joints.png)


The CSV files containing the <body_joints> key and the `<output_folder>/body_joints/<algorithm_name>.npz` file represent the results of this component. 

The algorithms estimate the position of joints in 2D (x and y coordinates) along with a confidence score for each joint. The `…_2d.csv` files and `2d.npy` data is saved inside the `<output_folder>/body_joints/<algorithm_name>.npz` file represent the raw output of the algorithm. These 2D estimates are further refined during post-processing. 

The algorithm's results are smoothed in post-processing using Savitzky-Golay filter (see `…_2d_filtered.csv` or `2d_filtered.npy` file). This smoothing helps mitigate the well-known flickering issue in pose estimation but may also smooth out small, meaningful movement changes. Filtering is optional and users can deactivate or fine-tune its parameters (see `frameworks.mmpose.filtered`, `frameworks.mmpose.window_length`, and `frameworks.mmpose.polyorder`  parameters in the [`./configs/detectors_config.toml`](../../configs/detectors_config.toml) file.

Joint estimations with a confidence score below 0.60 are marked as missing because they often indicate an occluded joint or an incorrect estimate. These likely incorrect estimations are replaced with missing values, and linear interpolation is applied between the last two non-missing estimates of the joint. If the gap exceeds 1/3 of a second, the joint positions remain empty (see `…_2d_interpolated.csv` or `2d_interpolated.npy` file).

With calibrated stereo cameras, the 3D positions (x, y, and z coordinates) of the body joints are computed via the triangulation method (see `..._3d.csv` or `3d.npy` file). Since 3D estimation is performed after interpolation of the 2D estimations, any missing 2D joint point will also be missing in the 3D results.
If the user has more than two camera views, the first two camera views listed in the `frameworks.mmpose.camera_names` parameter in the [`./configs/detectors_config.toml`](../../configs/detectors_config.toml) file will be used for triangulation.

## Hand joints
Tracks the positions of hand joints to analyze hand movements and gestures. Available algorithm is *HRNet-w48*. The figure below represents the identified hand joints.

[<img src="../graphics/hand_joints.png" height="250">](../graphics/hand_joints.png)

The CSV files containing the <hand_joints> key and the `<output_folder>/hand_joints/<algorithm_name>.npz` file represent the results of this component. The post-processing steps and naming conventions are the same as those used for body joints.

## Face landmarks
Detects the position of key landmarks to analyze facial expressions and movements. Available algorithm is *HRNet-w48*. The figure below represents the identified face landmarks. 

[<img src="../graphics/face_landmarks.png" height="350">](../graphics/face_lanmarks.png)

The CSV files containing the <face_landmarks> key and the `<output_folder>/face_landmarks/<algorithm_name>.npz` file represent the results of this component. The post-processing steps and naming conventions are the same as those used for body joints.

## Gaze Individual
Tracks the individual's gaze using the *Multiview_eth_xgaze* algorithm. The CSV files containing the <gaze_individual> key and the `<output_folder>/gaze_individual/<algorithm_name>.npz` file represent the results of this component.

The algorithm first detects the eye region and then calculates the 3D gaze direction. It is capable of tracking gaze in 3D space even with a single camera. When multiple cameras are used, the algorithm aggregates gaze detection results from each camera that captures the subject's gaze.

The `…_3d.csv` file and `3d.npy` data is saved inside the `<output_folder>/gaze_individual/<algorithm_name>.npz` contains the 3D gaze direction, with the starting point derived from the position of the eye. The 2D eye region positions are stored in `…_landmarks_2d.csv` and `landmarks_2d.npy` file.

Gaze direction results of the algorithm are further smoothed during post-processing using Savitzky-Golay filter (see `…_3d_filtered.csv` or `3d_filtered.npy` file). Filtering is optional and users can deactivate or fine-tune its parameters (see `algorithms.multiview_eth_xgaze.filtered`, `algorithms.multiview_eth_xgaze.window_length`, and `algorithms.multiview_eth_xgaze.polyorder` parameters in the [`./configs/detectors_config.toml`](../../configs/detectors_config.toml) file).

## Kinematics
The *velocity-body* algorithm analyzes the movement dynamics of body joints by calculating their displacement and velocity. The CSV files containing the <kinematics> key and the `<output_folder>/kinematics/<algorithm_name>.npz` file represent the results of this component.

The displacement vectors for each body joint, calculated per camera view, are stored in the `…_displacement_vector_body_2d.csv` and `displacement_vector_body_2d.npy` data is saved inside the `<output_folder>/kinematics/<algorithm_name>.npz`. The velocity values, also computed per camera view, are stored in the `…_velocity_body_2d.csv` and `velocity_body_2d.npy` file.

When using calibrated stereo cameras, the algorithm computes 3D movement dynamics as well (see `…_displacement_vector_body_3d.csv`/`displacement_vector_body_3d.npy` and `…_velocity_body_3d.csv`/`velocity_body_3d.npy`).

## Gaze Interaction
Monitors the gaze interaction between dyads (mutual-gaze) to provide more insights into the communication dynamics. The CSV files containing the <gaze_interaction> key and the `<output_folder>/gaze_interaction/<algorithm_name>.npz` file represent the results of this component.

The *gaze-distance* algorithm measures the Euclidean distance between an individual’s gaze vector and the position of another person’s face (results are stored in `…_distance_gaze_3d.csv` and `distance_gaze_3d.npy` data is saved inside the `<output_folder>/gaze_interaction/<algorithm_name>.npz`). 

If the measured distance is below a predefined threshold, the algorithm labels the gaze as directed at the other person’s face (see `…_look_at_3d.csv` or `look_at_3d.npy` file). Additionally, the algorithm detects 'mutual gaze' when both individuals are simultaneously looking at each other's face (see `…_gaze_mutual_3d.csv` or `gaze_mutual_3d.npy` file).

## Proximity
The *body-distance* algorithm measures the physical proximity between dyads by calculating between user-defined joint/s. 

For each camera view, the algorithm computes this distance based on the 2D positions of the selected joints (see `...body_distance_2d.csv` or `body_distance_2d.npy` data is saved inside the `<output_folder>/proximity/<algorithm_name>.npz`). With calibrated stereo cameras, the algorithm's measurement is based on 3D position of the joint/s (see `...body_distance_3d.csv` or `body_distance_3d.npy` file). 

## Emotion Detection

Utilizes **Py-Feat**, an open-source facial expression analysis tool, to detect **facial landmarks, Action Units (AUs), and emotions** from images. This module detects **seven fundamental emotions**: **Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral**. By default, the **Detector** object in Py-Feat utilizes **CUDA acceleration** when available, ensuring faster **face detection, feature extraction, and emotion classification**. If a GPU is not available, processing falls back to the **CPU**.

Results are stored in `.npz` files under `<output_folder>/emotion_individual/<algorithm_name>.npz`. The output includes **Face bounding boxes** (`faceboxes`), **Action Units (AUs)** (`aus`), **Emotion scores** (`emotions`), and **Head pose estimation** (`poses`)  

### Detector Configuration
- **Batch Size (`batch_size`)**: Determines the number of images processed in each inference batch. A **higher value improves efficiency** but requires more RAM.  
- **Max Cores (`max_cores`)**: Controls the number of CPU cores used for **multiprocessing** during inference. Set to **-1** to use all available cores for maximum performance.
---
