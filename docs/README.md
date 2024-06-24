# Project Overview 

#TODO: UPDATE THE TOOL/Project NAME
#TODO: Perhaps add a paragraph about non-verbal communication


XXX is an ongoing project aims to develop a comprehensive and roboust framework for exploring nonverbal human communication, which enables the investigation of nonverbal cues and observable (behavioral) signs that reflect emotional and cognitive state of the individual, as well as the interpersonal dynamics between people in relatively fixed positions(?).

The toolbox incorporate a set of deep-learning- and rule-based algorithms to track and identify potentially important non-verbal visual components/aspects. The initial release of the toolbox includes whole-body pose estimation and gaze tracking for each individual. It also encompasses forward and backward leaning detection, movement dynamics calculation (kinematics), gaze interaction monitoring (mutual-gaze), and measurement of physical body distance between dyads using video data from a single camera or calibrated multi-camera setups. For more details see [Components Overview](#components-overview) Section. 

The toolbox includes a visualizer module, which allows users to visualize and investigate the algorithm's outputs. For more details see XXX-Visual ...


#### Next Steps for Project XXX:

1. Extention of toolbox with new components. 

In the future releases, we aim to extend the toolbox by adding new components, such as: 

- recognizing head shake and nod.  
- tracking head direction.   
- detection of active speaker 
- eye closure detection 
- emotional valence/arousal estimation
- attention estimation module incorporating the gaze 

##Todo: user-feature request 

2. Integrating an evaluation framework

Based on our extensive experience in computer vision, we are aware that no single algorithm can perform flawlessly across all capture settings. 
Implementing automated algorithms or procedures has the potential to enhance current workflows and enable the analysis of previously unexplored aspects, yet they also carry error rates and patterns that require careful examination. To achieve this delicate balance, another key objective of our project is to develop an evaluation workflow that better elucidates the limitations of the algorithms, allows systematic comparison of the algorithms and assess their accuracy within the current setting or across various settings. 

By moving beyond mere visual inspection, our goal is to provide a more comprehensive and objective evaluation of algorithm results, ultimately creating a useful toolbox for researchers analyzing human interaction and communication. 

If you are interested in collaborating with us or contributing to the project, please reach out to us at [contact information].(Alternative) For more details see [How-To-Contribute](). 


## Components Overview: 

### 1. Individual Components: 

- **Pose Estimation**:

    - **Body Joints**: Identifies and tracks the position of key body joints, (e.g., shoulders, elbows) to analyze body posture and movements. Available algorithms include MMPose implementation of [HRNet-w48+DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-hrnet-dark-on-coco-wholebody) and [ViTPose-L](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-vitpose-on-coco). 

    - **Hand Joint Estimation**: Tracks the positions of hand joints (e.g., wrists, fingers) to analyze hand movements and gestures. Available algorithm is MMPose implementation of [HRNet-w48+DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-hrnet-dark-on-coco-wholebody)

    - **Face Landmark Estimation**: Detects the position of key landmarks to analyze facial expressions and
    movements. Available algorithm is MMPose implementation of [HRNet-w48+DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-hrnet-dark-on-coco-wholebody) 


The algorithms estimate the position of joints in 2D. These 2D estimates are further refined during post-processing.

In post-processing, the algorithm's results are filtered using Savitzky-Golay filter. Filtering the algorithm's results helps address the well-known flickering problem in pose estimation, but this comes at the cost of potentially smoothing out small, meaningful movement changes.This filtering option is kept optional and controlled by the [frameworks.mmpose]`filtered` parameter in detectors_config.toml.If `filtered` set to true, the `window_length` and `polyorder` parameters can be used to fine-tune it. For more details see [`scipy.signal.savgol_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html#scipy-signal-savgol-filter). 


Joint estimations with a confidence score below 0.60 are marked as missing because they often indicate an occluded joint or an incorrect estimate. After replacing these likely incorrect estimations with missing values, linear interpolation is applied to the data points between the last two non-missing estimates of the joint. If the interval is greater than 1/3 of a second (#TODO: apply as 1/FPS currently hardcoded as 10 frames), the joint positions are left as empty.


With calibrated stereo cameras, the 3D positions of the body joints are determined using the triangulation method. 


- **Gaze Tracking**: Tracks the individual's gaze using XGaze_3cams algorithm which is a customized version of [ETH-XGAZE](https://github.com/xucong-zhang/ETH-XGaze) written by Xucong?? (##TODO - ask & link to Xucong). 

The algorithm can track gaze in 3D even with a single camera. When more than one camera is used, the algorithm aggregates the results from each camera that can detect gaze. Gaze direction results of the algorithm are further refined during post-processing using Savitzky-Golay filter (for more details see [`scipy.signal.savgol_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html#scipy-signal-savgol-filter))


The algorithm requires the intrinsics parameters of the used cameras. For more details and How-To Tutorials see [Calibration Section in Getting Started](getting_started.md#calibration-file) 


- **Forward/Backward Leaning**: The 'body-angle' algorithm detects the individual's forward/backward leaning movement to get more insights about the engagement of the individual. With calibrated stereo cameras, its detection is based on the angle between the 3D positions of shoulder, hip, and knee joints. Although the algorithm also calculates 'body-angle' in 2D camera views, these results should be used cautiously, as the accuracy of the algorithm highly depends on the camera-view. In 2D camera views, the algorithm would be more reliable when the camera sees the person from side. 

- **Kinematics**: The body-velocity algorithm calculates the movement dynamics of body joints by determining the per-frame velocity. With calibrated stereo cameras, it calculates 3D movement dynamics. Although the algorithm can also calculate 2D movement dynamics, these results should be used cautiously. The joints' movements in a 2D camera view may not represent the actual movements for some joints, as they highly depend on the angle from which the camera sees the person.


### 2. Interpersonal Components:

- **Gaze Interaction**: Monitors the gaze interaction between dyads (mutual-gaze) to provide more insights about the communication dynamics.'gaze-distance' algorithm detects if the individual is looking the face of other individual and detects 'mutual gaze' when both individuals are looking each other's face simultaneously.  

- **Proximity**: Measures the physical distance between dyad to to provide more insights about the communication dynamics. 'body-distance' algorithm calcultes the proximity between dyad based on the position of user-defined joint/s (controlled by [algorithms.body_distance]`used_keypoints` parameter in detectors_config.toml).  With calibrated stereo cameras, algorithm's measurement is based 3D position of the joint/s. The algorithm measures the proximity in 2D camera-views as well. For 2D scenarios, the user can define used_keypoints based on custom settings to improve accuracy and tailor the algorithm to specific needs.


## Workflow of XXX Toolbox 
#TODO: add workflow overview image

## Installation
Please see the [Installation Instructions](installation.md)  for more information 


## Acknowledgments 
ISA-Visual is based on [rerun.io](https://rerun.io/). 


## Licence 
ISA-Tool is licenced under .... 

List of Licences and Links to the 3rd party tools use in NICE toolbox

| 3rd Party Name | Licence Type     | Link                                                                           |
|----------------|------------------|--------------------------------------------------------------------------------|
| mmPOSE         | Apache 2.0       | https://github.com/open-mmlab/mmpose/blob/main/LICENSE                         |
| HRNet          | MIT              | https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/LICENSE |
| VITPose        | Apache 2.0       | https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE                 |
| DarkPose       | Apache 2.0       | https://github.com/ilovepose/DarkPose/blob/master/LICENSE                      |
| ETH-XGaze      | CC BY-NC-SA 4.0  | https://creativecommons.org/licenses/by-nc-sa/4.0/                             |
| rerun.io       | MIT & Apache 2.0 | https://rerun.io/docs/reference/about                                          |
