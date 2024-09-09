# Components

[Individual Components](#body-joints)
- [Body joints](#body-joints)
- [Hand joints](#hand-joints)
- [Face landmarks](#face-landmarks)
- [Gaze](#gaze)
- [Forward/backward leaning](#forwardbackward-leaning)
- [Kinematics](#kinematics)

[Interpersonal Components](#gaze-interaction)
- [Gaze Interaction](#gaze-interaction)
- [Proximity](#proximity)

[Future extensions](#future-extensions)

<br>


## Body joints
Identifies and tracks the position of key body, (e.g., shoulders, elbows) to analyze body posture and movements. Available algorithms include MMPose implementation of [HRNet-w48+DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-hrnet-dark-on-coco-wholebody)
and [ViTPose-L](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-vitpose-on-coco).


Output files: 
- 2d: -x/y coordinates of body/hand joints or face landmarks & their confidence score. It saves the raw output of the algorithm 
    - shape: [..., number_of_bodyjoints, 3] last dimension: 'coordinate_x', 'coordinate_y', 'confidence_score' 
- 2d_filtered: if user set filtered true in detectors_config - applied Savitzky-Golay filter to algorithm output. 
window_length and polyorder parameters can be adjusted in detectors_config. 
    - shape - same as 2d results
- 2d_interpolated: applied a correction on algorithm output (raw/filtered) -- 
the detections with a low confidence score were removed. The missing values were interpolated if the number of consecutive missing value is below 10
    - shape - same as 2d results
- bbox_2d: coordinates of the bounding box of the full body of the subject.
    - shape: shape: [..., 1, 5] last dimension: 'top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y', 'confidence_score'
- 3d: if more than 1 camera, calculates 3d coordinates of the body joints by triangulating two camera views. 
Note: using 2d_interpolated results
    - shape: [..., number_of_bodyjoints, 3] last dimension: 'coordinate_x', 'coordinate_y', 'coordinate_z'



## Hand joints
Tracks the positions of hand joints (e.g., wrists, fingers) to analyze hand movements and gestures. 
Available algorithm is MMPose implementation of [HRNet-w48+DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-hrnet-dark-on-coco-wholebody).


## Face landmarks
Detects the position of key landmarks to analyze facial expressions and movements. 
Available algorithm is MMPose implementation of [HRNet-w48+DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-hrnet-dark-on-coco-wholebody).


## Gaze
Tracks the individual’s gaze using XGaze_3cams algorithm which is a customized version of [ETH-XGAZE](https://github.com/xucong-zhang/ETH-XGaze) written by Xucong?? (##TODO - ask & link to Xucong).


## Forward/backward leaning
The ‘body-angle’ algorithm detects the individual’s forward/backward leaning movement to get more insights about the engagement of the individual. With calibrated stereo cameras, its detection is based on the angle between the 3D positions of shoulder, hip, and knee joints. Although the algorithm also calculates ‘body-angle’ in 2D camera views, these results should be used cautiously, as the accuracy of the algorithm highly depends on the camera-view. In 2D camera views, the algorithm would be more reliable when the camera sees the person from side.


## Kinematics
The body-velocity algorithm calculates the movement dynamics of body joints by determining the per-frame velocity. With calibrated stereo cameras, it calculates 3D movement dynamics. Although the algorithm can also calculate 2D movement dynamics, these results should be used cautiously. The joints’ movements in a 2D camera view may not represent the actual movements for some joints, as they highly depend on the angle from which the camera sees the person.


## Gaze Interaction
Monitors the gaze interaction between dyads (mutual-gaze) to provide more insights about the communication dynamics.’gaze-distance’ algorithm detects if the individual is looking the face of other individual and detects ‘mutual gaze’ when both individuals are looking each other’s face simultaneously.


## Proximity
Measures the physical distance between dyad to to provide more insights about the communication dynamics. ‘body-distance’ algorithm calculates the proximity between dyad based on the position of user-defined joint/s (controlled by [algorithms.body_distance]\ ``used_keypoints`` parameter in detectors_config.toml). With calibrated stereo cameras, algorithm’s measurement is based 3D position of the joint/s. The algorithm measures the proximity in 2D camera-views as well. For 2D scenarios, the user can define used_keypoints based on custom settings to improve accuracy and tailor the algorithm to specific needs.


## Future extensions
In future releases, we aim to extend the toolbox by adding new components, such as
-  recognizing head shake and nod.
-  tracking head direction.
-  detection of active speaker
-  eye closure detection
-  emotional valence/arousal estimation
-  attention estimation module incorporating the gaze