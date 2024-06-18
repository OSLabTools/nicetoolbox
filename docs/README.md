# Welcome!

#TODO: UPDATE THE TOOL NAME

ISA-Tool is a framework designed for the automatic human behavior analysis to study 
social rapport and communication during face-to-face human interactions. 

The goal of our ongoing project is to develop an accurate, robust, and automatic framework for 
analyzing human behavior, cognition, and interaction in a dyadic communication setting, 
using video/image data from a single camera or calibrated multi-camera setups. 

The framework is designed as a comprehensive toolbox, incorporating state-of-the-art deep learning libraries 
for pose estimation and gaze tracking of individuals. 

## System Functionalities Overview: 
TODO-- explain somewhere filtering and interpolating

### 1. Individual Analysis: 

- **Body Joint Detection**: Identifies and tracks the position of key body joints, (e.g., shoulders, 
elbows) to analyze body posture and movements.Available algorithms include MMPose implementation 
of [HRNet-w48+DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-hrnet-dark-on-coco-wholebody) 
and [ViTPose-L](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-vitpose-on-coco) algorithms. 
The algorithms in use estimate the body joints in 2D. For calibrated multi-camera setups, 
we are able to determine the body joints' 3D position.  

- **Hand Joint Detection**: Tracks the positions of hand joints (e.g., wrists, fingers) to analyze 
hand movements and gestures. Available algorithm is MMPPose implementation of 
[HRNet-w48+DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-hrnet-dark-on-coco-wholebody) 
The algorithm in use estimates the hand joints in 2D. For calibrated multi-camera setups, 
we are able to determine the hand joints' 3D position. 

- **Face Landmark Detection**: Detects the position of key landmarks to analyze facial expressions and
movements. Available algorithm is MMPPose implementation of 
[HRNet-w48+DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-hrnet-dark-on-coco-wholebody) 
The algorithm in use estimates the facial landmarks in 2D. For calibrated multi-camera setups, 
we are able to determine the facial landmarks' 3D position. 

- **Gaze Direction**: Determines the direction in which the individual is looking to understand focus and attention.
Available algorithm is XGaze_3cams which is a customized version of [ETH-XGAZE](https://github.com/xucong-zhang/ETH-XGaze)

- **Forward/Backward Leaning Behavior**: Analyzes the individual's leaning posture to assess engagement
or interest. Available algorithm rule-based 'body-angle' which was developed by ourselves.

- **Kinematics**: Calculates the kinematics of body joints of the individual. 
Available algorithm is body-velocity algorithm which calculates the velocity of the body joints.

### 2. Interpersonal Interaction Analysis:

- **Gaze Interaction**: Assess the gaze patterns between individuals to evaluate the social interaction
and communication. Available algorithm is rule-based 'gaze-distance' which was developed by ourselves. 
It detects if the individual is looking the face of other individual and creates a 'mutual gaze' label
if both individuals are looking each other's face simultaneously.  

- **Proximity**: Measures the physical distance between individuals to evaluate the social interaction
between individuals. Available algorithm is rule-based 'body-distance'. The proximity score is 
calculated in 2D for each camera view, and also in 3D for calibrated multi-camera setups.  

### Future extensions of the tool will include features:
- active speaker detection
- head direction estimation
- eye closure detection
- emotional valence/arousal estimation
- attention estimation

## Installation
Please see the [Installation Instructions](installation.md)  for more information 

## Visualizations

ISA-Visual is a tool developed for visualization of ISA-Tool results. 
ISA-Visual is based on [rerun.io](https://rerun.io/). 
For more details see https://gitlab.tuebingen.mpg.de/gergn/isa_visual


## Code Contributors 
ISA-Tool code was developed by ... 

## Licence 
ISA-Tool is licenced under .... 
