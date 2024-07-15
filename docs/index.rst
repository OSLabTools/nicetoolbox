
.. toctree::
   :hidden:

   Home page <self>
   Overview of Project <README>
   Installation <installation>
   Getting started <getting_started>
   Tutorial <tutorials>
   Detectors api <_autosummary/detectors>
   Evaluation api <_autosummary/evaluation>
   Visualization api <_autosummary/visual>


Welcome to NICE Toolbox's documentation!
========================================

**Non-Verbal Interpersonal Communication Exploration Toolbox**

#TODO: Perhaps add a paragraph about non-verbal communication

NICE Toolbox is an ongoing project aims to develop a comprehensive and
roboust framework for exploring nonverbal human communication, which
enables the investigation of nonverbal cues and observable (behavioral)
signs that reflect emotional and cognitive state of the individual, as
well as the interpersonal dynamics between people in relatively fixed
positions(?).

The toolbox incorporate a set of deep-learning- and rule-based
algorithms to track and identify potentially important non-verbal visual
components/aspects. The initial release of the toolbox includes
whole-body pose estimation and gaze tracking for each individual. It
also encompasses forward and backward leaning detection, movement
dynamics calculation (kinematics), gaze interaction monitoring
(mutual-gaze), and measurement of physical body distance between dyads
using video data from a single camera or calibrated multi-camera setups.
For more details see `Components Overview <#components-overview>`__
Section.

The toolbox includes a visualizer module, which allows users to
visualize and investigate the algorithm’s outputs. For more details see
NICE-Visual …

Next Steps for the NICE Toolbox:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Extention of toolbox with new components.

In the future releases, we aim to extend the toolbox by adding new
components, such as:

-  recognizing head shake and nod.
-  tracking head direction.
-  detection of active speaker
-  eye closure detection
-  emotional valence/arousal estimation
-  attention estimation module incorporating the gaze

##Todo: user-feature request

2. Integrating an evaluation framework

Based on our extensive experience in computer vision, we are aware that
no single algorithm can perform flawlessly across all capture settings.
Implementing automated algorithms or procedures has the potential to
enhance current workflows and enable the analysis of previously
unexplored aspects, yet they also carry error rates and patterns that
require careful examination. To achieve this delicate balance, another
key objective of our project is to develop an evaluation workflow that
better elucidates the limitations of the algorithms, allows systematic
comparison of the algorithms and assess their accuracy within the
current setting or across various settings.

By moving beyond mere visual inspection, our goal is to provide a more
comprehensive and objective evaluation of algorithm results, ultimately
creating a useful toolbox for researchers analyzing human interaction
and communication.

If you are interested in collaborating with us or contributing to the
project, please reach out to us at [contact information].(Alternative)
For more details see `How-To-Contribute <>`__.
