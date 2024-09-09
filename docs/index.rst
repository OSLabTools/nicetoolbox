
.. toctree::
   :hidden:

   Project Overview <self>
   Installation <installation>
   Getting started <getting_started>
   tutorials/index
   wikis/index
   Detectors API <_autosummary/detectors>
   Evaluation API <_autosummary/evaluation>
   Visualization API <_autosummary/visual>


Welcome to NICE Toolbox's documentation!
========================================


IMAGE


Non-Verbal Interpersonal Communication Exploration Toolbox
----------------------------------------------------------

**Gökce Ergün, Timo Lübbing, Senya Polikovsky, Carolin Schmitt**

Project page: `https://nice.is.tue.mpg.de/ <https://nice.is.tue.mpg.de/>`__


NICE Toolbox is an easy-to-use framework for exploring nonverbal human communication. 
It aims to enable the investigation of observable signs that reflect the mental state 
and behaviors of the individual. Additionally, these visual nonverbal cues reveal the 
interpersonal dynamics between people in face-to-face conversations.

NICE Toolbox incorporates a growing set of Computer Vision algorithms to track and 
identify important visual components of nonverbal communication. Existing deep-learning 
and rule-based algorithms are combined into a single, easy-to-use software toolbox. 
Based on single- or multi-camera video data, the initial release encompasses whole-body 
pose estimation and gaze tracking for each individual, as well as movement dynamics 
calculation (kinematics), gaze interaction monitoring (mutual-gaze), and the measurement 
of physical body distance between dyads. 
This first set of components and algorithms is going to be extended in future releases.
For more details, please see the :doc:`components overview <wikis/wiki_components>` page in 
the wiki.

The toolbox  also includes a visualizer module, which allows users to
visualize and investigate the algorithm’s outputs. 


Installation & getting started
------------------------------

For instructions on installing the toolbox on a Linux or Windows machine, please see the 
:doc:`installation instructions <installation.md>` page.
For a quick start into the toolbox, we provide an example dataset and documentation to
set it up on the :doc:`getting started <getting_started.md>` page.
Further tutorials and documentation can be found on the :doc:`tutorials <tutorials/index>` 
and :doc:`wiki <wikis/index>` pages.



Future releases
---------------

In future releases, we plan to extend the NICE Toolbox to include detectors for facial 
expressions, head movements, eye closure, active speaking, emotional valence and arousal, 
and micro-action recognition.


Further, we will move beyond mere visual inspection and integrate a versatile 
evaluation framework. Based on our experience in computer vision, we are aware that no 
single algorithm can perform flawlessly across all capture settings.
To support you to choose the best algorithms for your settings, we are developing an 
evaluation workflow that better elucidates the limitations of the algorithms, that allows 
for systematic comparisons of the algorithms, and that assess their accuracy within a 
given setting.
Our goal is to provide comprehensive and objective evaluations of the algorithms, 
ultimately creating a practically useful toolbox for researchers analyzing human 
interaction and communication.

If you are interested in collaborating with us or contributing to the project, please 
reach out to us at **nicetoolbox@tue.mpg.de**.


Acknowledgments
---------------

The NICE Toolbox is using the following existing tools, methods, and frameworks:
`MMPose <https://github.com/open-mmlab/mmpose/tree/main>`__,
`HigherHRNet <https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/tree/master>`__,
`ViTPose <https://github.com/ViTAE-Transformer/ViTPose/tree/main>`__,
`DarkPose <https://github.com/ilovepose/DarkPose/tree/master>`__,
`ETH-XGaze <https://github.com/xucong-zhang/ETH-XGaze>`__, and
`rerun.io <https://rerun.io/>`__.


License
-------

NICE Toolbox is licensed under ….

List of Licenses and Links to the 3rd party tools use in NICE toolbox

+-------------+-----------------+--------------------------------------------------------------------------------+
| 3rd         | License         | Link                                                                           |
| Party       | Type            |                                                                                |
| Name        |                 |                                                                                |
+=============+=================+================================================================================+
| MMPose      | Apache 2.0      | https://github.com/open-mmlab/mmpose/blob/main/LICENSE                         |
+-------------+-----------------+--------------------------------------------------------------------------------+
| HigherHRNet | MIT             | https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/LICENSE |
+-------------+-----------------+--------------------------------------------------------------------------------+
| ViTPose     | Apache 2.0      | https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE                 |
+-------------+-----------------+--------------------------------------------------------------------------------+
| DarkPose    | Apache 2.0      | https://github.com/ilovepose/DarkPose/blob/master/LICENSE                      |
+-------------+-----------------+--------------------------------------------------------------------------------+
| ETH-XGaze   | CC BY-NC-SA 4.0 | https://creativecommons.org/licenses/by-nc-sa/4.0/                             |
+-------------+-----------------+--------------------------------------------------------------------------------+
| rerun.io    | MIT & Apache 2.0| https://rerun.io/docs/reference/about                                          |
+-------------+-----------------+--------------------------------------------------------------------------------+
