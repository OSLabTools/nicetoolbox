![](docs/graphics/nicetoolbox_480p.mp4)

# Non-Verbal Interpersonal Communication Exploration Toolbox

&emsp;&emsp;&emsp;
[Project page](https://nice.is.tue.mpg.de/) &emsp;&emsp;&emsp;
[Documentation](https://nicetoolbox.readthedocs.io/en/stable/index.html) &emsp;&emsp;&emsp;
mailto: <nicetoolbox@tue.mpg.de>

<br>

> 🎉 We are releasing version 0.2.0, which includes code refactoring, easier installation, and new detectors for emotion individuals and head orientation.

NICE Toolbox is an easy-to-use framework for exploring nonverbal human communication.
It aims to enable the investigation of observable signs that reflect the mental state
and behaviors of the individual. Additionally, these visual nonverbal cues reveal the
interpersonal dynamics between people in face-to-face conversations.

NICE Toolbox incorporates a growing set of Computer Vision algorithms to track and
identify important visual components of nonverbal communication. Existing deep-learning
and rule-based algorithms are combined into a single, easy-to-use software toolbox.
Based on single- or multi-camera video data, the initial release encompasses whole-body
pose estimation and gaze tracking for each individual, as well as movement dynamics
calculation (kinematics), gaze interaction monitoring (mutual-gaze), the measurement
of physical body distance between dyads, and emotion detection.
This first set of components and algorithms is going to be extended in future releases.
For more details, please see the
[components overview](https://nicetoolbox.readthedocs.io/en/stable/wikis/wiki_components.html)
page in the wiki.

The toolbox  also includes a visualizer module, which allows users to
visualize and investigate the algorithm’s outputs.

## Installation & getting started

For instructions on installing the toolbox on a Linux or Windows machine, please see the
[installation instructions](https://nicetoolbox.readthedocs.io/en/stable/installation.html)
page. For a quick start into the toolbox, we provide an example dataset and documentation to
set it up on the [getting started](https://nicetoolbox.readthedocs.io/en/stable/getting_started.html)
page. Further tutorials and documentation can be found on the
[tutorials](https://nicetoolbox.readthedocs.io/en/stable/tutorials/index.html) and
[wiki](https://nicetoolbox.readthedocs.io/en/stable/wikis/index.html) pages. You can also
access this [documentation](https://nicetoolbox.readthedocs.io/en/stable/index.html) offline
by downloading it as a PDF. Just use the ReadTheDocs pop-up menu located in the bottom right
corner of the screen.

## Code structure

![toolbox_design.png](docs/graphics/toolbox_design.png)

## Future releases

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
reach out to us at **<nicetoolbox@tue.mpg.de>**.

## Acknowledgments

The NICE Toolbox is using the following existing tools, methods, and frameworks:
[MMPose](https://github.com/open-mmlab/mmpose/tree/main),
[HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/tree/master),
[ViTPose](https://github.com/ViTAE-Transformer/ViTPose/tree/main),
[DarkPose](https://github.com/ilovepose/DarkPose/tree/master),
[ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze),
[SPIGA](https://github.com/andresprados/SPIGA),
[Py-FEAT](https://py-feat.org/pages/intro.html), and
[rerun.io](https://rerun.io/).

## Authors

Carolin Schmitt,
Gökce Ergün,
Timo Lübbing,
Ashutosh Jha,
Senya Polikovsky,
Aleksandr Evgrashin

All authors are with the Optics and Sensing Laboratory at Max-Planck Insitute for Intelligent Systems.

## License

[NICE Toolbox](https://github.com/OSLabTools/nicetoolbox) © 2025 by Carolin Schmitt, Gökce Ergün,
Timo Lübbing, Ashutosh Jha, Senya Polikovsky, Aleksandr Evgrashin is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1), see [LICENSE.md](https://github.com/OSLabTools/nicetoolbox/blob/main/LICENSE.md).

Some components of the NICE Toolbox further use algorithms that are being distributed under other licenses
listed in [LICENSES_ALGORITHMS.md](https://github.com/OSLabTools/nicetoolbox/blob/main/LICENSES_ALGORITHMS.md).

## Copyright

Copyright 2025, Max Planck Society / Optics and Sensing Laboratory - Max Planck Institute for Intelligent Systems
