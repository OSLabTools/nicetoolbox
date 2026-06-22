![](docs/graphics/NICE_Toolbox_4.png)

# Nonverbal Interpersonal Communication Exploration Toolbox

&emsp;&emsp;&emsp;
[Project page](https://nice.is.tue.mpg.de/) &emsp;&emsp;&emsp;
[Documentation](https://nicetoolbox.readthedocs.io/en/stable/index.html) &emsp;&emsp;&emsp;
[Changelog](https://nicetoolbox.readthedocs.io/en/stable/link_changelog.html) &emsp;&emsp;&emsp;
mailto: <nicetoolbox@tue.mpg.de>

<br>

> 🚀 We are releasing a new major 0.3.0 version which includes [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) model support, [WhisperX](https://github.com/m-bain/whisperX) audio transcription, updated evaluation pipeline, connectors to [ELAN](https://archive.mpi.nl/tla/elan) and [napari](https://napari.org/), and many other improvements and fixes. Please check [the changelog](https://nicetoolbox.readthedocs.io/en/stable/link_changelog.html) for more information.

NICE Toolbox is an easy-to-use framework for exploring nonverbal human communication.
It aims to enable the investigation of observable signs that reflect the mental state
and behaviors of the individual. Additionally, these visual nonverbal cues reveal the
interpersonal dynamics between people in face-to-face conversations.

NICE combines existing computer vision **detectors** into a single, easy-to-use framework. Working from single- or multi-camera video data, it covers whole-body pose estimation, gaze tracking, movement dynamics (kinematics), gaze interaction monitoring (mutual gaze), physical proximity between dyads, emotion detection and more. For a full list, see the [components overview](https://nicetoolbox.readthedocs.io/en/stable/wikis/wiki_components.html).

The toolbox also includes a **visualizer** molule for interactively exploring outputs, an **evaluation** module that runs configurable metrics and a collection of **connectors** for importing/exporting data to third-party tools (i.e. for labelling in [ELAN](https://archive.mpi.nl/tla/elan) or [napari-deeplabcut](https://github.com/DeepLabCut/napari-deeplabcut)).

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

## Future releases

In future releases, we plan to extend the NICE Toolbox to include detectors for facial
expressions, head movements, eye closure, active speaking, emotional valence and arousal,
and micro-action recognition.

Our goal is to provide comprehensive and objective evaluations of the algorithms,
ultimately creating a practically useful toolbox for researchers analyzing human
interaction and communication.

If you are interested in collaborating with us or contributing to the project, please
reach out to us at **<nicetoolbox@tue.mpg.de>**.

## Acknowledgments

The NICE Toolbox is using the following existing tools, methods, and frameworks:
[MMPose](https://github.com/open-mmlab/mmpose/tree/main),
[MotionBERT](https://arxiv.org/abs/2210.06551),
[HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/tree/master),
[ViTPose](https://github.com/ViTAE-Transformer/ViTPose/tree/main),
[DarkPose](https://github.com/ilovepose/DarkPose/tree/master),
[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose),
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body),
[ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze),
[SPIGA](https://github.com/andresprados/SPIGA),
[WhisperX](https://github.com/m-bain/whisperx),
[CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper),
[Py-FEAT](https://py-feat.org/pages/intro.html), and
[rerun.io](https://rerun.io/).

## Authors

Aleksandr Evgrashin,
Carolin Schmitt,
Timo Lübbing,
Ashutosh Jha,
Sophie Bauer,
Gökce Ergün,
Senya Polikovsky.

All authors are with the Optics and Sensing Laboratory at Max Planck Institute for Intelligent Systems.

We thank the [MPI-IS Software Workshop](https://is.mpg.de/en/software-workshop) for their thoughtful feedback and support during the project refactoring. 

## License

[NICE Toolbox](https://github.com/OSLabTools/nicetoolbox) © 2026 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1), see [LICENSE.md](https://github.com/OSLabTools/nicetoolbox/blob/main/LICENSE.md).

Some components of the NICE Toolbox further use algorithms that are being distributed under other licenses
listed in [LICENSES_ALGORITHMS.md](https://github.com/OSLabTools/nicetoolbox/blob/main/LICENSES_ALGORITHMS.md).
