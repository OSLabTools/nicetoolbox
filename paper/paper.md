---
title: 'NICE Toolbox: Nonverbal Interpersonal Communication Exploration Toolbox'
tags:
  - Python
  - computer vision
  - psychology
  - interpersonal communication
authors:
  - name: Aleksandr Evgrashin
    corresponding: true
    affiliation: 1
  - name: Carolin Schmitt
    affiliation: 1
  - name: Timo Lübbing
    affiliation: 1
  - name: Ashutosh Jha
    affiliation: 1
  - name: Buket Naz Zeren
    affiliation: 1
  - name: Sophie Bauer
    affiliation: 1
  - name: Johannes Kopf-Beck
    affiliation: 2
  - name: Anton KG Marx
    affiliation: 2
  - name: Gökce Ergün
    affiliation: 1
  - name: Senya Polikovsky
    affiliation: 1
affiliations:
 - name: Max Planck Institute for Intelligent Systems, Tübingen, Germany
   index: 1
   ror: 04fq9j139
 - name: Ludwig-Maximilians-Universität München, Munich, Germany
   index: 2
   ror: 05591te55
date: 10 July 2026
bibliography: paper.bib
---
# Summary
 
Understanding human communication is a long-standing goal in psychological
research. People express themselves through many verbal and nonverbal
channels: posture, gesture, gaze, facial expression, and speech. Measurement
and annotation of these signals has traditionally required time-consuming
manual work by trained human labellers. The **Nonverbal Interpersonal
Communication Exploration (NICE) Toolbox** is a free, open-source Python
framework that automatically measures these signals by combining existing
state-of-the-art computer vision and speech-processing methods into a single
software package. It includes support for multi-camera systems,
interchangeable detection algorithms, 2D and 3D visualization of detector outputs, evaluation on
the user's custom data, and export to third-party software. With the release
of this tool, we aim to make data exploration of interpersonal communication
signals scalable, reproducible, and accessible for psychological research.


# Statement of need
 
Labelling and annotation of nonverbal behaviour signals from video recordings is one of the key methods
of psychological research on interpersonal communication. Traditionally, this
is done manually by trained student assistants or professional labellers
rewatching videos multiple times. The process is slow, hard to scale, and
inconsistent across annotators [@bakeman1997observing]. It also limits study dataset size,
due to labelling budget and time constraints. These challenges compound the broader reproducibility
concerns that have emerged in psychological research [@open2015estimating].

Computer vision and speech-processing models exist for each of the underlying
signals (body pose, gaze, facial expression, speech transcription, and others),
but they are distributed as separate research libraries with different software
dependencies, input/output formats, and configuration conventions. Combining
them into a working analysis workflow requires machine-learning and
software-engineering expertise that most psychology researchers do not have.

Even when suitable models exist, choosing among them is difficult. Recording
setups in psychological studies vary widely in camera placement, lighting,
microphone setup, participant distance, appearance, interaction style, and
data formats. Algorithm rankings on public benchmarks often do
not transfer to a specific study's setup, so the best choice of model is one
made empirically on the target data. Even within a given model, detectors
are sensitive to hyperparameters that typically need tuning per setup.
This requires an evaluation integrated with the detection pipeline, so
researchers can compare candidate models and hyperparameter configurations
on their own recordings before committing to one for their study.

`NICE Toolbox` is a Python framework for psychology researchers that unifies
data processing and model inference across a large set of machine-learning
algorithms. It uses a flexible configuration system for defining custom user
experiments, visualizations, evaluations, and integration with third-party software.
The measured signals are exported as tabular data for further statistical analysis
as part of a psychological study.

# State of the field
 
Manual annotation tools such as `ANVIL` [@kipp2001anvil] and
`ELAN` [@wittenburg2006elan] have long been the standard for behavioural coding in
psychological and linguistic research. They provide advanced user interfaces for frame-accurate
labelling and are well integrated with the workflows of the communities that
use them, but the annotation itself remains a time-consuming and
labour-intensive process.

Automated open-source signal measurement tools usually cover only one specific channel.
`Motion Energy Analysis` (MEA) [@ramseyer2020motion] quantifies frame-to-frame
pixel change as a proxy for body movement and has been widely used to study
interpersonal synchrony in psychotherapy. `OpenFace` [@Baltrusaitis2018] and
`Py-Feat` [@cheong2023py] are open-source libraries for facial analysis used in
psychological research. `CrisperWhisper` [@wagner2024] and `WhisperX` [@whisperx]
are considered the standard for audio transcription.

Commercial platforms such as `FaceReader` [@lewinski2014automated] and
`iMotions` [@imotions] provide integrated multimodal analysis with polished
user interfaces, but their proprietary licensing, closed algorithms, and
cost limit reproducibility and accessibility for academic research. Their
coverage is also bounded by what the vendor supports: signals or models not
offered by the platform cannot be added by the user.

The closest comparison to `NICE Toolbox` is `DISCOVER` [@hallmen2025discover], a
server-based back-end for computationally driven behaviour analysis. Users
drive it through its graphical front-end, the `NOVA` annotation tool
[@Heimerl2019], to explore content, inspect recordings, and refine
annotations.

Compared to these alternatives, `NICE Toolbox` focuses on automated
large-scale dataset processing without human intervention or refinement.
Each user-created configuration defines a computational pipeline in which
detector outputs from different modalities can feed into one another. The
same pipeline supports second-level measurements built on top of these
outputs (for example, kinematics from body pose, or interpersonal distance
from two-person tracking). Rather than committing to a single model,
`NICE Toolbox` provides an interchangeable collection of models with a
unified output structure. This allows users to evaluate algorithms and
choose the one that works best for their specific task. `NICE Toolbox` supports
multi-camera recordings and fuses per-camera detections into a single 3D
space through triangulation, whereas most existing tools either
accept only one camera per recording or leave per-camera outputs unfused.

# Software design

`NICE Toolbox`'s design is based on three core principles: (1) to provide a
configuration-rich experiment design that lets non-programmers run a full
multimodal analysis without writing code, (2) to maintain consistent data
structures (components) across different detectors, and (3) to reuse
established open-source models and libraries rather than reimplementing them
(including MMPose [@mmpose2020], SAM 3D Body [@sam3dbody], ETH-XGaze [@Zhang2020ETHXGaze],
`Py-Feat` [@cheong2023py], `CrisperWhisper` [@wagner2024], `WhisperX` [@whisperx],
and others).

![`NICE Toolbox` architecture. Multi-camera video, audio, and calibration files (1) are combined with user-supplied TOML configuration (2) that describes the dataset, detectors, evaluation, visualisation, and connectors. A detector pipeline (3) chains primary models (e.g., MMPose, ETH-XGaze) with second-level detectors that compute kinematics, proximity, blinks, and mutual gaze. Results are saved in tabular formats (4), exported to external annotation tools (5), evaluated across algorithms (6), and visualised (7).\label{fig:software-design}](images/software-design.png){ width=100% }

The toolbox is organised into modules (see \autoref{fig:software-design}) with separated
responsibilities: **detectors** runs the computer vision and audio models on
recordings, **evaluation** computes configurable metrics based on the
user-specific dataset, **visualizer** provides the interactive `rerun` [@RerunSDK] 3D visualizer,
and **connectors** handles import and export to third-party tools such as
`ELAN` and `napari-deeplabcut` [@napari_deeplabcut] for ground-truth annotation.

![An example layout of the interactive `NICE Toolbox` visualiser, showing detector results from a dyadic interaction: triangulated body joints, body mesh, gaze direction, head orientation, and synchronised kinematics time series. \label{fig:visualizer}](images/interactive-visualization.png){ width=100% }

Users control the software through TOML configuration files that describe
algorithm parameters, evaluation metrics, visualizer layouts (see \autoref{fig:visualizer}), and dataset
structure. Configurations are validated against schema definitions before
execution, so errors are caught before a long-running pipeline starts. To
keep configuration concise and flexible, `NICE Toolbox` supports
variables and named wildcards inside the configuration system.
Detector algorithms additionally support template inheritance, so
users can define a parameter set once and derive variants that override only
the fields that differ — for example, running the same pose estimator with
different confidence thresholds, or comparing two backbones that share the
same camera setup and I/O components.

Detectors can use the output of other detectors as their input, allowing the
construction of a directed computational graph. The order of execution is
defined by topological sorting of nodes. Nodes outputting the same components
are interchangeable and configurable by the user. For example, a user can
replace the MMPose model `body_joints` component with the SAM-3D-Body
`body_joints` component and use it as input for the kinematics detector; the
kinematics detector stays the same, but its input can be easily replaced.

Most of the detectors run in their own isolated virtual or conda environment.
The main reason is that ML libraries often pin conflicting versions of shared
dependencies (e.g., `PyTorch`, `CUDA`); isolating each detector lets us use
published models as-is without patching them into a shared dependency tree.
The main pipeline launches each detector as a subprocess in its own
environment, passes configuration through a serialized file, and reads the
detector's outputs back from disk in the unified data format.

# Research impact statement

`NICE Toolbox` has been used to extract behavioural cues for
psychological research at the Max Planck Institute for Intelligent Systems
and by external collaborators. In work led by Ludwig-Maximilians-Universität
Munich, the toolbox was used to quantify movement synchrony from
pose-estimation-based kinematic displacement measures across 32 videos of
16 parent–child dyads undergoing psychotherapy, distinguishing genuine
interactive coupling from coincidental covariation and comparing free-play
with structured-play conditions [@buaria2026synchrony]. In a separate study on
emotional and behavioural responses to microaggressions in human–AI
interaction at work, the toolbox provided multimodal behavioural
measurements (facial expression, gaze, posture) across 482 interaction
sequences recorded from six synchronised cameras, collected alongside
self-reported responses [@Singh:2026].

`NICE Toolbox` is released under the GNU Affero General Public License v3
(AGPL-3.0) and is publicly available on GitHub and Docker Hub, with documentation,
tutorials, and an example dataset.


# AI usage disclosure

Generative AI tools (Anthropic Claude Sonnet 4.6 and Opus 4.8) were used
during the development of this software and during manuscript editing.
All AI-generated code was reviewed by the original contributor and by a
second human reviewer in the merge-request process.
All AI-assisted text was reviewed and edited by the authors, who take full
responsibility for the final content.

# Acknowledgements

This work was partially supported by Max Planck & Amazon Science Hub.

We thank the MPI-IS Software Workshop for their thoughtful feedback and
support during the project development.

# References