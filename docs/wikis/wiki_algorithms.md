# Algorithms

- [HrNet & VitPose](#hrnet--vitpose)
- [Multiview_eth_xgaze](#multiview_eth_xgaze)
- [Py-FEAT](#pyfeat)
- [SPIGA](#spiga)

<br>


## HrNet & VitPose

COMING SOON...



## Multiview_eth_xgaze

COMING SOON...


## Py-FEAT (Facial Expression Analysis Toolbox)

Py-FEAT includes a variety of **pre-trained models** for **face detection, facial landmark tracking, action unit (AU) recognition, emotion detection, and identity verification**. These models enable automated facial expression analysis. In the NICE toolbox, we only use *face detection, action unit (AU) recognition, emotion detection, and identity verification**. The associated algorithms are listed below.

### Face Detection & Pose Estimation
- **img2pose**: A one-shot model for simultaneous **face detection** and **6DoF head pose estimation**.  
  [Albiero et al., 2020](https://arxiv.org/pdf/2012.07791v2)

### Action Unit (AU) Detection
- **xgb** (**default**): An **XGBoost classifier** trained on multiple facial expression datasets (BP4D, DISFA, CK+, etc.). It provides **continuous AU probabilities**, except for AU07, which is optimized for **binary detection**.

### Emotion Detection
- **resmasknet** (**default**): A **deep learning model** trained for facial expression recognition using a **Residual Masking Network**.  
  [Pham et. al., 2020](https://ieeexplore.ieee.org/document/9411919)

### Identity Detection
- **facenet**: A **face recognition model** based on **Inception-ResNet (V1)**, pretrained on **VGGFace2 and CASIA-Webface**.  
  [Schroff et al., 2015](https://arxiv.org/abs/1503.03832)

## SPIGA (Shape Preserving Facial Landmarks with Graph Attention)

**SPIGA** is a **state-of-the-art face alignment and head pose estimation model** that combines **CNNs and Graph Neural Networks (GNNs)** to predict stable facial landmarks under challenging conditions (e.g., occlusions, expressions, pose). In the NICE toolbox, SPIGA is used for **landmark localization** and **6DoF head pose estimation**. It operates on multi-camera image sequences and produces vectorized nose origin and orientation data for each visible subject.

### Landmark Localization & Head Pose Estimation
- **SPIGA** (**default**): A hybrid **CNN-GNN** model trained for **dense face alignment** and **pose estimation**, achieving top performance on WFLW, COFW, and 300W benchmarks.  
  [Prados-Torreblanca et al., 2022](https://arxiv.org/abs/2210.07233)

SPIGA uses **InsightFace** for face detection, then applies its GNN-powered inference module to extract facial landmarks and head orientation vectors. Outputs include annotated images (if enabled) and compressed `.npz` files containing head pose vectors for each camera-subject-frame triplet.



