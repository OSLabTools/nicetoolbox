# Algorithms

- [HrNet & VitPose](#hrnet--vitpose)
- [Multiview_eth_xgaze](#multiview_eth_xgaze)
- [Py-FEAT] (#pyfeat)

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



