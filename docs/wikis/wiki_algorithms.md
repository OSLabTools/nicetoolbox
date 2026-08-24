# Algorithms

```{contents} Contents
:depth: 2
```


## HRNet, ViTPose, & RTMPose (2D Pose Estimation)

The NICE toolbox integrates the **MMPose framework** to provide a suite of robust **top-down 2D human pose estimation** models. These algorithms operate by first using an object detector (e.g., Faster R-CNN) to extract bounding boxes around subjects, followed by dedicated keypoint localization on those cropped regions. The toolbox supports a range of architectures, from high-precision transformers to real-time optimized CNNs.

### High-Resolution Networks
- **HRNet (w48)**: A convolutional neural network designed to maintain high-resolution representations throughout the entire inference process. Instead of encoding images into low-resolution feature maps and decoding them later, HRNet connects high-to-low resolution subnetworks in parallel with repeated multi-scale fusion. This preserves fine spatial details and excels at localizing complex or occluded joints.   
  [Sun et al., 2019](https://arxiv.org/abs/1902.09212)

### Vision Transformers
- **ViTPose**: A state-of-the-art baseline that employs plain, non-hierarchical Vision Transformers (ViTs) as the feature extraction backbone. By leveraging global self-attention mechanisms, ViTPose efficiently models the long-range relationships between different anatomical joints, offering superior robustness against severe occlusions and challenging body variations.   
  [Xu et al., 2022](https://arxiv.org/abs/2204.12484)

### Real-Time Pose Estimation
- **RTMPose (m/l/wholebody)**: A framework explicitly optimized for real-time multi-person pose estimation. RTMPose utilizes a highly efficient CSPNeXt backbone and replaces traditional heatmap generation with a SimCC-based prediction head (treating keypoint localization as a classification task). This architectural shift delivers an exceptional balance between low latency and high accuracy.   
  [Jiang et al., 2023](https://arxiv.org/abs/2303.07399)


## MotionBERT (3D body pose lifting)

**MotionBERT** is a **2D-to-3D pose lifter** based on a Dual-stream Spatio-temporal Transformer (DSTformer), integrated through **MMPose**. It lifts per-frame 2D keypoints into **root-relative 3D poses** without requiring a second camera calibration or world-space alignment pass.

In NICE Toolbox, MotionBERT runs as the **`motionbert`** algorithm. It reads 2D keypoints and bounding boxes directly from an existing **`body_joints`** NPZ (default upstream: **`vitpose_huge`**), so subject ordering and tracking stay consistent with the 2D pipeline. The lifter outputs **17 keypoints in Human3.6M order**.

Post-processing applies optional temporal filtering and confidence masking on the 2D and 3D arrays. The upstream algorithm can be configured via **`input_detector_names`** in `detectors_config.toml`

[Zhu et al., 2023](https://arxiv.org/abs/2210.06551)

## SAM 3D Body

```{warning}
SAM 3D Body uses a gated model on Hugging Face. You must request access and set your token before running. See [Hugging Face access token](../installation.md#hugging-face-access-token).
```

**SAM 3D Body** is a whole-body 3D pose and shape estimation model that recovers 2D/3D keypoints, body mesh vertices, and body/hand/shape parameters in **MHR** (Mesh Human Recovery) parameterization. It supports single- and multi-view setups and uses a DINOv3 or ViT backbone for robust feature extraction.

In the NICE Toolbox, SAM 3D Body runs as the **`sam_3d_body`** algorithm and outputs four components: **`body_joints`** (world-space 3D, requires camera calibration), **`body_joints_local`** (camera-native 3D, always written), **`hand_joints`**, **`hand_joints_local`**, and **`body_mesh`** (mesh vertices and faces).


[Yang et al., 2026](https://arxiv.org/abs/2602.15989)

## ETH-XGaze (Gaze Estimation)

**ETH-XGaze** is a gaze estimation model trained on the large-scale [ETH-XGaze dataset](https://ait.ethz.ch/xgaze), which covers a wide range of head poses and gaze directions under controlled lighting conditions. It estimates 3D gaze vectors per subject per camera using face detection, 68-point facial landmark localization (dlib), and a deep appearance-based gaze network.

In the NICE Toolbox, ETH-XGaze runs as the **`eth_xgaze`** algorithm and outputs **`gaze_individual`** component. It processes all configured cameras independently and outputs per-camera 3D gaze vectors and pitch/yaw angles.

[Zhang et al., 2020](https://arxiv.org/abs/2007.15837)

## UniGaze (Gaze Estimation)

**UniGaze** is a generalized appearance-based gaze estimation framework. It detects faces using face alignment, normalizes the face crops, and runs a pre-trained network to estimate 3D gaze vectors and pitch/yaw gaze angles.

In the NICE Toolbox, UniGaze runs as the **`unigaze`** algorithm and outputs the **`gaze_individual`** component. It processes all configured cameras independently and outputs:
- `gaze_3d_camera_space` (camera-space 3D gaze vectors)
- `gaze_3d` (world-space 3D gaze vectors)
- `gaze_2d` (reprojected 2D pixel-plane gaze arrows)
- `landmarks_2d` (all 68 facial landmarks detected by the face-alignment model)
- `gaze_origin_2d` (midpoint between the eyes)

> [!NOTE]
> The UniGaze detector does not natively output prediction confidence scores. For schema compatibility, all predictions are assigned a constant confidence score of `1.0`.

[Qin et al., 2025](https://arxiv.org/pdf/2502.02307)

## Gaze Interaction

Two derived algorithms build on top of `gaze_individual` outputs to characterize interpersonal gaze dynamics:

- **`gaze_fusion`** (`gaze_multiview` component): aggregates per-camera ETH-XGaze estimates into a single fused gaze direction using weighted averaging, with optional temporal smoothing. Optionally, this can be configured to take `eye_closed_state` as input and use it to filter/interpolate gaze_fusion output during eye closures via `closed_eye_filter_mode` (supports `"none"`, `"nan"`, or `"interpolate"`, defaulting to `"none"`).
- **`gaze_distance`** (`gaze_interaction` component): detects mutual gaze between subjects by computing the angular distance between each subject's fused gaze vector and the direction towards the other subject. The `threshold_look_at` parameter (default `0.4`) controls the sensitivity of mutual gaze detection.

## Kinematics

**`velocity_body`** (`kinematics` component) computes frame-by-frame movement dynamics — velocity and acceleration — from body joint positions produced by an upstream `body_joints` detector (default: `hrnetw48`). No separate model weights are required.

## Proximity

**`body_distance`** (`proximity` component) estimates the physical distance between subjects based on selected body keypoints (default: `nose`). It reads from an upstream `body_joints` detector (default: `hrnetw48`) and requires camera calibration for world-space distance estimates. No separate model weights are required.

## Py-FEAT (Facial Expression Analysis Toolbox)

**Py-FEAT** is a toolkit for automated facial expression analysis. In the NICE Toolbox it is used for face detection (`img2pose`), action unit recognition (`xgb`, XGBoost-based, continuous AU probabilities), emotion detection (`resmasknet`), and subject identity verification (`facenet`).

[Cheong et al., 2023](https://doi.org/10.1007/s42761-023-00191-4)

## SPIGA (Shape Preserving Facial Landmarks with Graph Attention)

**SPIGA** is a hybrid CNN-GNN model for facial landmark localization and 6DoF head pose estimation. In the NICE Toolbox it is used for **landmark localization** and **head orientation** estimation across all configured cameras. Face detection is handled by InsightFace; outputs are head pose vectors per camera-subject-frame.

[Prados-Torreblanca et al., 2022](https://arxiv.org/abs/2210.07233)

## Eye Closure

Two algorithms support tracking blink/closure states and dynamics from facial landmark outputs:

- **`eye_closure_ear`** (`eye_closure_score` component): Computes the Eye Aspect Ratio (EAR) as the ratio of eye height to eye width, indicating closure degree. [Soukupová and Čech, 2016](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
- **`eye_closure_threshold`** (`eye_closed_state` component): Categorizes the eye state (closed/open) via a threshold applied to the closure score, with optional duration-based temporal filtering. It outputs per-camera (`per_camera_state`) and global (`global_state`) eye closed states for `"left_eye"`, `"right_eye"`, and `"both_eyes"`.


## WhisperX (Audio Transcription & Speaker Diarization)

```{warning}
WhisperX uses a gated model on Hugging Face for speaker diarization. You must request access and set your token before running. See [Hugging Face access token](../installation.md#hugging-face-access-token).
```

**WhisperX** provides fast automatic speech recognition (ASR) with word-level timestamps and speaker diarization. By integrating voice activity detection (VAD) preprocessing and forced alignment, WhisperX significantly reduces hallucinations and improves timestamp accuracy compared to the original Whisper model. In the NICE toolbox, it is utilized to process audio tracks for transcription and to systematically identify who is speaking and when.

The pipeline runs in four stages: VAD preprocessing to segment speech, transcription via a `faster-whisper` backend, forced phoneme alignment for word-level timestamps, and speaker diarization via `pyannote-audio`. Note that speaker labels cannot currently be mapped to specific subjects in the video.

[Bain et al., 2023](https://arxiv.org/abs/2303.00747)

## CrisperWhisper

**CrisperWhisper** is a fine-tuned Whisper variant focused on **verbatim** speech transcription with highly accurate **word-level timestamps**. Unlike standard Whisper models, which paraphrase and omit disfluencies, CrisperWhisper transcribes every spoken word — including fillers, false starts, and repetitions — and was retrained with an adjusted tokenizer and a custom attention-based timestamp objective to sharply localize word boundaries, even around pauses.

In the NICE Toolbox, CrisperWhisper runs as the **`crisper_whisper`** algorithm and outputs the **`audio_transcription`** component only. It processes each configured audio track independently, returning per-word chunks with start/end timestamps. A post-processing step (ported from the original repository) redistributes inter-word pauses up to a small threshold evenly between adjacent words to refine the boundaries.

The raw word chunks are then converted into the same unified `{segments, words}` structure used by WhisperX's `audio_transcription` component, so the two transcription backbones are interchangeable downstream. Note that CrisperWhisper performs neither speaker diarization nor speaker-aligned transcription.

[Wagner et al., 2024](https://arxiv.org/abs/2408.16589)