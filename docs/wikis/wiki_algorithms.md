# Algorithms

```{contents} Contents
:depth: 2
:local:
```


## Subject Identity System

Two service algorithms assign all results to the named subjects of a sequence. They read the `subjects` rules in `dataset_properties.toml` and do not use a model. See the [subject rules wiki](wiki_subject_rules.md).

- **`video_subject_detector`** (`subject_regions` component) computes, for each subject, camera and frame, the image region in which this subject is expected. It uses the `subjects` rule of each camera, for example `vsplit`, `assign` or `bbox`. Most video algorithms take `subject_regions` as input and assign their detections to the subjects with it. The `[subject_assignment.video]` section of `detectors_config.toml` sets how detections are matched to the regions.
- **`audio_subject_detector`** (`audio_diarization` component) computes, for each audio track, the time segments in which each subject speaks. It uses the `subjects` rule of each track, for example `assign` or `audio_range`. For `audio_range`, it takes the anonymous speakers of a diarization algorithm such as [`pyannote`](#pyannote-speaker-diarization) and gives them subject names. The `[subject_assignment.audio]` section sets `min_overlap` (default `0.3`): the part of a subject's time range that a speaker must cover to get this subject's name. The transcription algorithms take its output as input to assign each word to a subject.

Keep `video_subject_detector` in the `algorithms` list of the run file for video algorithms, and `audio_subject_detector` for audio algorithms with subject names.

## Person Detection (MMDetection)

The NICE Toolbox integrates the **MMDetection framework** to detect people in each camera image. The detected person bounding boxes are the input of the top-down 2D pose estimation models below and of MotionBERT. Each detection is assigned to a subject with the `subject_regions` of the [video subject detector](#subject-identity-system).

All instances share `mmdet_template` and output the **`person_bounding_box`** component. Three instances are configured by default:

- **`rtmdet_m_person`**: RTMDet-medium with weights fine-tuned for person detection. This is the default input of the pose estimation models. [Lyu et al., 2022](https://arxiv.org/abs/2212.07784)
- **`yolox_s_person`**: YOLOX-small, a fast and modern alternative. [Ge et al., 2021](https://arxiv.org/abs/2107.08430)
- **`faster_rcnn_person`**: Faster R-CNN, a commonly used baseline. [Ren et al., 2015](https://arxiv.org/abs/1506.01497)

`det_thresh` (default `0.5`) discards boxes with a lower confidence, and `iou_thresh` (default `0.8`) discards a box if it overlaps the box of another subject by more than this value.

## HRNet, ViTPose, & RTMPose (2D Pose Estimation)

The NICE toolbox integrates the **MMPose framework** to provide a suite of robust **top-down 2D human pose estimation** models. These algorithms operate by first using a person detector (by default `rtmdet_m_person`, see [Person Detection](#person-detection-mmdetection)) to extract bounding boxes around subjects, followed by dedicated keypoint localization on those cropped regions. The toolbox supports a range of architectures, from high-precision transformers to real-time optimized CNNs.

All instances share `mmpose_2d_template` and output the **`body_joints`** component. The whole-body models also output **`hand_joints`** and **`face_landmarks`**. The configured instances are `hrnetw48`, `vitpose_large`, `vitpose_huge`, `rtmpose_l_aic`, `rtmpose_l_wholebody`, `rtmw_l_wholebody` and `rtmpose_m_mpii`.

### High-Resolution Networks
- **HRNet (w48)**: A convolutional neural network designed to maintain high-resolution representations throughout the entire inference process. Instead of encoding images into low-resolution feature maps and decoding them later, HRNet connects high-to-low resolution subnetworks in parallel with repeated multi-scale fusion. This preserves fine spatial details and excels at localizing complex or occluded joints.   
  [Sun et al., 2019](https://arxiv.org/abs/1902.09212)

### Vision Transformers
- **ViTPose**: A state-of-the-art baseline that employs plain, non-hierarchical Vision Transformers (ViTs) as the feature extraction backbone. By leveraging global self-attention mechanisms, ViTPose efficiently models the long-range relationships between different anatomical joints, offering superior robustness against severe occlusions and challenging body variations.   
  [Xu et al., 2022](https://arxiv.org/abs/2204.12484)

### Real-Time Pose Estimation
- **RTMPose (m/l/wholebody)**: A framework explicitly optimized for real-time multi-person pose estimation. RTMPose utilizes a highly efficient CSPNeXt backbone and replaces traditional heatmap generation with a SimCC-based prediction head (treating keypoint localization as a classification task). This architectural shift delivers an exceptional balance between low latency and high accuracy.   
  [Jiang et al., 2023](https://arxiv.org/abs/2303.07399)
- **RTMW (l/wholebody)**: An RTMPose-based model for whole-body pose estimation, which predicts body, hand and face keypoints together. It runs as `rtmw_l_wholebody`.   
  [Jiang et al., 2024](https://arxiv.org/abs/2407.08634)


## MotionBERT (3D body pose lifting)

**MotionBERT** is a **2D-to-3D pose lifter** based on a Dual-stream Spatio-temporal Transformer (DSTformer), integrated through **MMPose**. It lifts per-frame 2D keypoints into **root-relative 3D poses** without requiring a second camera calibration or world-space alignment pass.

In NICE Toolbox, MotionBERT runs as the **`motionbert`** algorithm. It reads 2D keypoints and bounding boxes directly from an existing **`body_joints`** NPZ (default upstream: **`vitpose_huge`**), so subject ordering and tracking stay consistent with the 2D pipeline. The lifter outputs **17 keypoints in Human3.6M order**.

Post-processing applies optional temporal filtering and confidence masking on the 2D and 3D arrays. The upstream algorithms can be changed in the `[algorithms.motionbert.inputs]` section of `detectors_config.toml`: `pose_2d` selects the 2D body joints, and `person_bboxes` selects the person bounding boxes (default: `rtmdet_m_person`).

[Zhu et al., 2023](https://arxiv.org/abs/2210.06551)

## SAM 3D Body

```{warning}
SAM 3D Body uses a gated model on Hugging Face. You must request access and set your token before running. See [Hugging Face access token](../installation.md#hugging-face-access-token).
```

**SAM 3D Body** is a whole-body 3D pose and shape estimation model that recovers 2D/3D keypoints, body mesh vertices, and body/hand/shape parameters in **MHR** (Mesh Human Recovery) parameterization. It supports single- and multi-view setups and uses a DINOv3 or ViT backbone for robust feature extraction.

In the NICE Toolbox, SAM 3D Body runs as the **`sam_3d_body`** algorithm type. Two instances are configured: **`sam_3d_body_dinov3`** (DINOv3 backbone) and **`sam_3d_body_vith`** (ViT-H backbone). They output five components: **`body_joints`** (world-space 3D, requires camera calibration), **`body_joints_local`** (camera-native 3D, always written), **`hand_joints`**, **`hand_joints_local`**, and **`body_mesh`** (mesh vertices and faces).


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

## Gaze Fusion

**`gaze_fusion`** (`gaze_multiview` component) aggregates per-camera gaze estimates from an upstream `gaze_individual` detector into a single fused gaze direction, with optional temporal smoothing. Optionally, it can take `eye_closed_state` as input and use it to filter or interpolate the fused output during eye closures via `closed_eye_filter_mode` (supports `"none"`, `"nan"`, or `"interpolate"`, defaulting to `"none"`).

Four instances are configured by default:

- **`gaze_weighted`** — averages all cameras, weighted by confidence.
- **`gaze_per_subject`** — selects one dedicated camera per subject (`view_left` / `view_right`).
- **`gaze_only_center`** — uses the central view only, where both subjects are visible.
- **`gaze_weighted_closed_eyes`** — weighted averaging over UniGaze estimates, interpolating gaps where `eye_state_ear` reports closed eyes (`closed_eye_max_interpolate_gap = 10`).

## Gaze Interaction

**`gaze_distance`** (`gaze_interaction` component) detects mutual gaze between subjects by computing the angular distance between each subject's fused gaze vector and the direction towards the other subject. The `threshold_look_at` parameter (default `0.4`) controls the sensitivity of mutual gaze detection.

Two instances are configured by default: **`gaze_distance_2d`** and **`gaze_distance_3d`**, for image-plane and world-space geometry respectively.

## Kinematics

**`velocity_body`** (`kinematics` component) computes frame-by-frame movement dynamics — velocity and acceleration — from body joint positions produced by an upstream `body_joints` detector. No separate model weights are required.

Four instances are configured by default: **`velocity_body_2d`** and **`velocity_body_3d`** read from the 2D pose pipeline, while **`velocity_body_motiobert_local`** and **`velocity_body_sam_3d_local`** read the camera-native 3D output of MotionBERT and SAM 3D Body respectively.

## Proximity

**`body_distance`** (`proximity` component) estimates the physical distance between subjects based on selected body keypoints (default: `nose`). It reads from an upstream `body_joints` detector and requires camera calibration for world-space distance estimates. No separate model weights are required. Two instances are configured by default: **`body_distance_2d`** and **`body_distance_3d`**.

## InsightFace (Face Detection)

**InsightFace** is a face analysis toolkit. In the NICE Toolbox it provides **face detection** and five-point face keypoints, running as the **`insight_face`** algorithm and outputting the **`face_bounding_box`** component. The detected bounding boxes are consumed downstream by SPIGA, which relies on them instead of running its own face detector.

The model pack is selectable via `model_pack` (`buffalo_l` by default; `buffalo_m`, `buffalo_s`, `buffalo_sc`, and `antelopev2` are also supported), and `det_thresh` sets the minimum detection confidence. Keypoint post-processing shares the same options as SPIGA: Savitzky-Golay filtering (`filter_keypoints`), linear interpolation over short dropouts (`interpolate_keypoints`), and 3D triangulation across cameras (`triangulate_keypoints`).

The default `buffalo_l` pack detects faces with **SCRFD-10GF**; the packs also bundle ArcFace recognition and dense landmark models, which the NICE Toolbox does not use.

[Guo et al., 2021](https://arxiv.org/abs/2105.04714) (SCRFD) — see the [InsightFace repository](https://github.com/deepinsight/insightface#citation) for the full list of citations covering the other bundled models.

## Py-FEAT (Facial Expression Analysis Toolbox)

**Py-FEAT** is a toolkit for automated facial expression analysis. In the NICE Toolbox it runs as the **`py_feat`** algorithm and outputs the **`emotion_individual`** component. It is used for face detection (`img2pose`), action unit recognition (`xgb`, XGBoost-based, continuous AU probabilities), emotion detection (`resmasknet`), and subject identity verification (`facenet`).

[Cheong et al., 2023](https://doi.org/10.1007/s42761-023-00191-4)

## SPIGA (Shape Preserving Facial Landmarks with Graph Attention)

**SPIGA** is a hybrid CNN-GNN model for facial landmark localization and 6DoF head pose estimation. In the NICE Toolbox it runs as the **`spiga`** algorithm and is used for **landmark localization** (`face_landmarks`) and **head orientation** estimation (`head_orientation`) across all configured cameras. Face detection is handled by InsightFace; outputs are head pose vectors per camera-subject-frame.

[Prados-Torreblanca et al., 2022](https://arxiv.org/abs/2210.07233)

## MediaPipe Face Landmarker

**MediaPipe Face Landmarker** is a real-time, lightweight facial landmark and expression analysis model developed by Google. It detects faces in the input frames, estimates 478 3D facial keypoints, and predicts 52 facial blendshape coefficients (coefficients of facial muscles representing expressions like eye blink, smile, etc.).

In the NICE Toolbox, it runs as the **`mediapipe_face_landmarks`** algorithm and populates the following output components:
- **`face_landmarks_local`**: 2D keypoints on the image plane plane (`2d`, `2d_filtered`, `2d_interpolated`) and 3D keypoint coordinates in pixel-scale space (`3d`), relative to the center of the face.
- **`face_blendshapes`**: 52 facial blendshape coefficients (`blendshapes`).

[Kartynnik et al., 2019](https://arxiv.org/abs/1907.06724)

## Eye Closure

Two algorithms support tracking blink/closure states and dynamics from facial landmark outputs:

- **`eye_closure_ear`** (`eye_closure_score` component): Computes the Eye Aspect Ratio (EAR) as the ratio of eye height to eye width, indicating closure degree. [Soukupová and Čech, 2016](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
- **`eye_closure_threshold`** (`eye_closed_state` component): Categorizes the eye state (closed/open) via a threshold applied to the closure score, with optional duration-based temporal filtering. It outputs per-camera (`per_camera_state`) and global (`global_state`) eye closed states for `"left_eye"`, `"right_eye"`, and `"both_eyes"`.

Four instances are configured by default:

- **`eye_closure_score_ear`** computes the EAR score from the face landmarks of `spiga`.
- **`eye_closure_score_ear_rtmpose`** computes the EAR score from the face landmarks of `rtmw_l_wholebody`.
- **`eye_state_ear`** detects closed eyes (`threshold = 0.12`). `gaze_weighted_closed_eyes` uses it to interpolate over eye closures.
- **`eye_blinks_state_ear`** detects blinks: eye closures between `min_duration = 0.05` and `max_duration = 0.5` seconds.


## pyannote (Speaker Diarization)

```{warning}
pyannote uses a gated model on Hugging Face. You must request access and set your token before running. See [Hugging Face access token](../installation.md#hugging-face-access-token).
```

**pyannote** finds "who speaks when" on an audio track. In the NICE Toolbox it runs as the **`pyannote`** algorithm with the [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) model, and outputs the **`audio_diarization`** component. It runs in the `whisperx` environment.

pyannote separates the different voices, but it does not know the names of the speakers. Its speaker labels are anonymous. To give them subject names, use the `audio_range` subject rule on the track: the [`audio_subject_detector`](#subject-identity-system) then matches each pyannote speaker to a subject. So `pyannote` is necessary only for tracks with the `audio_range` rule.

The main parameters are:

- `exclusive`: if `true`, only one speaker speaks at a time (default `false`).
- `min_duration_off`: pauses shorter than this value (in seconds) inside the turn of one speaker are filled (default `0.0`).
- `embedding_exclude_overlap`: if `true`, the speaker embeddings use only speech without overlap (default `true`).

[Bredin, 2023](https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.html)

## WhisperX (Audio Transcription)

**WhisperX** provides fast automatic speech recognition (ASR) with word-level timestamps. By integrating voice activity detection (VAD) preprocessing and forced alignment, WhisperX significantly reduces hallucinations and improves timestamp accuracy compared to the original Whisper model.

The pipeline runs in three stages: VAD preprocessing to segment speech, transcription via a `faster-whisper` backend, and forced phoneme alignment for word-level timestamps. WhisperX does not run the diarization itself. Instead, it takes the `audio_diarization` of the [`audio_subject_detector`](#subject-identity-system) as input and assigns each word to a subject.

Two instances are configured by default. Both share `whisperx_template`:

- **`whisperx`** outputs **`audio_transcription`** and **`speaker_aligned_transcription`**, in which each word has a subject name. It needs `audio_subject_detector` in the `algorithms` list.
- **`whisperx_no_diar`** outputs only **`audio_transcription`**, without subject names.

`track_names` selects the audio tracks (`"*"` for all tracks), and `language` sets the language code, or `"auto"` for automatic detection.

[Bain et al., 2023](https://arxiv.org/abs/2303.00747)

## CrisperWhisper

**CrisperWhisper** is a fine-tuned Whisper variant focused on **verbatim** speech transcription with highly accurate **word-level timestamps**. Unlike standard Whisper models, which paraphrase and omit disfluencies, CrisperWhisper transcribes every spoken word — including fillers, false starts, and repetitions — and was retrained with an adjusted tokenizer and a custom attention-based timestamp objective to sharply localize word boundaries, even around pauses.

In the NICE Toolbox, CrisperWhisper runs as the **`crisper_whisper`** algorithm type. It processes each configured audio track independently, returning per-word chunks with start/end timestamps. A post-processing step (ported from the original repository) redistributes inter-word pauses up to a small threshold evenly between adjacent words to refine the boundaries.

The raw word chunks are then converted into the same unified `{segments, words}` structure used by WhisperX's `audio_transcription` component, so the two transcription backbones are interchangeable downstream.

Like WhisperX, CrisperWhisper does not run the diarization itself. Two instances are configured by default. Both share `crisper_whisper_template`:

- **`crisper_whisper`** takes the `audio_diarization` of the [`audio_subject_detector`](#subject-identity-system) as input and outputs **`audio_transcription`** and **`speaker_aligned_transcription`**.
- **`crisper_whisper_no_diar`** outputs only **`audio_transcription`**, without subject names.

[Wagner et al., 2024](https://arxiv.org/abs/2408.16589)