# Changelog

## 0.3.1

- **[License Change]** To support the open-source community and make NICE Toolbox more accessible, our team decided to change the license of the project from `CC BY-NC-SA 4.0` to `AGPL-3.0`. You can learn more about the [new license terms here](https://choosealicense.com/licenses/agpl-3.0/).
- [UniGaze](https://github.com/ut-vision/UniGaze) (`unigaze`) - a new state-of-the-art gaze direction estimation model.
- [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) (`crisper_whisper`) - audio transcription model for exact word capture (including verbatims like `"[um], [uh], [laughter]"`).
- [InsightFace](https://github.com/deepinsight/insightface) (`insight_face`) - face analysis model, providing face bounding boxes and facial landmarks.
- [Py-Feat](https://github.com/cosanlab/py-feat) (`py_feat`) facial expression library is updated to the latest version, [v2.1.1](https://github.com/cosanlab/py-feat/releases/tag/v2.1.1). The new release adds support for the [face_multitask_v2](https://huggingface.co/py-feat/face_multitask_v2) model, capable of estimating FACS action units, emotions, valence/arousal, gaze, and other modalities in a single model pass.
- **Eye closure detector** (`eye_closure_ear`) using the [eye aspect ratio](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf), which estimates an eyelid closure score from 2D facial landmarks.
- **Closed eyes and blinks detector** (`eye_closure_threshold`) with configurable timing thresholds.
- **Audio evaluation metrics**: transcription word error rate against reference transcripts, with configurable text normalization and plots.
- Extension of the **ELAN connector** for audio transcription and the **napari-deeplabcut connector** for body joint labeling.
- New **unified detector input/output** system with typed array schemas and metadata, applied to both video and audio detectors.
- **Gaze fusion** can now use the detected eye state to interpolate over or discard gaze direction when eyes are closed.
- Rework of the **installation** - new `make` commands and a Docker build script that give more control over which model virtual environments are installed.
- `dataset_properties.toml` was redesigned to support customization at the individual sequence level, named wildcards and templates.
- Rerun visualization of **SAM 3D Body** meshes.
- Many other small fixes and changes made across different detectors and visualizers.

**Breaking changes:**
- Most of the third-party detector venv and conda environments were updated, please reinstall them.
- The [NICE Toolbox Docker image](https://hub.docker.com/r/mpioslab/nicetoolbox) now distributes only the base NICE Toolbox dependencies and environment. Check the [updated documentation](https://nicetoolbox.readthedocs.io/en/stable/installation.html#docker) on how to install specific detectors inside Docker.
- `dataset_properties.toml` was redesigned, please update it based on the provided example.
- The `cur_session_ID` placeholder is deprecated. The `cur_sequence_ID` placeholder was renamed to `cur_sequence_id` (lowercase).
- The `videos` section in `detectors_run_file.toml` was renamed to `sequences`. `sequence_id` now supports wildcards (e.g. `session_*_take_*`).
- Most of the detectors in `detectors_config.toml` were updated with new asset links and the new input dependency system.

## 0.3.0

- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (`sam_3d_body`) - 3D whole-body pose estimation. Supports single- and multi-view setups.
- [WhisperX](https://github.com/m-bain/whisperX) (`whisperx`) - audio transcription and speaker diarization.
- [MotionBERT](https://github.com/Walter0807/MotionBERT) (`motionbert`) - 3D body pose lifting from precomputed 2D body joint detections.
- New MMPose algorithms: `vitpose_huge`, `rtmpose_l_aic`, `rtmpose_l_wholebody`, and `rtmpose_m_mpii`.
- New [ELAN](https://archive.mpi.nl/tla/elan) connector: export outputs to ELAN annotation format for manual labeling workflows.
- New [napari-deeplabcut](https://github.com/DeepLabCut/napari-deeplabcut) connector: export body joint detections to DeepLabCut format.
- Updated **evaluation pipeline**: new ground-truth-based metrics, improved configuration schema, and flexible group-by and aggregation options.
- New asset **download manager**: model weights are now downloaded automatically during setup or first run.
- New **project config**: a central config file per project that holds paths to your dataset and detector configs, decoupling project settings from the NICE Toolbox installation folder.
- **Algorithms Instances** support, allows to create multiple configurations of the same algorithms with different parameters.
- Sequences time ranges `video_start` and `video_stop` now accept timestamps (e.g. `"00:01:30"`) in addition to frame numbers.

**Breaking changes:**
- `detectors_run_file.toml` has changed. `component_algorithm_mapping` and per sequence `components` lists are deprecated. Use `algorithms` list for all desired algorithms instances.
- `evaluation_config.toml` was redesigned, please update it based on the provided example.
- Separate evaluation summaries are currently deprecated and now a part of metrics.
- `EvaluationWrapper` for exporting evaluation results to pandas is deprecated.
- `frameworks` in `detectors_config.toml` are deprecated. There are more general use `templates` now. Please update your config.

## 0.2.2
- Refactoring of data preprocessing and inference for all detectors.
- Major optimization and bug-fixing of py-feat inference.
- Refactoring, optimization, and bug-fixing of multiview-ethgaze.
- Refactoring of config placeholders resolution, making it faster and more stable.
- New config validation system. It will detect missing required fields or wrong field types across all configs.
- Fixes for subject tracking consistency in multiple detectors.
- In `detectors_run_file.toml` you can set `video_length = -1` to process all frames inside a video.

**Breaking changes:**
- The frame index leading zeroes format was extended from `05d` to `09d` to support longer videos. This results in new filenames.
- CSV exported files are now saved inside individual video folders, not inside the root output folder. This can be customized in config.
- All runtime placeholders now start with `cur_<placeholder_name>`. For example, the `<session_ID>` placeholder was renamed to `<cur_session_ID>`.
- Cyclic placeholder dependencies are deprecated. For example, `git_hash = "<git_hash>"` will now raise an error.
- Placeholder shadowing is deprecated. Use unique placeholder names at each level of the config file.
- NICE Toolbox now uses submodule forks of [mmpose](https://github.com/OSLabTools/mmpose) and [SPIGA](https://github.com/OSLabTools/SPIGA). Library versions remain the same, so there should be no changes in results.
- [Multiview-ETH-XGaze](https://github.com/OSLabTools/ETH_XGaze) now supports multiview only inside NICE Toolbox. All logic for multi-camera fusion was moved to NICE.
- `eth_xgaze` now exports raw `3d` and `3d_filtered` for individual cameras and `xgaze_gaze_fused` and `xgaze_gaze_fused_filtered` fused from all cameras.
- `eth_xgaze` now exports `landmarks_2d` with confidence scores.
- `detectors_run_file.toml` config now requires `log_level` and `error_level` fields to be set.

## 0.2.1

- Evaluation module, Docker support, additional detector output, and many other improvements.

## 0.2.0

- Code refactoring, easier installation, and new detectors for emotion individuals and head orientation.

## 0.1.0

- Initial release.