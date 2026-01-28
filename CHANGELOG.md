# Changelog

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