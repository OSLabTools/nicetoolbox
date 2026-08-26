# Connectors

**Connectors** are integrations between NICE Toolbox and third-party tools. Each one translates between the toolbox's data formats and the format a particular tool expects, so data can move *into* the toolbox for processing and *out* of it for use elsewhere.

```{contents} Contents
:depth: 3
:local:
```

## ELAN

[ELAN](https://archive.mpi.nl/tla/elan) (EUDICO Linguistic Annotator) is a free annotation tool from the Max Planck Institute for Psycholinguistics, widely used for frame-accurate annotation of audio and video.

ELAN organizes annotations into **tiers** — named tracks along a shared timeline, each holding intervals with a start time, an end time, and a text label. The connector exchanges **tab-delimited text** exported from ELAN. 

Currently implemented tasks are:

| Task | Direction | Data |
|---|---|---|
| `import_gaze` | ELAN → toolbox | Gaze interaction annotations |
| `export_transcription` | toolbox → ELAN | Audio transcription |
| `import_transcription` | ELAN → toolbox | Corrected audio transcription |

For a full walkthrough over gaze annotation, see [Tutorial 6](../tutorials/tutorial6_elan_connector.md).

### Import gaze

Imports manually annotated gaze interaction as ground truth for evaluation. This task is import-only — the toolbox does not export gaze to ELAN.

```bash
elan_connector import_gaze --project_folder_path . --connector_config <configs_folder_path>/connectors/elan_import_gaze.toml --machine_specifics machine_specific_paths.toml
```

Annotate in ELAN, then export via **File → Export As → Tab-delimited Text**. In the export dialog:

- Under **By Tier Names**, check every tier you want to import.
- Under **Include time column for**, check **Begin Time**, **End Time**, and **Duration**.
- Under **Include time format**, check **hh:mm:ss.ms** and **ss.msec**.

Together these produce the 9-column layout the connector expects. The remaining output options can be left at their defaults.

Each subject needs one tier named `<subject_id> eyes` (for example `client eyes`). The prefix before the space is the subject ID that `[subjects]` maps to a toolbox name.

Two interval labels are recognized:

| Label | Meaning |
|---|---|
| `eyfx` | Looking at the other person (fixed gaze) |
| `eyga` | Gaze averted / not looking at partner |

```toml
export_csv = true

[run.sequence_xyz]
input  = "<dataset_folder>/annotations_raw/gaze_elan/sequence_xyz_gaze.txt"
output = "<elan_output>/sequence_xyz_gaze.npz"
video  = "<dataset_folder>/sequence_xyz/view_top.mp4"
start  = 0
end    = -1

[subjects]
client    = "person_left"
therapist = "person_right"
```

| Field | Description |
|---|---|
| `input` | The ELAN tab-delimited `.txt` export |
| `output` | Where to write the resulting `.npz` |
| `video` | Video file, used to read the true FPS and validate that the annotation aligns with the recording |
| `start` / `end` | Frame range to extract, as a frame index or `"HH:MM:SS"`; `end = -1` means to the end |
| `reset_frames` | If `true`, output frame numbering starts at `000000000`; if `false` (default), original video frame numbers are kept |
| `export_csv` | Also write the result as CSV alongside the `.npz` |

`reset_frames` matters when the annotated clip is a subsequence of a longer recording — leave it `false` so imported frames line up with detector output covering the same frames.

The `[subjects]` table maps ELAN tier subject IDs to the subject names from your `dataset_properties.toml`. Any subject left unmapped keeps its original name.

### Export transcription

Writes an existing transcription into an ELAN tab-delimited file for review or correction.

```bash
elan_connector export_transcription --project_folder_path . --connector_config <configs_folder_path>/connectors/elan_export_transcription.toml --machine_specifics machine_specific_paths.toml
```

Point `input` at the component JSON to export — either a raw `audio_transcription` output or a `speaker_aligned_transcription` output. The `tracks` field names which audio tracks to process (for example `["left_mic", "right_mic"]`, or `["room"]` for a shared microphone); track names come from your `dataset_properties.toml`.

| Field | Description |
|---|---|
| `input` | The component JSON to export |
| `output` | Where to write the ELAN `.txt` file |
| `tracks` | Audio tracks to export (at least one) |
| `export_segments` | Write a segment-level tier |
| `export_words` | Write a word-level tier |

At least one of `export_segments` / `export_words` must be true, otherwise the export produces an empty file.

This task takes no frame range: a component output already covers exactly one subsequence and is exported whole.

The `.txt` file it writes uses a 4-column layout: tier, begin time, end time, annotation. Open it in ELAN via **File → Import → Tab-delimited Text**, and in the import dialog assign the column data types **Tier**, **Begin Time**, **End Time**, and **Annotation** to columns 1–4, with the delimiter set to **Tab**.

### Import transcription

Reads a corrected ELAN transcription back into the toolbox JSON format, completing the round trip started by the export.

Once you have corrected the annotations in ELAN, export them again via **File → Export As → Tab-delimited Text**. This time the connector expects the 4-column layout:

- Under **By Tier Names**, check the tiers you want to import back.
- Under **Include time column for**, check **Begin Time** and **End Time**, and leave **Duration** unchecked.
- Under **Include time format**, check **hh:mm:ss.ms** only.

A file with any other number of columns is rejected with an error naming the expected layout.

```bash
elan_connector import_transcription --project_folder_path . --connector_config <configs_folder_path>/connectors/elan_import_transcription.toml --machine_specifics machine_specific_paths.toml
```

| Field | Description |
|---|---|
| `input` | The corrected ELAN `.txt` export |
| `output` | Where to write the resulting transcription JSON |
| `tracks` | Audio tracks to import (at least one) |
| `import_segments` | Read the segment-level tier |
| `import_words` | Read the word-level tier |
| `export_srt` | Also write a subtitle `.srt` file |
| `export_csv` | Also write a CSV alongside the JSON |

## Napari

[napari-deeplabcut](https://github.com/DeepLabCut/napari-deeplabcut) is a plugin for the [napari](https://napari.org/) image viewer that edits keypoint annotations in the DeepLabCut format.

| Task | Direction | Data |
|---|---|---|
| `export_body_joints` | toolbox → napari | Body joint detections |
| `import_body_joints` | napari → toolbox | Corrected body joints |

For a full walkthrough, see [Tutorial 7](../tutorials/tutorial7_napari_deeplabcut_connector.md). For guidance on annotating body joints consistently, see [Body joints labeling](wiki_body_joints_labeling.md).

### Export body joints

Writes detected body joints as a napari project for manual correction. The output is a `labeled-data/<camera>/` folder per camera, holding the extracted frames and an `.h5` keypoint file.

```bash
napari_connector export_body_joints --project_folder_path . --connector_config <configs_folder_path>/connectors/napari_export_body_joints.toml --machine_specifics machine_specific_paths.toml
```

| Field | Description |
|---|---|
| `input` | The `body_joints` NPZ to export |
| `output` | napari project root; gets a `labeled-data/<camera>/` folder per camera |
| `frames_folder` | Folder holding the extracted frames, normally `nicetoolbox_input` |
| `cameras` | Toolbox camera names to export; `"*"` for all |
| `npz_key` | Which array to read from the NPZ, e.g. `2d_filtered` (top-level, applies to every entry) |

Correcting every frame of a long recording is impractical, so the export supports **sliding-window sampling**:

```toml
[window]
size   = 5   # frames kept per window
stride = 30  # distance between window starts
```

This keeps 5 consecutive frames out of every 30. Windows may not overlap, so `stride` must be at least `size`. Omit the `[window]` table to export every frame.

### Import body joints

Reads corrected keypoints back from a napari project into a toolbox NPZ.

```bash
napari_connector import_body_joints --project_folder_path . --connector_config <configs_folder_path>/connectors/napari_import_body_joints.toml --machine_specifics machine_specific_paths.toml
```

| Field | Description |
|---|---|
| `input` | Folder holding the napari annotations |
| `output` | Where to write the resulting `.npz` |
| `fps` | Frame rate of the annotated video, used to resolve timestamps — napari annotations carry no frame rate of their own |
| `start` / `end` | Frame range to extract, as a frame index or `"HH:MM:SS"`; `end = -1` means to the end |
| `reset_frames` | If `true`, output frame numbering starts at `000000000`; if `false`, original video frame numbers are kept |
| `export_csv` | Also write the result as CSV alongside the `.npz` |

Optional `[subjects]` and `[cameras]` tables map napari names onto the subject and camera names from your `dataset_properties.toml`; anything left unmapped keeps its original name.
