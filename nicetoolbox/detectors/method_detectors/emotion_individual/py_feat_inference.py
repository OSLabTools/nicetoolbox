"""
Run Py-FEAT on the data.
"""

import logging
import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from feat import Detector

# ensure project root on PYTHONPATH
top_level_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(top_level_dir))
from utils import filehandling as fh  # noqa: E402
from utils import system as oslab_sys  # noqa: E402

matplotlib.use("Agg")

# Set multiprocessing start method based on OS
START_METHOD = "spawn" if oslab_sys.detect_os_type() == "windows" else "forkserver"
multiprocessing.set_start_method(START_METHOD, force=True)

# Enable cuDNN optimization for faster CNN inference
torch.backends.cudnn.benchmark = True


def process_view(args):
    """
    Run emotion / action-unit / pose detection on a batch of frames.

    Uses Py-FEAT's Detector to process a list of image file paths in one batch.
    Returns an empty DataFrame if `frames` is empty or an error occurs.

    Args:
        args (tuple):
            frames (list[str]): List of image file paths to process.
            batch_size (int): Number of images per batch for inference.

    Returns:
        pandas.DataFrame:
            A DataFrame containing one row per detected face with columns such as:
            ['frame', 'input', 'faceboxes', 'aus', 'emotions', 'poses', 'Identity', …].
            Returns an empty DataFrame if no frames provided or on error.
    """
    frames, batch_size = args
    if not frames:
        logging.error(
            "Empty frame list passed to process_view -> returning empty DataFrame"
        )
        return pd.DataFrame()

    try:
        detector = Detector(
            face_model="retinaface",
            au_model="xgb",
            emotion_model="resmasknet",
            facepose_model="img2pose",
            device="cuda",
        )
        return detector.detect_image(
            frames,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            progress_bar=False,
            output_size=None,
        )
    except Exception as e:
        logging.warning(
            f"Error in process_view (first frame: {frames[0]}) -> {e}", exc_info=True
        )
        return pd.DataFrame()


def run_pyfeat(frames_list, config):
    """
    Perform per-view resizing, batch inference, and identity remapping.

    1) Infer the view-column order from the first row of `frames_list`.
    2) Build `frames_dict[view_name] = list of paths for that view`.
    3) For each view:
       • Compute target height/width (within 5% of median).
       • If all images share same size, reuse original paths.
       • Otherwise resize into out_folder/resized/<view>/ and collect new paths.
    4) Run `process_view` in parallel on each resized list.
    5) Drop any `Person_N` labels where N >= number of subjects that view sees,
       then remap remaining `Person_N` -> actual subject label via
       `config["cam_sees_subjects"]` and `config["subjects_descr"]`.

    Args:
        frames_list (list[list[str]]):
            List of per-frame lists of image paths (inner lists in arbitrary order).
        config (dict):
            Must contain at least:
             - batch_size (int)
             - out_folder (str)
             - subjects_descr (list[str])
             - cam_sees_subjects (dict[str, list[int]])

    Returns:
        dict[str, pandas.DataFrame]:
            Mapping from each view name to its processed DataFrame.
    """
    logging.info("Initializing Py-Feat for detection on GPU.")
    batch_size = int(config["batch_size"])

    # 1) Infer view order from first frame paths
    first_row = frames_list[0]
    ordered_views: list[str] = []
    for path in first_row:
        parts = path.split(os.sep)
        if "frames" in parts:
            idx = parts.index("frames")
            ordered_views.append(parts[idx - 1])
        else:
            ordered_views.append(Path(path).parent.parent.name)

    # 2) Build frames_dict: view -> list of paths
    frames_dict = {
        view: [row[ordered_views.index(view)] for row in frames_list]
        for view in ordered_views
    }

    # 3) Resize or reuse images per view
    resized: dict[str, list[str]] = {}
    for view, paths in frames_dict.items():
        # --- 3.1) Gather original (height, width) for each readable image
        hw: list[tuple[int, int]] = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                logging.warning(f"[{view}] could not read {p}")
                continue
            h, w = img.shape[:2]
            hw.append((h, w))

        # --- 3.2) If nothing was readable, skip this view entirely
        if not hw:
            logging.error(f"[{view}] no valid images -> skipping view")
            resized[view] = []
            continue

        # --- 3.3) Compute the unique sorted lists of heights and widths
        heights = sorted({h for h, _ in hw})
        widths = sorted({w for _, w in hw})

        # --- 3.4) If all images already share the same size, just reuse the originals
        if len(heights) == 1 and len(widths) == 1:
            logging.info(
                f"[{view}] all images are {heights[0]}×{widths[0]}, no resize needed"
            )
            resized[view] = paths.copy()
            continue

        # --- 3.5) Otherwise, compute the median H and W as our “target”
        med_h = np.median([h for h, _ in hw])
        med_w = np.median([w for _, w in hw])

        # --- 3.6) Pick a dimension within 5% of the median (try smallest, then next)
        def choose_dim(
            vals: list[int],
            median: float,
            name: str,
            _view: str = view,  # capture `view` so B023 is avoided
        ) -> int | None:
            for idx, v in enumerate(vals[:2]):
                diff = abs(v - median) / median
                if diff <= 0.05:
                    if idx == 1:
                        logging.warning(f"[{_view}] using 2nd-smallest {name}={v}")
                    return v
                logging.warning(f"[{_view}] {name}={v} >5% off median={median}")
            logging.error(f"[{_view}] no acceptable {name} within 5% -> skip")
            return None

        target_h = choose_dim(heights, med_h, "height")
        target_w = choose_dim(widths, med_w, "width")
        if target_h is None or target_w is None:
            resized[view] = []
            continue

        # --- 3.7) Resize each image into out_folder/resized/<view>/
        out_dir = os.path.join(config["out_folder"], "resized", view)
        os.makedirs(out_dir, exist_ok=True)
        new_paths: list[str] = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            rimg = cv2.resize(img, (target_w, target_h))
            fname = os.path.basename(p)
            dst = os.path.join(out_dir, fname)
            cv2.imwrite(dst, rimg)
            new_paths.append(dst)

        # --- 3.8) Store the resized paths for downstream inference
        resized[view] = new_paths

    # 4) Parallel detection
    args = [(resized[view], batch_size) for view in ordered_views]
    workers = max(1, multiprocessing.cpu_count() // 2)
    with multiprocessing.Pool(processes=workers) as pool:
        dfs = pool.map(process_view, args)

    # 5) Identity remapping per view
    view_outputs: dict[str, pd.DataFrame] = {}
    subjects_descr = config["subjects_descr"]
    sees_map = config["cam_sees_subjects"]

    for view, df in zip(ordered_views, dfs):
        if df is None or df.empty:
            logging.warning(f"[{view}] empty result -> skipping identity remap")
            view_outputs[view] = df
            continue

        seen = sees_map.get(view, [])
        count_seen = len(seen)

        # drop any Person_N where N >= count_seen
        def is_invalid(label: str, _max=count_seen) -> bool:
            if not label.startswith("Person_"):
                return False
            try:
                n = int(label.split("_", 1)[1])
            except ValueError:
                return False
            return n >= _max

        mask = df["Identity"].apply(is_invalid)
        if mask.any():
            bad = df.loc[mask, "Identity"].unique().tolist()
            logging.warning(f"[{view}] dropping invalid identities: {bad}")
            df = df.loc[~mask]

        if df.empty:
            logging.warning(f"[{view}] no valid rows remain -> skipping")
            view_outputs[view] = df
            continue

        mapping: dict[str, str] = {}
        for label in df["Identity"].unique():
            try:
                n = int(label.split("_", 1)[1])
            except (IndexError, ValueError):
                logging.warning(
                    f"[{view}] unexpected Identity label '{label}' -> skipping"
                )
                continue
            mapping[label] = subjects_descr[seen[n]]

        df["Identity"] = df["Identity"].replace(mapping)
        view_outputs[view] = df

    return view_outputs


def save_frame(output: pd.DataFrame, cam_name: str, frame_i: int, config: dict) -> None:
    """
    Generate and save a per-frame visualization for one camera view.

    Filters `output` for rows matching `frame_i`, calls `plot_detections()`,
    adds legends if multiple identities, rasterizes axes, and writes a PNG
    to `out_folder/<cam_name>/<filename>.png`.

    Args:
        output (pd.DataFrame): Detection results for one view.
        cam_name (str): Name of the camera view (used for output subfolder).
        frame_i (int): Frame index to visualize.
        config (dict): Must include:
            - out_folder (str)
            - subjects_descr (list[str])
    """
    try:
        frame_data = output[output["frame"] == frame_i]
        if frame_data.empty:
            logging.debug(
                f"[{cam_name}] no detections for frame {frame_i}, skipping save."
            )
            return

        frame_file = frame_data["input"].to_list()[0]
        _, filename = os.path.split(frame_file)

        save_dir = os.path.join(config["out_folder"], cam_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        logging.info(
            f"[{cam_name}] saving visualization for frame {frame_i} -> {save_path}"
        )

        fig_list = frame_data.plot_detections()
        if frame_data["Identity"].nunique() > 1:
            legend_cfg = {"loc": "upper right", "fontsize": 8}
            fig_list[0].axes[1].legend(config["subjects_descr"], **legend_cfg)
            fig_list[0].axes[2].legend(config["subjects_descr"], **legend_cfg)

        for ax in fig_list[0].axes:
            ax.set_rasterized(True)

        fig_list[0].savefig(save_path, dpi=80)
        logging.info(f"[{cam_name}] successfully saved viz for frame {frame_i}")
    except Exception as e:
        logging.warning(
            f"[{cam_name}] Error saving frame {frame_i} -> {e}", exc_info=True
        )
    finally:
        if "fig_list" in locals():
            for fig in fig_list:
                fig.clear()
                del fig


def visualize_and_save_frames_parallel(
    outputs: list[pd.DataFrame], camera_names: list[str], frame_i: int, config: dict
) -> None:
    """
    Schedule concurrent saving of visualizations across all camera views.

    1) Ensure each `out_folder/<view>` directory exists.
    2) Use a ThreadPoolExecutor to call `save_frame` for each (out, view, frame_i).

    Args:
        outputs (list[pd.DataFrame]): View-specific DataFrames in same order
                                      as `camera_names`.
        camera_names (list[str]): List of view names.
        frame_i (int): Frame index to visualize.
        config (dict): Must include `out_folder` and `subjects_descr`.
    """
    for cam in camera_names:
        path = os.path.join(config["out_folder"], cam)
        os.makedirs(path, exist_ok=True)
        logging.debug(f"Ensured directory exists: {path}")

    num_workers = max(1, multiprocessing.cpu_count() // 2)
    pending = [(out, cam, frame_i, config) for out, cam in zip(outputs, camera_names)]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(lambda args: save_frame(*args), pending))

    logging.info(f"Finished scheduling saves for frame {frame_i}")


def main(config: dict) -> None:
    """
    Orchestrate the full Py-FEAT detection pipeline.

    1) Validate and reshape `config["frames_list"]` (must be non-empty list of lists).
    2) Extract `camera_names` and verify each row matches its length.
    3) Call `run_pyfeat` to get view_name -> DataFrame.
    4) Assemble per-view outputs in `camera_names` order.
    5) Aggregate all detections into NumPy arrays
       (faceboxes, AUs, emotions, poses).
    6) Optionally visualize each frame via
       `visualize_and_save_frames_parallel`.
    7) Save compressed results to `<result_folders>/.npz` with proper
       `data_description`.

    Args:
        config (dict): Configuration dictionary.
    """
    logging.basicConfig(
        filename=config["log_file"],
        level=config["log_level"],
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s",
    )
    logging.info("Running emotion intensity detection 'py-feat'!")
    frames_list = config.get("frames_list")
    if not isinstance(frames_list, list) or not frames_list:
        logging.error("config['frames_list'] is missing or empty -> aborting")
        return

    camera_names = [
        n for n in config["camera_names"] if isinstance(n, str) and n.strip()
    ]
    expected = len(camera_names)
    for idx, row in enumerate(frames_list):
        if not isinstance(row, (list, tuple)):
            logging.error(f"frames_list[{idx}] is not a list/tuple -> aborting")
            return
        if len(row) != expected:
            logging.error(
                f"frames_list[{idx}] has {len(row)} items but expected {expected} \
                    -> aborting"
            )
            return

    n_frames = len(frames_list)
    logging.info(f"{n_frames} frames × {expected} cameras detected")

    view_outputs = run_pyfeat(frames_list, config)
    outputs = [view_outputs.get(view, pd.DataFrame()) for view in camera_names]

    non_empty = [df["Identity"] for df in outputs if not df.empty]
    if not non_empty:
        logging.error("No detections at all -> aborting")
        return

    all_ids = pd.concat(non_empty).unique()
    id_to_idx = {idn: i for i, idn in enumerate(all_ids)}
    n_sub = len(id_to_idx)

    faceboxes = np.zeros((n_frames, expected, n_sub, 5))
    aus = np.zeros((n_frames, expected, n_sub, 20))
    emotions = np.zeros((n_frames, expected, n_sub, 7))
    poses = np.zeros((n_frames, expected, n_sub, 3))

    frame_indices: list[str] = []
    for idx, files in enumerate(frames_list):
        names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        if len(set(names)) == 1:
            frame_indices.append(names[0])
        else:
            logging.warning(f"frame {idx} has multiple names {names}")

        for cam_i, df in enumerate(outputs):
            fd = df[df["frame"] == idx]
            for _, row in fd.iterrows():
                s = id_to_idx[row["Identity"]]
                faceboxes[idx, cam_i, s, :] = row.faceboxes
                aus[idx, cam_i, s, :] = row.aus
                emotions[idx, cam_i, s, :] = row.emotions
                poses[idx, cam_i, s, :] = row.poses

        valid = not (
            np.isnan(faceboxes[idx]).any()
            or np.isnan(aus[idx]).any()
            or np.isnan(emotions[idx]).any()
            or np.isnan(poses[idx]).any()
        )
        if config["visualize"] and valid:
            try:
                visualize_and_save_frames_parallel(outputs, camera_names, idx, config)
            except Exception:
                logging.warning(f"visualization error at frame {idx}", exc_info=True)

        if idx % config["log_frame_idx_interval"] == 0:
            logging.info(f"Finished frame {idx} / {n_frames}")

    if not config["visualize"]:
        logging.info("Visualization turned off")

    out = {
        "faceboxes": faceboxes.transpose(2, 1, 0, 3),
        "aus": aus.transpose(2, 1, 0, 3),
        "emotions": emotions.transpose(2, 1, 0, 3),
        "poses": poses.transpose(2, 1, 0, 3),
        "data_description": {
            "faceboxes": {
                "axis0": config["subjects_descr"],
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": [
                    "FaceRectX",
                    "FaceRectY",
                    "FaceRectWidth",
                    "FaceRectHeight",
                    "FaceScore",
                ],
            },
            "aus": {
                "axis0": config["subjects_descr"],
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": [
                    "AU01",
                    "AU02",
                    "AU04",
                    "AU05",
                    "AU06",
                    "AU07",
                    "AU09",
                    "AU10",
                    "AU11",
                    "AU12",
                    "AU14",
                    "AU15",
                    "AU17",
                    "AU20",
                    "AU23",
                    "AU24",
                    "AU25",
                    "AU26",
                    "AU28",
                    "AU43",
                ],
            },
            "emotions": {
                "axis0": config["subjects_descr"],
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": [
                    "anger",
                    "disgust",
                    "fear",
                    "happiness",
                    "sadness",
                    "surprise",
                    "neutral",
                ],
            },
            "poses": {
                "axis0": config["subjects_descr"],
                "axis1": camera_names,
                "axis2": frame_indices,
                "axis3": ["Pitch", "Roll", "Yaw"],
            },
        },
    }

    fn = os.path.join(
        config["result_folders"]["emotion_individual"], f"{config['algorithm']}.npz"
    )
    np.savez_compressed(fn, **out)
    logging.info("'py_feat' COMPLETED!\n")


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
        config = fh.load_config(config_path)
        main(config)
    except Exception as e:
        logging.critical(f"Script crashed -> {e}", exc_info=True)
        sys.exit(1)
