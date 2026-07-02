"""
Helper functions for video processing, conversion, splitting, ...
"""

import glob
import json
import logging
import os
import shutil
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

import cv2

from ..configs.models.video_timestamp import timestamp_to_ms
from .system import normalize_ffmpeg_filter_path_in_windows


def get_number_of_frames(video_file: str) -> int:
    """
    Get the number of frames in a video file.

    Args:
        video_file (str): The path to the video file.

    Returns:
        int: The number of frames in the video file.
    """
    return int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FRAME_COUNT))


def get_fps(video_file) -> int:
    """
    Get the frame rate of a video file.

    Args:
        video_file (str): The path to the video file.

    Returns:
        int: The frame rate of the video file.
    """
    fps = int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FPS))
    if (fps == 0) or (fps is None):
        # fmt: off
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of", "json",
            video_file,
        ]
        # fmt: on

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)

            if "streams" in info and info["streams"]:
                rate = info["streams"][0]["avg_frame_rate"]  # e.g. "30/1"
                num, den = map(int, rate.split("/"))
                fps = int(num / den)
                return fps
        except Exception as e:
            logging.error(f"Error in get_fps for {video_file}: {e}")
            return -1
    else:
        return fps


def get_ffmpeg_base_args(video_file: str) -> list:
    """
    Constructs the base argument list for running ffmpeg.

    Args:
        video_file (str): The path to the video file.

    Returns:
        list: The ffmpeg base arguments as a list of strings.
    """
    # fmt: off
    return [
        "ffmpeg",
        "-i", video_file,
        "-loglevel", "error",
        "-vsync", "passthrough",
        "-bsf:v", "setts=pts=N:dts=N"
    ]
    # fmt: on


def split_into_frames(
    video_file: str, output_base: str, n_frames_expected: Optional[int], keep_indices: bool = True
) -> None:
    """
    Split a video into individual frames using ffmpeg.

    Args:
        video_file (str): Path to the input video file.
        output_base (str): Base directory where the frames will be saved.
        n_frames_expected(Optional[int]): Expected number of frames
        keep_indices (bool, optional): Whether to keep the original frame indices
            or convert them to sequential numbers. Defaults to True.

    Raises:
        AssertionError: If splitting the video into frames fails.

    Note:
        This function uses ffmpeg to split the video into frames. Make sure ffmpeg
            is installed and accessible in the system's PATH.
    """
    output_pattern = os.path.join(output_base, "%09d_tmp.png")

    # Construct the command with the modern flag
    cmd = get_ffmpeg_base_args(video_file) + [output_pattern]

    # Split the video
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        logging.error(f"FFmpeg failed to process {video_file}.")
        logging.error("Please update FFmpeg to version >= 5.1.")
        raise AssertionError("Splitting video into frames failed. See log for details.") from None

    # Verify extraction count
    frames_list_tmp = glob.glob(os.path.join(output_base, "*_tmp.png"))
    n_frames_extracted = len(frames_list_tmp)

    if n_frames_expected and n_frames_expected != n_frames_extracted:
        logging.warning(
            f"Expected {n_frames_expected} frames, but extracted {n_frames_extracted} frames from {video_file}."
        )
        raise AssertionError("Splitting video into frames failed (frame count mismatch). See log for details.")

    # Convert continuous file numbers (1-based from ffmpeg) to 0-based frame indices.
    for file in sorted(frames_list_tmp):
        old_idx = int(os.path.basename(file)[:9])
        if keep_indices:
            new_idx = old_idx - 1
            new_filename = os.path.join(output_base, f"{new_idx:09d}.png")
            shutil.move(file, new_filename)


def frames_to_video(
    input_folder: Optional[str],
    out_filename: str,
    fps: float = 30.0,
    start_frame: int = 0,  # TODO: support ms timestamps for super-accurate audio
    audio_path: Optional[str] = None,
    srt_path: Optional[str] = None,
    frame_limit: Optional[int] = None,  # TODO: support ms timestamps for super-accurate audio
) -> int:
    """
    Convert a folder of frames to a video using ffmpeg.

    Args:
        input_folder (Optional[str]): Path to the folder containing the frames.
            If None, a black fallback video is generated.
        out_filename (str): Path to the output video file.
        fps (float, optional): Frames per second of the output video. Defaults to 30.0.
        start_frame (int, optional): The starting frame number. Defaults to 0.
        audio_path (Optional[str], optional): Path to an audio file to include. Defaults to None.
        srt_path (Optional[str], optional): Path to a subtitle (SRT) file to include. Defaults to None.
        frame_limit (int, optional): Limit on how many frames to compose. Defaults to None.

    Returns:
        int: Return code of the ffmpeg command.
    """
    # fmt: off
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
    ]
    out_format = os.path.basename(out_filename).rsplit(".")[-1]

    if input_folder:
        if os.path.isdir(input_folder):
            if os.listdir(input_folder) == []:
                logging.error("Image folder is empty")
                return 1
            num, file_format = os.listdir(input_folder)[0].split(".")
            input_folder = os.path.join(input_folder, f"%0{len(num)}d.{file_format}")
        cmd.extend([
            "-framerate", str(fps), "-start_number", str(start_frame),
            "-i", input_folder,
        ])
    else:  # black screen
        cmd.extend([
            "-f", "lavfi", "-i", f"color=c=black:s=1280x720:r={fps}",
        ])

    if audio_path:
        start_frame_ts_sec = timestamp_to_ms(start_frame, fps) / 1000
        cmd.extend(["-ss", str(start_frame_ts_sec)])
        if frame_limit:
            end_frame_ts_sec = timestamp_to_ms(frame_limit, fps) / 1000
            cmd.extend(["-to", str(end_frame_ts_sec)])
        cmd.extend(["-i", audio_path])

    vf_filters = []
    if srt_path:
        srt_escape = normalize_ffmpeg_filter_path_in_windows(srt_path)
        vf_filters.append(f"subtitles={srt_escape}")

    if out_format != "gif":
        # Even width/height required for yuv420p; odd sizes cause green lines / broken files.
        vf_filters.append("crop=trunc(iw/2)*2:trunc(ih/2)*2")
        cmd.extend([
            "-codec:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ])

    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    if audio_path:
        cmd.extend(["-c:a", "aac", "-shortest"])

    if frame_limit:
        cmd.extend(["-vframes", str(frame_limit)])

    cmd.append(out_filename)
    # fmt: on
    output = subprocess.run(cmd, check=False)
    return output.returncode


def render_subtitled_track_video(
    srt_path: str,
    audio_path: str,
    output_path: str,
    fps: float,
    default_start_frame: int = 0,
    video_recipe=None,
    camera: Optional[str] = None,
    fallback_camera: Optional[str] = None,
) -> bool:
    """
    Render a single subtitled video for one transcription track.

    Encapsulates the per-track work shared by transcription detectors: it skips missing or empty
    SRT files, resolves the frame folder from the video recipe for the track's own ``camera``
    (or renders a black background when unavailable), and bakes the subtitles into the video
    via :func:`frames_to_video`. Callers only need to provide the per-track paths
    inside their own track loop.

    Args:
        srt_path (str): Path to the SRT subtitle file for this track.
        audio_path (str): Path to the track's audio source.
        out_filename (str): Path to the output ``.mp4`` (its directory is created if missing).
        fps (float): Default frames per second (overridden by the recipe range when available).
        default_start_frame (int, optional): Start frame used when no video recipe is given.
        video_recipe (optional): Video input recipe exposing ``root_path``, ``camera_names``,
            ``range_start`` and ``range_end``. When ``None`` a black background video is produced.
        camera (Optional[str]): Camera to overlay subtitles on. ``None`` or a name not present
            in the recipe falls back to ``fallback_camera``; if that is also unavailable a black
            background video is produced.
        fallback_camera (Optional[str]): Camera to use when ``camera`` is unavailable.

    Returns:
        bool: ``True`` if a video was rendered, ``False`` if the track was skipped.
    """
    if not os.path.exists(srt_path):
        logging.warning(f"No SRT found at {srt_path}, skipping visualization.")
        return False

    if os.path.getsize(srt_path) == 0:
        logging.warning(
            f"SRT file {srt_path} is empty. This probably means no speech was detected "
            "for this track. Skipping visualization."
        )
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frame_folder = None
    start_frame = default_start_frame
    frame_limit = None

    if video_recipe:
        chosen = None
        if camera and camera in video_recipe.camera_names:
            chosen = camera
        elif fallback_camera and fallback_camera in video_recipe.camera_names:
            chosen = fallback_camera
        if chosen:
            frame_folder = os.path.join(video_recipe.root_path, chosen, "frames")
        start_frame = video_recipe.range_start
        frame_limit = video_recipe.range_end - video_recipe.range_start

    logging.info(f"Baking subtitles into {output_path}")
    logging.info(f"Using frame folder: {frame_folder}" if frame_folder else "Using black background fallback.")

    frames_to_video(
        input_folder=frame_folder,
        audio_path=audio_path,
        srt_path=srt_path,
        out_filename=output_path,
        fps=fps,
        start_frame=start_frame,
        frame_limit=frame_limit,
    )
    return True


def probe_video(video_path: str) -> dict:
    """
    Parse video information using ffprobe.
    The collected information: codec, fps, number_of_frames, width, height, duration

    Args:
        video_path (str): Path to the video file.

    Returns:
        dict: Return the dictionary holds the video information.
    """
    # fmt: off
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    # fmt: on

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        logging.error(f"Failed to execute ffprobe for {video_path}, error: {e}")
        raise
    if proc.returncode != 0:
        logging.error(f"ffprobe failed while probing video {video_path}, stderr: {proc.stderr.strip()}")
        raise

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse ffprobe JSON output for video: {video_path}")
        raise

    return data


@dataclass
class VideoInfo:
    video_path: Path
    codec: str
    fps: Optional[float]
    frames: Optional[int]
    width: int
    height: int
    duration_in_sec: Optional[int]


def json_to_video_info(data: dict) -> VideoInfo:
    """
    Parse ffprobe video json to compact video info.

    Args:
        data (dict): Dictionary holds video information.

    Returns:
        VideoInfo: Video meta information.
    """
    format = data["format"]
    video_path = Path(format["filename"])

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise RuntimeError(f"No video stream found in file: {video_path}")

    codec = video_stream["codec_name"]
    width = video_stream["width"]
    height = video_stream["height"]

    # get fps information
    try:
        rate = video_stream["avg_frame_rate"]  # fps in format e.g., "30/1"
        if not rate or rate in ("0/0", "N/A"):
            fps = None
        else:
            fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError, TypeError) as e:
        logging.warning(f"Video fps rate could not be extracted: {e}")
        fps = None

    # get number of frames info
    nb_frames_raw = video_stream["nb_frames"]
    frames = int(nb_frames_raw) if nb_frames_raw and nb_frames_raw.isdigit() else None

    # get duration info
    duration = None
    with suppress(ValueError, TypeError):
        duration = float(video_stream.get("duration"))

    if duration is None:
        with suppress(ValueError, TypeError):
            duration = float(data.get("format", {}).get("duration"))

    if duration is None:
        logging.warning("Video duration could not be extracted")

    video_info = {
        "video_path": video_path,
        "codec": codec,
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration_in_sec": duration,
    }
    logging.info(", ".join(f"{k}={v}" for k, v in video_info.items()))

    return VideoInfo(**video_info)
