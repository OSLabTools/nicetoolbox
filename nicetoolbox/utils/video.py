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
from typing import Any, Optional

import cv2

from ..configs.models.video_timestamp import timestamp_to_ms
from ..configs.schemas.detectors_run_file import FrameExtractionConfig
from .ffmpeg import get_ffmpeg_path, get_ffprobe_path
from .system import normalize_ffmpeg_filter_path_in_windows


def get_ffmpeg_base_args(video_file: str, hwaccel: str) -> list:
    """
    Constructs the base argument list for running ffmpeg.
    """
    # fmt: off
    return [
        get_ffmpeg_path(),
        # "none" decodes on cpu; others fail if the hw device can't be created
        "-hwaccel", hwaccel,
        "-i", video_file,
        "-loglevel", "error",
        "-fps_mode", "passthrough",
        "-bsf:v", "setts=pts=N:dts=N",
    ]
    # fmt: on


def split_into_frames(
    video_file: str,
    output_base: str,
    n_frames_expected: Optional[int],
    frame_extraction: FrameExtractionConfig,
    hwaccel: str,
    keep_indices: bool = True,
) -> None:
    """
    Split a video into individual frames using ffmpeg.
    """
    ext = frame_extraction.frame_extension
    output_pattern = os.path.join(output_base, f"%09d_tmp{ext}")
    cmd = get_ffmpeg_base_args(video_file, hwaccel) + frame_extraction.ffmpeg_args() + [output_pattern]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        logging.error(f"FFmpeg failed to process {video_file}.")
        if hwaccel != "none":
            logging.error(
                f"Hardware decoding with ffmpeg_hwaccel = '{hwaccel}' may be unavailable on this machine. "
                "Try ffmpeg_hwaccel = 'none' in machine_specific_paths.toml to decode on the CPU."
            )
        raise AssertionError("Splitting video into frames failed. See log for details.") from None

    frames_list_tmp = glob.glob(os.path.join(output_base, f"*_tmp{ext}"))
    n_frames_extracted = len(frames_list_tmp)

    if n_frames_expected and n_frames_expected != n_frames_extracted:
        logging.warning(
            f"Expected {n_frames_expected} frames, but extracted {n_frames_extracted} frames from {video_file}."
        )
        raise AssertionError("Splitting video into frames failed (frame count mismatch). See log for details.")

    for file in sorted(frames_list_tmp):
        old_idx = int(os.path.basename(file)[:9])
        if keep_indices:
            new_idx = old_idx - 1
            new_filename = os.path.join(output_base, f"{new_idx:09d}{ext}")
            shutil.move(file, new_filename)


def probe_video(video_path: str) -> dict:
    """
    Parse video information using native ffprobe subprocess.
    The collected information: codec, fps, number_of_frames, width, height, duration

    Args:
        video_path (str): Path to the video file.

    Returns:
        dict: Dictionary with 'format' and 'streams' keys matching ffprobe JSON output,
              compatible with json_to_video_info().

    Raises:
        RuntimeError: If ffprobe returns a non-zero exit code.
    """
    # fmt: off
    cmd = [
        get_ffprobe_path(),
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    # fmt: on
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
    except Exception as e:
        logging.error(f"Failed to run ffprobe on {video_path}: {e}")
        raise

    if result.returncode != 0:
        stderr_msg = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffprobe failed for '{video_path}' (exit {result.returncode}): {stderr_msg}")

    return json.loads(result.stdout.decode(errors="replace"))


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

    video_stream: dict[str, Any] = None
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
    frames = None
    with suppress(ValueError, TypeError):
        frames = int(video_stream.get("nb_frames"))

    # get duration info
    duration = None
    with suppress(ValueError, TypeError):
        duration = float(video_stream.get("duration"))

    # failed to find it in the stream, lets try format?
    if duration is None:
        with suppress(ValueError, TypeError):
            duration = float(format.get("duration"))

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


def get_active_log_file() -> Optional[Path]:
    """Find the active log file (e.g. nicetoolbox.log) from logging handlers, if configured."""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


class FFmpegPipeWriter:
    """
    Context manager for streaming NumPy frame arrays directly to FFmpeg via stdin pipe,
    with optional image frame saving to disk and an optional audio track muxed into the video.
    """

    def __init__(
        self,
        output_path: str,
        size: tuple[int, int],  # (width, height)
        save_frames_dir: Optional[str],  # None disables saving individual frames
        fps: float = 30.0,
        codec: str = "libx264",
        pix_fmt_in: str = "bgr24",
        pix_fmt_out: str = "yuv420p",
        output_params: Optional[list[str]] = None,
        log_file: Optional[Path | str] = None,
        audio_path: Optional[str] = None,
        audio_start: Optional[float] = 0.0,
        audio_duration: Optional[float] = None,
    ):
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self.save_frames_dir = save_frames_dir
        if self.save_frames_dir:
            os.makedirs(self.save_frames_dir, exist_ok=True)
        self._frame_count = 0

        params = ["-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "-movflags", "+faststart"]
        if output_params:
            params.extend(output_params)

        audio_input = []
        if audio_path:
            audio_input = ["-ss", f"{audio_start:.6f}"]
            if audio_duration is not None:
                audio_input += ["-t", f"{audio_duration:.6f}"]
            audio_input += ["-i", audio_path]
            params.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"])

        width, height = size
        # fmt: off
        cmd = [
            get_ffmpeg_path(),
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-pix_fmt", pix_fmt_in,
            "-i", "pipe:0",
            *audio_input,
            "-vcodec", codec,
            "-pix_fmt", pix_fmt_out,
            *params,
            output_path,
        ]
        # fmt: on

        target_log = log_file or get_active_log_file()
        self._log_handle = open(target_log, "a", encoding="utf-8") if target_log else None  # noqa: SIM115
        stderr_dest = self._log_handle if self._log_handle else subprocess.DEVNULL

        try:
            self.proc: subprocess.Popen = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr_dest)
        except Exception:
            if self._log_handle:
                self._log_handle.close()
                self._log_handle = None
            raise

    def write_frame(self, frame: Any, frame_idx: Optional[int] = None) -> None:
        """Stream a single numpy frame array into the pipe and optionally write frame file to disk."""
        if self.proc.poll() is not None:
            raise RuntimeError(f"FFmpeg process exited prematurely with code {self.proc.returncode}.")

        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError, ValueError) as e:
            raise RuntimeError(f"FFmpeg pipe broken (exit code {self.proc.returncode}).") from e

        if self.save_frames_dir:
            idx = frame_idx if frame_idx is not None else self._frame_count
            cv2.imwrite(os.path.join(self.save_frames_dir, f"{idx:09d}.jpg"), frame)
        self._frame_count += 1

    def close(self) -> None:
        """Flush buffer and terminate ffmpeg process."""
        with suppress(BrokenPipeError, OSError):
            self.proc.stdin.close()

        ret = self.proc.wait()

        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

        if ret != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {ret}. Check log for details.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def encode_from_folder(
    input_folder: Optional[str],
    out_filename: str,
    fps: float = 30.0,
    start_frame: int = 0,
    audio_path: Optional[str] = None,
    srt_path: Optional[str] = None,
    frame_limit: Optional[int] = None,
) -> bool:
    """Encode image folder/canvas + optional audio + optional subtitles to video using FFmpeg CLI."""
    # fmt: off
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-loglevel", "error",
    ]
    # fmt: on
    out_format = os.path.basename(out_filename).rsplit(".")[-1]

    if input_folder:
        if os.path.isdir(input_folder):
            if os.listdir(input_folder) == []:
                logging.error("Image folder is empty")
                return False
            num, file_format = os.listdir(input_folder)[0].split(".")
            input_folder = os.path.join(input_folder, f"%0{len(num)}d.{file_format}")
        # fmt: off
        cmd.extend([
            "-framerate", str(fps),
            "-start_number", str(start_frame),
            "-i", input_folder,
        ])
        # fmt: on
    else:  # black screen
        # fmt: off
        cmd.extend([
            "-f", "lavfi",
            "-i", f"color=c=black:s=1280x720:r={fps}",
        ])
        # fmt: on

    if audio_path:
        start_frame_ts_sec = timestamp_to_ms(start_frame, fps) / 1000
        cmd.extend(["-ss", str(start_frame_ts_sec)])
        if frame_limit:
            end_frame_ts_sec = timestamp_to_ms(frame_limit, fps) / 1000
            cmd.extend(["-t", str(end_frame_ts_sec)])
        cmd.extend(["-i", audio_path])

    vf_filters = []
    if srt_path:
        srt_escape = normalize_ffmpeg_filter_path_in_windows(srt_path)
        vf_filters.append(f"subtitles={srt_escape}")

    if out_format != "gif":
        vf_filters.append("crop=trunc(iw/2)*2:trunc(ih/2)*2")
        # fmt: off
        cmd.extend([
            "-codec:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ])
        # fmt: on

    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    if audio_path:
        cmd.extend(["-c:a", "aac", "-shortest"])

    if frame_limit:
        cmd.extend(["-vframes", str(frame_limit)])

    cmd.append(out_filename)
    output = subprocess.run(cmd, check=False)
    return output.returncode == 0


def render_subtitled_track(
    srt_path: str,
    audio_path: str,
    output_path: str,
    fps: float,
    default_start_frame: int = 0,
    video_recipe: Optional[Any] = None,
    camera: Optional[str] = None,
    fallback_camera: Optional[str] = None,
) -> bool:
    """
    Render a single subtitled video for one transcription track using FFmpeg.

    Args:
        srt_path (str): Path to the SRT subtitle file for this track.
        audio_path (str): Path to the track's audio source.
        output_path (str): Path to the output mp4 file.
        fps (float): Frames per second.
        default_start_frame (int): Default start frame when no recipe is provided.
        video_recipe (Optional[Any]): Video recipe with frame folder and time range.
        camera (Optional[str]): Target camera to pull video frames from.
        fallback_camera (Optional[str]): Fallback camera if target camera is unavailable.

    Returns:
        bool: True if video rendering succeeded, False if skipped due to missing SRT.
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
            frame_folder = str(video_recipe.frames_dir(chosen))
        start_frame = video_recipe.range_start
        frame_limit = video_recipe.range_end - video_recipe.range_start

    logging.info(f"Baking subtitles into {output_path}")
    logging.info(f"Using frame folder: {frame_folder}" if frame_folder else "Using black background fallback.")

    return encode_from_folder(
        input_folder=frame_folder,
        out_filename=output_path,
        fps=fps,
        start_frame=start_frame,
        audio_path=audio_path,
        srt_path=srt_path,
        frame_limit=frame_limit,
    )
