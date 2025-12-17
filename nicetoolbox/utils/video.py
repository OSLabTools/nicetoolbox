"""
Helper functions for video processing, conversion, splitting, ...
"""

import glob
import json
import logging
import os
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

from . import system as oslab_sys


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
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "json",
            video_file,
        ]
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


def get_ffmpeg_input_string(video_file: str) -> str:
    """
    Constructs the input string for running ffmpeg in the command line.

    Args:
        video_file (str): The path to the video file.

    Returns:
        str: The constructed ffmpeg input string.
    """
    return f"ffmpeg -i {video_file} -loglevel error "


def sequential2frame_number(number: int, start_frame: int) -> int:
    """
    Converts a sequential number to a frame number based on the given start frame
    and skip frames.

    Args:
        number (int): The sequential number.
        start_frame (int): The starting frame number.

    Returns:
        int: The corresponding frame number.

    """
    return start_frame + (number - 1)


def split_into_frames(
    video_file: str, output_base: str, start_frame: int = 0, keep_indices: bool = True
) -> tuple:
    """
    Split a video into individual frames using ffmpeg.

    Args:
        video_file (str): Path to the input video file.
        output_base (str): Base directory where the frames will be saved.
        start_frame (int, optional): The starting frame index.
            Defaults to 0.
        keep_indices (bool, optional): Whether to keep the original frame indices
            or convert them to sequential numbers. Defaults to True.

    Raises:
        AssertionError: If splitting the video into frames fails.

    Note:
        This function uses ffmpeg to split the video into frames. Make sure ffmpeg
            is installed and accessible in the system's PATH.

    Warning:
        The `skip_frames` option is not properly working yet. Its output is not fully
            understood yet.
    """
    string = get_ffmpeg_input_string(video_file)
    string += os.path.join(output_base, "%09d_tmp.png")

    # split the video
    os.system(string)

    frames_list_tmp = glob.glob(os.path.join(output_base, "*_tmp.png"))
    n_frames_extracted = len(frames_list_tmp)
    n_frames_expected = get_number_of_frames(video_file)
    if n_frames_expected != n_frames_extracted:
        logging.warning(
            f"Expected {n_frames_expected} frames, but extracted "
            f"{n_frames_extracted} frames from {video_file}."
        )
        raise AssertionError("Splitting video into frames failed. See log for details.")

    # convert continuous file numbers to actual frame indices
    frames_list, frame_indices_list = [], []  # BACKWARD COMPATIBILITY
    for file in sorted(frames_list_tmp):
        old_idx = int(os.path.basename(file)[:9])
        if keep_indices:
            new_idx = sequential2frame_number(old_idx, start_frame)
            new_filename = os.path.join(output_base, f"{new_idx:09d}.png")
            if oslab_sys.detect_os_type() == "windows":
                shutil.move(file, new_filename)
            else:
                os.system(f"mv {file} {new_filename}")
            # Below: BACKWARD COMPATIBILITY
            frame_indices_list.append(new_idx)
            frames_list.append(new_filename)
        else:
            frame_indices_list.append(old_idx)
            frames_list.append(file)

    return frames_list, frame_indices_list


def equal_splits_by_frames(
    video_file: str,
    output_base: str,
    frames_per_split: int,
    keep_last_split: bool = True,
    start_frame: int = None,
    number_of_frames: int = None,
) -> list:
    """
    Splits a video into equal segments based on the number of frames.

    Args:
        video_file (str): The path to the input video file.
        output_base (str): The base path for the output segments.
        frames_per_split (int): The number of frames per split segment.
        keep_last_split (bool, optional): Whether to keep the last split segment if it
            is shorter than the others. Defaults to True.
        start_frame (int, optional): The starting frame index.
            Defaults to None.
        number_of_frames (int, optional): The total number of frames to consider.
            Defaults to None.

    Returns:
        List[str]: A list of paths to the created segments.

    Raises:
        AssertionError: If the total number of frames is less than the frames per split.

    """
    # detect the format of the input video
    input_format = video_file[video_file.rfind(".") + 1 :]

    # extract the total number of frames in the video
    total_frames = min(number_of_frames, get_number_of_frames(video_file))

    assert (
        total_frames >= frames_per_split
    ), f"total_frames ({total_frames}) > frames_per_split ({frames_per_split})"

    # create a string of the frame numbers at which the video should be split
    segment_frames = ",".join(
        str(i)
        for i in np.arange(0, total_frames, frames_per_split, dtype=int)[1:].tolist()
    )

    # define file to save a list of all created segments
    output_folder = os.path.dirname(output_base)
    segments_list_file = os.path.join(output_folder, "segments_list.csv")

    # define keyframe interval
    gop = 10  # 12 is the default gop in ffmpeg
    assert frames_per_split >= gop, "NOT IMPLEMENTED PROPERLY"
    while frames_per_split % gop != 0:
        gop += 1

    # construct the string to run ffmpeg in command line
    string = get_ffmpeg_input_string(video_file, number_of_frames, start_frame)
    string += (
        f"-codec:v h264 -g {gop} -f segment "
        f"-segment_frames {segment_frames} "
        f"-segment_list {segments_list_file} "
        f"-segment_list_entry_prefix '{output_folder}/' "
        f"-reset_timestamps 1 {output_base}%05d.{input_format}"
    )

    # split the video
    os.system(string)

    # remove the very last segment if it is shorter than the others
    if not keep_last_split and total_frames % frames_per_split != 0:
        remove_last_segment_from_file(segments_list_file)

    # change to descriptive filenames
    # convert continuous file numbers to actual frame indices
    results_files = []
    segments_list = read_segments_list_from_file(segments_list_file)
    for video_file, start_time, end_time in segments_list:
        fps = get_fps(video_file)
        start = start_frame + int(start_time * fps)
        end = start_frame + int(end_time * fps)
        os.system(f"mv {video_file} {output_base}s{start}_e{end}.{input_format}")
        results_files.append(f"{output_base}s{start}_e{end}.{input_format}")

    return results_files


def cut_length(
    video_file: str,
    output_base: str,
    start_frame: int = None,
    number_of_frames: int = None,
) -> str:
    """
    Cuts a specified length from a video file and saves it as a new file.

    Args:
        video_file (str): The path to the input video file.
        output_base (str): The base name for the output file. The file extension will
            be added automatically.
        start_frame (int, optional): The starting frame index for the cut.
            Defaults to None.
        number_of_frames (int, optional): The number of frames to include in the cut.
            Defaults to None.

    Returns:
        str: The path to the output file.
    """
    format = video_file.split(".")[-1]
    # start to construct the string to run ffmpeg in command line
    string = get_ffmpeg_input_string(
        video_file, number_of_frames, start_frame, skip_frames=None
    )

    # add desired output file and format
    string += f"{output_base}.{format} -y"

    # split the video
    os.system(string)

    return f"{output_base}.{format}"


def read_segments_list_from_file(segments_list_file: str) -> list:
    """
    Reads a CSV file containing a list of segments and returns a list of tuples.

    Args:
        segments_list_file (str): The path to the CSV file containing the list of
            segments. The CSV file should have columns named 'file', 'start', and 'end'.

    Returns:
        List[Tuple[str, float, float]]: A list of tuples, where each tuple represents a
            segment. Each tuple contains the video file name, the start time of the
            segment, and the end time of the segment.
    """

    # Read the CSV file into a pandas DataFrame
    csv = pd.read_csv(segments_list_file, names=["file", "start", "end"])

    # Convert the DataFrame to a list of tuples
    return csv.values.tolist()


def remove_last_segment_from_file(segments_list_file: str) -> None:
    """
    Removes the last segment from the given segments list file.

    This function reads the segments list file into a pandas DataFrame, drops the last
    row, and then writes the updated DataFrame back to the CSV file.

    Args:
        segments_list_file (str): The path to the CSV file containing the list of
            segments.

    Returns:
        None
    """
    segments_df = pd.read_csv(segments_list_file)
    segments_df.drop(index=segments_df.index[-1], inplace=True)
    segments_df.to_csv(segments_list_file)


def frames_to_video(
    input_folder: str, out_filename: str, fps: float = 30.0, start_frame: int = 0
) -> int:
    """
    Convert a folder of frames to a video using ffmpeg.

    Args:
        input_folder (str): Path to the folder containing the frames.
        out_filename (str): Path to the output video file.
        fps (float, optional): Frames per second of the output video. Defaults to 30.0.

    Returns:
        int: Return code of the ffmpeg command.
    """
    if os.path.isdir(input_folder):
        if os.listdir(input_folder) == []:
            logging.error("Image folder is empty")
            return 1
        num, file_format = os.listdir(input_folder)[0].split(".")
        input_folder = os.path.join(input_folder, f"%0{len(num)}d.{file_format}")

    out_format = os.path.basename(out_filename).rsplit(".")[-1]
    if out_format != "gif":
        command = (
            f"ffmpeg -framerate {fps} -start_number {start_frame} "
            f"-loglevel error -i {input_folder} -codec:v h264 "
            f"-pix_fmt yuv420p {out_filename} -y"
        )
    else:
        command = f"ffmpeg -framerate {fps} -start_number {start_frame} "
        f"-loglevel error -i {input_folder} {out_filename} -y"

    output = subprocess.run(command, shell=True, check=False)
    return output.returncode
