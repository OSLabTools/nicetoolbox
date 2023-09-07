"""

"""
import os
import cv2
import shutil
import pandas as pd

from oslab_utils.video import equal_splits_by_frames, get_number_of_frames, \
    get_video_GOP, read_segments_list_from_file


def test_video_equal_splits_by_frames():
    input_file = './tests/data/cam1.avi'
    frames_per_split = 100

    tmp_folder = './tests/tmp'
    os.makedirs(tmp_folder, exist_ok=True)
    output_base = os.path.join(tmp_folder, 'out')

    # run function
    segments_list_file = equal_splits_by_frames(input_file, output_base,
                                                frames_per_split)

    # read result list
    segments_list = read_segments_list_from_file(segments_list_file)

    correct_frame_number = True
    for video_file in segments_list[:-1]:
        total_frames = get_number_of_frames(video_file)
        correct_frame_number *= (total_frames == frames_per_split)
        print(total_frames)

    get_video_GOP('./tests/tmp/out_00000.avi', tmp_folder)
    shutil.rmtree(tmp_folder)
    assert correct_frame_number

