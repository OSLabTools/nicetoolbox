"""
Run the MMPose inference algorithm and save the results as npz files.
"""

import logging
import os
import sys

import numpy as np
import toml
from mmpose.apis import MMPoseInferencer

from nicetoolbox_core.dataloader import ImagePathsByCameraLoader


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (array_like): Bounding box coordinates of the first box in the format
            [x1, y1, x2, y2]. (x1, y1) represents the top-left coordinate and (x2, y2)
            represents the bottom-right coordinate.
        box2 (array_like): Bounding box coordinates of the second box in the format
            [x1, y1, x2, y2]. (x1, y1) represents the top-left coordinate and (x2, y2)
            represents the bottom-right coordinate.

    Returns:
        float: The Intersection over Union (IoU) overlap between the two bounding boxes.
            Returns 0 if there is no overlap.

    Notes:
        IoU = Area of Overlap / Area of Union
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Check if there is no overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of each bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the Intersection over Union
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def filter_overlapping_bboxes(bboxes, confidence_scores, overlapping_threshold):
    """
    Filter out bounding boxes that have high intersection-over-union (IoU) values
    with others, retaining only the bounding boxes with the highest confidence scores
    among those that overlap significantly.

    Args:
        bboxes (list of lists): A list where each element is a bounding box defined as
            a list of four integers [x1, y1, x2, y2], where (x1, y1) are the coordinates
            of the top-left corner, and (x2, y2) are the coordinates of the bottom-right
            corner of the bounding box.
        confidence_scores (list of float): A list of confidence scores corresponding to
            each bounding box in `bboxes`.
        overlapping_threshold (float): A float number (< 1) which defines the threshold.
            If the IoU of two bounding boxes exceeds this threshold, the bounding box
            with the lower confidence score is marked for removal. The default is 0.8.

    Returns:
        keep_indices (list of int): The indices of the bounding boxes that are kept,
            referring to their positions in the original `bboxes` list.
    """
    removed = set()
    for i in range(len(bboxes)):
        if i in removed:
            continue
        for j in range(i + 1, len(bboxes)):
            if j in removed:
                continue
            if calculate_iou(bboxes[i], bboxes[j]) > overlapping_threshold:
                # keep the one with the higher confidence score
                if confidence_scores[i] > confidence_scores[j]:
                    removed.add(j)
                else:
                    removed.add(i)
                    break  # Since i is removed, no need to compare it further

    # only add indices not marked for removal
    keep_indices = [i for i in range(len(bboxes)) if i not in removed]
    return keep_indices


def check_correct_and_sort_person_detections(
    data, num_subjects, bbox_conf_threshold=0.7, bbox_overlapping_threshold=0.9
):
    """
    Check the person detections, correct and sort them from Left to Right (based on
    image 2d bbox coords).

    1. Check bounding box confidence score of each the person detected person in each
        frame if  the confidence score is bbox_conf_threshold delete this detection
    2. Check if number of detected person (after the correction in previous step) is
        equal to num_subjects in dataset config subjects descriptions.
        if not: check if there is any overlapping bbox and delete them if exists.
    3. Check again if number of detected subjects is correct.
        if yes: sort the person detections from left to right in image 2d coordinates
        (based on bbox top_left_x coord)
        if no: save the results of previous frame

    Args
        data (list of dict): A list where each element is frame results. MMpose
            inference result.
            ### Explanation about results structure of mmpose ###
            ### frame['predictions'][0] # list of dict. Each detect is a detected
            ### person.
            ### the keys for each person dictionary:
            # keypoints: [[x1,y1], [x2,y2], [x3,y3], ..., [xn,yn]], coords. of
            # keypoints, where n = # of keypoints (i.e., n=133 for coco-wholebody)
            # keypoint_score: [c1, c2, c3, ..., cn], confidence score of keypoints
            # (min=0.0, max=1.0), n = # of keypoints
            # bbox: ([x1,y1,x2,y2]), corners of bbox, x1y1 is top left corner
            # bbox_score: int, confidence score of bbox (person detection

        num_subjects (integer): An Integer that defines the number of expected subjects
            in dataset.

        bbox_conf_threshold (float): Threshold < 1. The person detections whose
            bounding boxesconfidence level is below this threshold, will be removed.
            The default is 0.7.

        bbox_overlapping_threshold (float): Threshold < 1.0. If the IoU of two bounding
            boxes exceeds this threshold, the bounding box with the lower confidence
            score is marked for removal. The default is 0.8.

    Returns:
        updated_frame_predictions_list (list of dict): A list of corrected and sorted
        frame results.
    """
    logging.info("Starting... Check correct and sort person detections")
    updated_frame_predictions_list = []
    for i, frame in enumerate(data):
        frame_predictions = frame["predictions"][0]  # [0] to unlist # list of dict. Each detect is a detected person.
        # delete bbox and related predictions if it is below confidence level threshold
        bbox_list = []
        bbox_score_list = []
        for person in frame_predictions:
            bbox_list.append(person["bbox"][0])
            bbox_score_list.append(person["bbox_score"])
        bboxes = np.array(bbox_list)
        bbox_conf_scores = np.array(bbox_score_list)
        # detect indices above confidence level threshold
        indices = np.where(bbox_conf_scores > bbox_conf_threshold)[0]
        updated_bboxes = bboxes[indices]
        updated_bbox_conf_scores = bbox_conf_scores[indices]
        updated_frame_predictions = [frame_predictions[index] for index in indices]

        if len(updated_frame_predictions) != num_subjects:
            # delete overlapping bbox
            keep_bbox_indices = filter_overlapping_bboxes(
                updated_bboxes, updated_bbox_conf_scores, bbox_overlapping_threshold
            )
            updated_bboxes = updated_bboxes[indices]
            updated_frame_predictions = [updated_frame_predictions[index] for index in keep_bbox_indices]
            if len(updated_frame_predictions) != num_subjects:
                is_correct_num_detections = False
            else:
                is_correct_num_detections = True
        else:
            is_correct_num_detections = True

        if is_correct_num_detections:
            # Sort detected people from left to right - lowest top_left x value will be
            # first
            sorted_indices = sorted(range(len(updated_bboxes)), key=lambda i: updated_bboxes[i][0])
            sorted_frame_predictions = [updated_frame_predictions[i] for i in sorted_indices]
            updated_frame_predictions_list.append(sorted_frame_predictions)
        else:
            logging.error(
                f" Frame index: {i} - Number of detected people "
                f"-{len(updated_frame_predictions)}- is not same as subject "
                f"description. previous frame detections will be used"
            )
            try:
                updated_frame_predictions_list.append(updated_frame_predictions_list[-1])
            except IndexError:
                updated_frame_predictions_list.append([])
    return updated_frame_predictions_list


def convert_output_to_numpy(data, num_persons):
    """
    Convert the output data from a pose estimation model to numpy arrays.

    The output has the following structure:
    - 2d: Numpy array of shape
        (num_persons, num_frames, num_keypoints,
        [coordinate_x, coordinate_y, confidence_score])
    - bbox_2d: Numpy array of shape
        (num_persons, num_frames, 1,
        [top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence_score])
    - data_description: A dictionary containing the description of the data.

    Args:
        data (list): The output data from the pose estimation model.
        num_persons (int): The number of persons detected in the data.

    Returns:
        tuple: A tuple containing the keypoints array, bbox array, and data description.
    """

    num_frames = len(data)
    num_keypoints = len(
        data[0]["predictions"][0][0]["keypoints"]
    )  # results[0] first frame: ["predictions"][0][0] first person
    num_estimations = len(data[0]["predictions"][0][0]["keypoints"][0]) + 1  # x, y, [z], and confidence_score
    logging.info(f"frames: {num_frames}, keypoints: {num_keypoints}, " f"estimations: {num_estimations}")
    sorted_frame_predictions = check_correct_and_sort_person_detections(data, num_persons)

    # Initialize numpy arrays
    keypoints_array = np.zeros((num_persons, num_frames, num_keypoints, num_estimations))
    bbox_array = np.zeros((num_persons, num_frames, 1, 5))  # 1 is because detected category is person

    for frame_index, frame in enumerate(sorted_frame_predictions):
        for person_index, person in enumerate(frame):
            # keypoints and scores
            for kp_index, (kp, score) in enumerate(zip(person["keypoints"], person["keypoint_scores"])):
                keypoints_array[person_index, frame_index, kp_index] = [
                    kp[0],
                    kp[1],
                    score,
                ]

            # Bounding boxes and scores
            bbox = person["bbox"][0]
            bbox_score = person["bbox_score"]
            bbox_array[person_index, frame_index, 0] = [
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                bbox_score,
            ]
    data_description = {
        "2d": ["coordinate_x", "coordinate_y", "confidence_score"],
        "bbox_2d": [
            "top_left_x",
            "top_left_y",
            "bottom_right_x",
            "bottom_right_y",
            "confidence_score",
        ],
    }

    return keypoints_array, bbox_array, data_description


def main(config):
    """
    Main function to run the MMPose inference.

    Saves the results as npz files to the output folder with the following structure:
    - 2d: Numpy array of shape
        (num_persons, num_cameras, num_frames, num_keypoints,
        [coordinate_x, coordinate_y, confidence_score])
    - bbox_2d: Numpy array of shape
        (num_persons, num_frames, 1,
        [top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence_score])
    - data_description: A dictionary containing the description of the data.

    Args:
        config (dict): A dictionary containing the configuration parameters for the
            MMPose inference algorithm.
    """
    logging.basicConfig(
        filename=config["log_file"],
        level=config["log_level"],
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s",
    )
    logging.info(f'RUNNING MMPOSE - {config["algorithm"]}!')

    logging.info("Creating input data loader...")
    logging.info(f'Using cameras: {config["camera_names"]}')
    # Create input data loader from nicetoolbox-core shared code
    dataloader = ImagePathsByCameraLoader(config=config, expected_cameras=config["camera_names"])
    logging.info(f"Data loader created with cams: {dataloader.cameras}")

    # Create inferencer object from MMPose API
    inferencer = MMPoseInferencer(
        pose2d=config["pose_config"],
        pose2d_weights=config["pose_checkpoint"],
        det_model=config["detection_config"],
        det_weights=config["detection_checkpoint"],
        det_cat_ids=[0],  # the category id of 'human' class
        device=config["device"],
    )

    # Prepare to collect outputs
    camera_keypoints_output = []
    camera_bbox_output = []

    # Inference per camera
    for camera_name, image_paths in dataloader:
        logging.info(f"Camera - {camera_name}")

        if config["visualize"]:
            result_generator = inferencer(
                image_paths,
                pred_out_dir=config["prediction_folders"][camera_name],
                show=False,
                vis_out_dir=config["image_folders"][camera_name],
            )
        else:
            result_generator = inferencer(
                image_paths,
                pred_out_dir=config["prediction_folders"][camera_name],
                show=False,
            )

        results = [r for r in result_generator]

        # convert results to numpy array
        num_subjects = len(config["subjects_descr"])
        keypoints_array, bbox_array, estimations_data_descr = convert_output_to_numpy(results, num_subjects)
        camera_keypoints_output.append(keypoints_array)
        camera_bbox_output.append(bbox_array)

    # infer frame indices from dataloader (for current npz saving)
    # TODO: Keep axis2 frame strings in data description or just tuple or range?
    start_frame, end_frame = dataloader.get_frames_range()
    frame_indices = [f"{idx:09d}" for idx in range(start_frame, end_frame)]

    #  save as npz files
    for component, result_folder in config["result_folders"].items():
        indices = config["keypoints_indices"][component]

        data_desc = {
            "2d": {
                "axis0": config["subjects_descr"],
                "axis1": config["camera_names"],
                "axis2": frame_indices,
                "axis3": config["keypoints_description"][component],
                "axis4": estimations_data_descr["2d"],
            },
            "bbox_2d": {
                "axis0": config["subjects_descr"],
                "axis1": config["camera_names"],
                "axis2": frame_indices,
                "axis3": ["full_body"],
                "axis4": estimations_data_descr["bbox_2d"],
            },
        }
        out_dict = {
            "2d": np.stack(camera_keypoints_output, axis=1)[:, :, :, indices],
            "bbox_2d": np.stack(camera_bbox_output, axis=1),
            "data_description": data_desc,
        }
        save_file_name = os.path.join(result_folder, f"{config['algorithm']}.npz")
        np.savez_compressed(save_file_name, **out_dict)

    logging.info(f'MMPOSE - {config["algorithm"]} COMPLETED!\n')


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = toml.load(config_path)
    main(config)
