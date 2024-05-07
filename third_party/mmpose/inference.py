import os
import sys
import numpy as np
import logging
from mmpose.apis import MMPoseInferencer

sys.path.append("./third_party/mmpose")
sys.path.append("./third_party")
import file_handling as fh
#import output_sanity_checks as sc
#import tests.test_data as test_data


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1 : array_like
        Bounding box, format: [x1, y1, x2, y2] where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate.
    box2 : array_like
        Bounding box, format: [x1, y1, x2, y2] where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate.

    Returns
    -------
    float
        The Intersection over Union (IoU) overlap between the two bounding boxes. This is 0 if there is no overlap.

    Notes
    -----
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

    Parameters
    ----------
    bboxes : list of lists
        A list where each element is a bounding box defined as a list of four integers [x1, y1, x2, y2],
        where (x1, y1) are the coordinates of the top-left corner, and (x2, y2) are the coordinates
        of the bottom-right corner of the bounding box.

    confidence_scores : list of float
        A list of confidence scores corresponding to each bounding box in `bboxes`.

    overlapping_threshold : float
        A float number (<1) which defines the threshold. If the IoU of two bounding boxes
        exceeds this threshold, the bounding box with the lower confidence score is marked for removal.
        The default is 0.8.

    Returns
    -------
    keep_indices : list of int
        The indices of the bounding boxes that are kept, referring to their positions in the original
        `bboxes` list.
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

def check_correct_and_sort_person_detections(data, num_subjects, bbox_conf_threshold=0.7, bbox_overlapping_threshold=0.8):
    """
       Check the person detections, correct and sort them from Left to Right (based on image 2d bbox coords).
       1. Check bounding box confidence score of each the person detected person in each frame
            if  the confidence score is bbox_conf_threshold delete this detection
       2. Check if number of detected person (after the correction in previous step) is equal to
        num_subjects in dataset config subjects descriptions.
            if not: check if there is any overlapping bbox and delete them if exists.
       3. Check again if number of detected subjects is correct.
            if yes: sort the person detections from left to right in image 2d corrdinates (based on bbox top_left_x coord)
            if no: save the results of previous frame

       Parameters
       ----------
       data : list of dict
           A list where each element is frame results. MMpose inference result.
            ### Explanation about results structure of mmpose ###
            ### frame['predictions'][0] # list of dict. Each detect is a detected person.
            ### the keys for each person dictionary:
            # keypoints: [[x1,y1], [x2,y2], [x3,y3], ..., [xn,yn]], coords. of keypoints, where n = # of keypoints (i.e., n=133 for coco-wholebody)
            # keypoint_score: [c1, c2, c3, ..., cn], confidence score of keypoints (min=0.0, max=1.0), n = # of keypoints
            # bbox: ([x1,y1,x2,y2]), corners of bbox, x1y1 is top left corner
            # bbox_score: int, confidence score of bbox (person detection

       num_subjects : integer
           An Integer that defines the number of expected subjects in dataset.

       bbox_conf_threshold : float ( < 1.0)
           The person detections whose bounding boxes confidence level is below this threshold, will be removed.
           The default is 0.7.

        bbox_overlapping_threshold : float ( < 1.0)
            If the IoU of two bounding boxes exceeds this threshold, the bounding box with
            the lower confidence score is marked for removal. The default is 0.8.

       Returns
       -------
       updated_frame_predictions_list : list of dict
           A list of corrected and sorted frame results.
       """
    logging.info("Starting... Check correct and sort person detections")
    updated_frame_predictions_list = []
    for i, frame in enumerate(data):
        frame_predictions = frame['predictions'][0] # [0] to unlist # list of dict. Each detect is a detected person.
        # delete bbox and related predictions if it is below confidence level threshold
        bbox_list = []
        bbox_score_list = []
        for person in frame_predictions:
            #logging.info(f"person bbox: {person['bbox'][0]}, bbox_conf_score = {person['bbox_score']}")
            bbox_list.append(person['bbox'][0])
            bbox_score_list.append(person['bbox_score'])
        bboxes = np.array(bbox_list)
        bbox_conf_scores = np.array(bbox_score_list)
        # detect indices above confidence level threshold
        indices = np.where(bbox_conf_scores > bbox_conf_threshold)[0]
        updated_bboxes = bboxes[indices]
        updated_bbox_conf_scores = bbox_conf_scores[indices]
        updated_frame_predictions = [frame_predictions[index] for index in indices]

        if len(updated_frame_predictions) != num_subjects:
            # delete overlapping bbox
            keep_bbox_indices = filter_overlapping_bboxes(updated_bboxes, updated_bbox_conf_scores, bbox_overlapping_threshold)
            updated_bboxes = updated_bboxes[indices]
            updated_frame_predictions = [updated_frame_predictions[index] for index in keep_bbox_indices]
            if len(updated_frame_predictions) != num_subjects:
                is_correct_num_detections = False
            else:
                is_correct_num_detections = True
        else:
            is_correct_num_detections = True

        if is_correct_num_detections:
            # Sort detected people from left to right - lowest top_left x value wiil be first
            sorted_indices = sorted(range(len(updated_bboxes)), key=lambda i: updated_bboxes[i][0])
            sorted_frame_predictions = [updated_frame_predictions[i] for i in sorted_indices]
            updated_frame_predictions_list.append(sorted_frame_predictions)
        else:
            logging.error(f" Frame index: {i} - Number of detected people "
                          f"-{len(updated_frame_predictions)}- is not same as subject description."
                          f"previous frame detections will be used")
            updated_frame_predictions_list.append(updated_frame_predictions_list[-1])
    return updated_frame_predictions_list

def convert_output_to_numpy(data, num_persons):
    num_frames = len(data)
    num_keypoints = len(
        data[0]["predictions"][0][0]['keypoints'])  # results[0] first frame: ["predictions"][0][0] first person
    num_estimations = len(data[0]["predictions"][0][0]['keypoints'][0]) + 1  # x, y, [z], and confidence_score
    logging.info(f"frames: {num_frames}, keypoints: {num_keypoints}, estimations: {num_estimations}")
    sorted_frame_predictions = check_correct_and_sort_person_detections(data, num_persons)

    # Initialize numpy arrays
    keypoints_array = np.zeros((num_persons, num_frames, num_keypoints, num_estimations))
    bbox_array = np.zeros((num_persons, num_frames, 1, 5)) #1 is because detected category is person

    for frame_index, frame in enumerate(sorted_frame_predictions):
        for person_index, person in enumerate(frame):
            # keypoints and scores
            for kp_index, (kp, score) in enumerate(zip(person['keypoints'], person['keypoint_scores'])):
                keypoints_array[person_index, frame_index, kp_index] = [kp[0], kp[1], score]

            # Bounding boxes and scores
            bbox = person['bbox'][0]
            bbox_score = person['bbox_score']
            bbox_array[person_index, frame_index, 0] = [bbox[0], bbox[1], bbox[2], bbox[3], bbox_score]
    data_description = {
        '2d': ['coordinate_x', 'coordinate_y', 'confidence_score'],
        'bbox_2d': ['top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y', 'confidence_score']
    }

    return keypoints_array, bbox_array, data_description

def main(config):
    """ Run inference of the method on the pre-loaded image
    """
    logging.basicConfig(filename=config['log_file'], level=config['log_level'],
                        format='%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s')
    logging.info(f'\n\nRUNNING MMPOSE - {config["algorithm"]}!')

    # create inferencer object
    inferencer = MMPoseInferencer(
        pose2d=config["pose_config"],
        pose2d_weights=config["pose_checkpoint"],
        det_model=config["detection_config"],
        det_weights=config["detection_checkpoint"],
        det_cat_ids=[0], # the category id of 'human' class
        device=config['device']
    )
    pass
    camera_keypoints_output = []
    camera_bbox_output = []
    frame_indices = None
    data_desc = None
    for camera_name in config["camera_names"]:
        logging.info(f'Camera - {camera_name}')
        camera_folder = os.path.join(config["input_data_folder"], camera_name)

        if not os.path.exists(camera_folder):
            logging.error(f"Data folder for camera: {camera_name} could not find")

        # camera_frames_list = [f for sublist in config["frames_list"] for f in sublist if camera_name in f] #since each frame inside a list
        #
        # try:
        #     assert len(camera_frames_list)==len(os.listdir(camera_folder)), \
        #         f"Different number of frames in frames list and frames under {camera_folder}"
        # except AssertionError as e:
        #     logging.error(f"Assertion failed: {e}")
        #     sys.exit(1)

        #inference_log = os.path.join(config["method_tmp_folder"], "inference_log.txt") # ToDo add inference log
        # inference
        # Load the image

        result_generator = None
        if config["save_images"]:
            result_generator = inferencer(camera_folder,
                                          pred_out_dir=config["prediction_folders"][camera_name],
                                          show=False,
                                          vis_out_dir=config["image_folders"][camera_name])

        else:
            result_generator = inferencer(camera_folder,
                                          pred_out_dir=config["prediction_folders"][camera_name],
                                          show=False)
        results = [r for r in result_generator]
        num_subjects = len(config["subjects_descr"])

        ### convert results to numpy array
        keypoints_array, bbox_array, estimations_data_descr = convert_output_to_numpy(results, num_subjects)

        camera_keypoints_output.append(keypoints_array)
        camera_bbox_output.append(bbox_array)

        # # data description
        # if data_description is not None:
        #     assert data_descr == data_description, f"Inference MMPose: Inconsistent data description!"
        # else:
        #     data_description = data_descr

        # get frame indices
        frame_inds = sorted(os.listdir(os.path.join(config['input_data_folder'], camera_name)))
        frame_inds = [s.strip('.png').strip('.jpg').strip('.jpeg') for s in frame_inds]
        if frame_indices is not None:
            assert frame_indices == frame_inds
        else:
            frame_indices = frame_inds

        #check if any [0,0,0] prediction
        #for person_results in person_results_list:
        #    test_data.check_zeros(person_results) #raise assertion if there is any [0,0,0] inference

    #  save as npz files
    for component, result_folder in config["result_folders"].items():
        indices = config['keypoints_indices'][component]

        data_desc = {
            '2d': {
                'axis0': config["subjects_descr"],
                'axis1': config["camera_names"],
                'axis2': frame_indices,
                'axis3': config['keypoints_description'][component],
                'axis4': estimations_data_descr['2d']
            },
            'bbox_2d': {
                'axis0': config["subjects_descr"],
                'axis1': config["camera_names"],
                'axis2': frame_indices,
                'axis3': ['full_body'],
                'axis4': estimations_data_descr['bbox_2d']
            }
        }
        out_dict = {
            '2d': np.stack(camera_keypoints_output, axis=1)[:, :, :, indices],
            'bbox_2d': np.stack(camera_bbox_output, axis=1),
            'data_description': data_desc
        }
        save_file_name = os.path.join(result_folder, f"{config['algorithm']}.npz")
        np.savez_compressed(save_file_name, **out_dict)

    # check if numpy results same as json - randomly choose 5 file,keypoint -  ##raise assertion if fails
    # sc.compare_data_values_with_saved_json(
    #     config["prediction_folders"],
    #     config["intermediate_results"],
    #     config["frame_indices_list"],
    #     config["person_threshold"]) ##TODO fix it - it gives an error when not start from 0

    logging.info(f'\nMMPOSE - {config["algorithm"]} COMPLETED!')

if __name__ == '__main__':
    config_path = sys.argv[1]
    config = fh.load_config(config_path)
    main(config)
