import os
import sys
import numpy as np
from mmpose.apis import MMPoseInferencer
import third_party.file_handling as fh
import output_sanity_checks as sc
#import tests.test_data as test_data
import logging


def convert_output_to_numpy(data, num_persons, person_threshold):
    num_frames = len(data)
    num_keypoints = len(
        data[0]["predictions"][0][0]['keypoints'])  # results[0] first frame: ["predictions"][0][0] first person
    num_estimations = len(data[0]["predictions"][0][0]['keypoints'][0]) + 1  # x, y, [z], and confidence_score
    logging.info(f"frames: {num_frames}, keypoints: {num_keypoints}, estimations: {num_estimations}")

    if num_persons == 2:
        personL_result = np.zeros((num_frames, num_keypoints, num_estimations))
        personR_result = np.zeros((num_frames, num_keypoints, num_estimations))

        for i, frame in enumerate(data):
            for person in frame['predictions'][0]:  # [0] to unlist
                if person['bbox'][0][0] < person_threshold:  ##bbox format x1y1x2y2 - x1y1 top left corner
                    for j, (keypoint, score) in enumerate(zip(person['keypoints'], person['keypoint_scores'])):
                        personL_result[i, j] = [keypoint[0], keypoint[1], score]
                else:
                    for j, (keypoint, score) in enumerate(zip(person['keypoints'], person['keypoint_scores'])):
                        personR_result[i, j] = [keypoint[0], keypoint[1], score]
        predictions_data = [personL_result, personR_result]
    elif num_persons == 1:
        person_result = np.zeros((num_frames, num_keypoints, num_estimations))
        for i, frame in enumerate(data):
            for j, (keypoint, score) in enumerate(zip(
                    frame['predictions'][0][0]['keypoints'],
                    frame['predictions'][0][0]['keypoint_scores'])):
                person_result[i, j] = [keypoint[0], keypoint[1], score]
        predictions_data = [person_result]
    else:
        raise NotImplementedError(
                "MMPose post-inference results conversion is not implemeted "
                "yet for scenes with no or 3 or more people.")
    return predictions_data

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
    camera_output = {}
    for camera_name in config["camera_names"]:
        logging.info(f'Camera - {camera_name}')
        camera_folder = os.path.join(config["input_data_folder"], camera_name)

        fh.assert_and_log(os.path.exists(camera_folder), f"Data folder for camera: {camera_name} could not find")


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
        if config["save_images"]:
            result_generator = inferencer(camera_folder,
                                          pred_out_dir=config["prediction_folders"][camera_name],
                                          show=False,
                                          vis_out_dir=config["image_folders"][camera_name])
            results = [r for r in result_generator]
        else:
            result_generator = inferencer(camera_folder,
                                          pred_out_dir=config["prediction_folders"][camera_name],
                                          show=False)
            results = [r for r in result_generator]
        ### convert results to numpy array
        # output personL, personR
        person_results_list = convert_output_to_numpy(results, len(config["subjects_descr"]),  config["person_threshold"])
        camera_output[camera_name] = person_results_list

        if len(config["subjects_descr"]) == 2:
            # check person data shape
            fh.assert_and_log(
                person_results_list[0].shape == person_results_list[1].shape,
                f"Shape mismatch: Shapes for personL and personR are not the same.")

        #check if any [0,0,0] prediction
        #for person_results in person_results_list:
        #    test_data.check_zeros(person_results) #raise assertion if there is any [0,0,0] inference

        #  save as hdf5 file
        save_file_name = os.path.join(config["intermediate_results"],
                                      f"algorithm_predictions_{camera_name}.hdf5")
        fh.save_to_hdf5(person_results_list, config["subjects_descr"],
                        save_file_name, index=sorted(os.listdir(camera_folder)))

    # check if numpy results same as json - randomly choose 5 file,keypoint -  ##raise assertion if fails
    # sc.compare_data_values_with_saved_json(
    #     config["prediction_folders"],
    #     config["intermediate_results"],
    #     config["frame_indices_list"],
    #     config["person_threshold"]) ##TODO fix it - it gives an error when not start from 0

    logging.info(f'\nMMPOSE - {config["algorithm"]} COMPLETED!')

   # return camera_output
if __name__ == '__main__':
    config_path = sys.argv[1]
    config = fh.load_config(config_path)
    main(config)