
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from xgaze_utils import draw_gaze, normalizeData_face

# Add top-level directory to sys.path depending on repo structure and not cwd
top_level_dir = Path(__file__).resolve().parents[4]
sys.path.append(str(top_level_dir) + "/submodules/ETH-XGaze")
import model as gaze_model  # noqa: E402
from utils import pitchyaw_to_vector  # noqa: E402

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_gaze_estimator(pretrained_model_filename):
    print('load gaze estimator')
    model = gaze_model.gaze_network()
    model.cuda()  # comment this line out if you are not using GPU
    if not os.path.isfile(pretrained_model_filename):
        print('the pre-trained gaze estimation model does not exist.')
        exit(0)
    else:
        print('load the pre-trained model: ', pretrained_model_filename)
    ckpt = torch.load(pretrained_model_filename)
    model.load_state_dict(ckpt['model_state'],
                          strict=True)  # load the pre-trained model
    model.eval()  # change it to the evaluation mode
    return model


def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec


class GazeEstimator():
    def __init__(self, face_model_filename, pretrained_model_filename):
        self.gaze_estimator = get_gaze_estimator(pretrained_model_filename)
        # Load the generic face model with 3D facial landmarks
        self.face_model_load = np.loadtxt(face_model_filename)

    def gaze_estimation(self, input_img, landmarks_sub, cam_matrix, cam_distor):
        is_debug = False

        landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
        landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
        landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
        face_model = self.face_model_load[landmark_use, :]
        facePts = face_model.reshape(6, 1, 3)

        hr, ht = estimateHeadPose(landmarks_sub, facePts, cam_matrix, cam_distor)
        img_normalized, landmarks_normalized, norm_mat, face_center = normalizeData_face(input_img, face_model,
                                                                                         landmarks_sub, hr, ht,
                                                                                         cam_matrix)

        input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
        input_var = trans(input_var)
        input_var = torch.autograd.Variable(input_var.float().cuda())
        input_var = input_var.view(1, input_var.size(0), input_var.size(1),
                                   input_var.size(2))  # the input must be 4-dimension
        pred_gaze = self.gaze_estimator(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
        pred_gaze = pred_gaze[
            0]  # here we assume there is only one face inside the image, then the first one is the prediction
        pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array
        if is_debug:
            # show_gaze = vector_to_pitchyaw(pred_gaze_cam0).reshape(-1)
            face_patch_gaze = draw_gaze(img_normalized, pred_gaze_np)
            cv2.imshow('img', face_patch_gaze)
            cv2.waitKey(0)

        # convert it back to the original camera cooridnate system
        pred_gaze = pitchyaw_to_vector(pred_gaze_np.reshape((1, 2)))
        pred_gaze = np.dot(np.linalg.inv(norm_mat), pred_gaze.reshape((3, 1)))
        pred_gaze = pred_gaze.reshape((3, 1))
        pred_gaze = pred_gaze / np.linalg.norm(pred_gaze)

        return pred_gaze
