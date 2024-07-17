import os
import numpy as np
import rerun as rr

# internal imports
import utils.visual_utils as vis_ut
import utils.config as confh

PREDICTIONS_MAPPING_FILE = "../configs/predictions_mapping.toml"
PREDICTIONS_MAPPING = confh.load_config(PREDICTIONS_MAPPING_FILE)

class Component:
    def __init__(self, visualizer_config, io, logger, component_name):
        self.config = visualizer_config
        self.component_name = component_name
        self.logger = logger
        self.video_name = visualizer_config['media']['video_name']
        self.component_prediction_folder = io.get_component_results_folder(self.video_name, component_name=component_name)
        self.algorithm_list = self.get_algorithm_list()
        self.algorithms_results = self.load_algorithms_results(io)
        self.canvas_list = self.get_canvas_list()
        self.canvas_data = self.create_canvas_data()
    def get_algorithm_list(self):
        algorithms_list = self.config['media'][self.component_name]['algorithms']
        return algorithms_list

    def load_algorithms_results(self, io):
        algorithms_results =[]
        for alg in self.algorithm_list:
            alg_path = io.get_algorithm_result(self.component_prediction_folder, alg)
            algorithms_results.append(np.load(alg_path, allow_pickle=True))
        return algorithms_results

    def get_algorithms_data(self, data_type):
        algorithms_data = []
        for i,alg in enumerate(self.algorithm_list):
            algorithms_data.append(self.algorithms_results[i][data_type])
        return algorithms_data

    def get_canvas_list(self):
        canvas_list = []
        for canvases in self.config['media'][self.component_name]['canvas'].values():
            canvas_list.extend(canvases)
        return list(set(canvas_list))

    def create_canvas_data(self):
        canvas_data = {}
        for data_name, canvas in self.config['media'][self.component_name]['canvas'].items():
            self.algorithms_data = self.get_algorithms_data(data_name)
            canvas_data[data_name] = self.algorithms_data
        return canvas_data

    def get_alg_color(self, alg_idx):
        return self.config['media'][self.component_name]['appearance']["colors"][alg_idx]

    def get_radii(self, type):
        if type == "3d":
            return self.config['media'][self.component_name]['appearance']["radii"]['3d']
        elif type == "camera_view":
            return self.config['media'][self.component_name]['appearance']["radii"]['camera_view']
        else:
            raise ValueError("Invalid type. Use either '3d' or 'camera_view'")

class BodyJointsComponent(Component):
    def __init__(self, visualizer_config, io, logger, component_name):
        super().__init__(visualizer_config, io, logger, component_name)
        self.camera_names = self.get_camera_names_from_data()
        self.subject_names = self.get_subject_names()

    def get_subject_names(self):
        # data description axis0 gives subject information
        # again the results will be read from first algorithm
        subject_names = self.algorithms_results[0]['data_description'].item()['2d']['axis0']
        return subject_names

    def get_camera_names_from_data(self):
        # note: All these numpy arrays share a common structure in their first 3 dimension :
        # [number_of_subjects, number_of_cameras, number_of_frames]
        # data description axis1 gives camera information
        ### by design all algorithms in same component shares the same cameras --
        ### therefore, camera information will be read only from first algorithm
        camera_names = self.algorithms_results[0]['data_description'].item()['2d']['axis1']
        return camera_names

    def get_algorithms_labels(self):
        # axix 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i,alg in enumerate(self.algorithm_list):
            algorithm_labels.append(self.algorithms_results[i]['data_description'].item()['2d']['axis3'])
        return algorithm_labels

    def get_camera_views_from_config(self):
        ## only save the camera view if it is in the dataset
        camera_views = []
        for canvas in self.canvas_list:
            if canvas in self.camera_names:
                camera_views.append(canvas)
        return camera_views

    def log_2d_data(self, entity_path, data_points, alg_idx):
        color = self.get_alg_color(alg_idx)
        radii = self.get_radii('camera_view')
        rr.log(
            entity_path,
            rr.Points2D(data_points,
                        keypoint_ids=list(range(data_points.shape[0])),
                        colors=color,
                        radii=radii)
        )

    def log_3d_data(self, entity_path, data_points, alg_idx):
        color = self.get_alg_color(alg_idx)
        radii = self.get_radii('3d')
        rr.log(
            entity_path,
            rr.Points3D(data_points,
                        keypoint_ids=list(range(data_points.shape[0])),
                        colors=color,
                        radii=radii)
        )
    def calculate_middle_eyes(self, dimension):
        #we will use first algorithm results
        labels = self.get_algorithms_labels()[0]
        right_eye_idx = labels.index("right_eye")
        left_eye_idx = labels.index("left_eye")
        if (dimension < 2) | (dimension >3):
            assert "supported dimensions are: 2 or 3"
        dim = f'{dimension}d'
        data = self.algorithms_results[0][dim]
        mean_value = np.mean(data[:, :, :, [right_eye_idx, left_eye_idx], :], axis=3)
        return (mean_value, self.camera_names)

    def get_alg_keypoint_type(self, alg_name):
        alg_keypoint_type = self.config['algorithms_config'][alg_name]['keypoint_mapping']
        return alg_keypoint_type

    def get_skeleton_connections(self, alg_idx, predictions_mapping):
        alg_name = self.algorithm_list[alg_idx]
        alg_type = self.get_alg_keypoint_type(alg_name)
        return predictions_mapping['human_pose'][alg_type]['connections'][self.component_name]

    def log_skeleton(self, entity_path, data_points, dimension, alg_idx):
        keypoints_dict = {label:i for i, label in
                                 enumerate(self.get_algorithms_labels()[alg_idx])}
        connections = self.get_skeleton_connections(alg_idx, PREDICTIONS_MAPPING)
        start_points, end_points = [], []
        for connect in connections:
            for k in range(len(connect) - 1):
                if (connect[k] in keypoints_dict.keys()) & (
                        connect[k + 1] in keypoints_dict.keys()):
                    start = keypoints_dict[connect[k]]
                    end = keypoints_dict[connect[k + 1]]
                    start_points.append([data_points[start]])
                    end_points.append([data_points[end]])

        start_points = np.array(start_points).reshape(-1, dimension)
        end_points = np.array(end_points).reshape(-1, dimension)
        color = self.get_alg_color(alg_idx)

        if dimension == 2:
            radii = self.config['media'][self.component_name]['appearance']['radii']['camera_view']
            rr.log(
                entity_path,
                rr.LineStrips2D(np.stack((start_points, end_points), axis=1),
                                colors=color, radii=radii)
            )
        else:
            radii = self.config['media'][self.component_name]['appearance']['radii']['3d']
            rr.log(
                entity_path,
                rr.LineStrips3D(np.stack((start_points, end_points), axis=1),
                                colors=color, radii=radii)
            )
    ## TODO: try again adding annotation context
    # def log_annotation_context(self, label_description, component_name, alg_name, class_id, color):
    #     rr.log(
    #         f"/{component_name}_{alg_name}",
    #         rr.AnnotationContext(
    #             rr.ClassDescription(
    #                 info=rr.AnnotationInfo(id=class_id, label=f"{component_name}_{alg_name}", color=color),
    #                 keypoint_annotations=[
    #                     rr.AnnotationInfo(id=i, label=label, color=np.array(color)) for i,label in enumerate(label_description)
    #                 ]
    #             )
    #         ),
    #          timeless=True,
    #     )

    def visualize(self, frame_idx):
        for canvas in self.canvas_list:
            if canvas == '3D_Canvas':
                for alg_idx, alg_data in enumerate(self.canvas_data["3d"]):
                    alg_name = self.algorithm_list[alg_idx]
                    for subject_idx, subject in enumerate(self.subject_names):
                        subject_3d_points = alg_data[subject_idx, 0, frame_idx]
                        entity_path = self.logger.generate_component_entity_path(self.component_name, is_3d = True, alg_name = alg_name, subject_name=subject)
                        self.log_3d_data(entity_path, subject_3d_points, alg_idx)
                        self.log_skeleton(f'{entity_path}/skeleton', subject_3d_points, dimension=3,
                                          alg_idx=alg_idx)
            else:
                cam_name = canvas
                data_key = [c for c in self.canvas_data.keys() if c != '3d']
                camera_index = self.camera_names.index(canvas)
                for k in data_key:
                    for alg_idx, alg_data in enumerate(self.canvas_data[k]):
                        alg_name = self.algorithm_list[alg_idx]
                        for subject_idx, subject in enumerate(self.subject_names):
                            subject_2d_points = alg_data[subject_idx, camera_index, frame_idx][:, :2] #select first 2 values, 3rd is confidence score
                            entity_path = self.logger.generate_component_entity_path(
                                self.component_name, is_3d = False, alg_name=alg_name, subject_name=subject, cam_name=cam_name)
                            self.log_2d_data(entity_path, subject_2d_points, alg_idx)
                            self.log_skeleton(f'{entity_path}/skeleton', subject_2d_points, dimension=2,
                                              alg_idx=alg_idx)
class HandJointsComponent(BodyJointsComponent):
    def __init__(self, visualizer_config, io, logger, component_name):
        super().__init__(visualizer_config, io, logger, component_name)

class FaceLandmarksComponent(BodyJointsComponent):
    def __init__(self, visualizer_config, io, logger, component_name):
        super().__init__(visualizer_config, io, logger, component_name)

class GazeIndividualComponent(Component):
    def __init__(self, visualizer_config, io, logger, component_name, calib, eyes_middle_3d_data = None, look_at_data_tuple= None):
        super().__init__(visualizer_config, io, logger, component_name)
        self.calib = calib
        self.camera_names = self.get_camera_names_from_data()
        self.subject_names = self.get_subject_names()
        self.eyes_middle_3d_data, cams = eyes_middle_3d_data
        self.landmarks_2d = self.get_landmarks_2d()
        self.camera_view_middle_point_subjects = self.get_camera_view_middle_point_dict()
        self.look_at_data = None
        self.look_at_labels = None
        if look_at_data_tuple:
            self.look_at_data = look_at_data_tuple[0]
            self.look_at_labels = look_at_data_tuple[1]
        self.projected_gaze_data_algs = self.project_gaze_to_camera_views()
    def get_subject_names(self):
        # data description axis0 gives subject information
        # again the results will be read from first algorithm
        subject_names = self.algorithms_results[0]['data_description'].item()['3d']['axis0']
        return subject_names

    def get_camera_names_from_data(self):
        # data description axis1 gives camera information
        ### by design all algorithms in same component shares the same cameras --
        ### therefore, camera information will be read only from first algorithm
        camera_names = self.algorithms_results[0]['data_description'].item()['landmarks_2d']['axis1']  ##we are getting camera names from here because 3d doesn't have any camera info
        return camera_names

    def get_algorithms_labels(self):
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]['data_description'].item()['3d']['axis3'])
        return algorithm_labels

    def get_landmarks_2d(self):
        landmarks_2d_data = self.algorithms_results[0]['landmarks_2d']
        return landmarks_2d_data

    def project_gaze_to_camera_views(self):
        projected_data_algs = []
        for alg_idx, alg in enumerate(self.algorithm_list):
            projected_data_camera_dict = {}
            # Iterate over all cameras
            cameras_list = [canvas for canvas in self.canvas_list if '3d' not in canvas.lower()]
            data = self.algorithms_results[alg_idx]['3d']

            for cam_name in cameras_list:
                cam_matrix, cam_distor, cam_rotation, cam_extrinsic = vis_ut.get_cam_para_studio(self.calib,
                                                                                          cam_name)
                image_width = self.calib[cam_name]['image_size'][0]
                projected_data_camera_dict[cam_name] = np.full(
                    (data.shape[0], data.shape[2], 2), np.nan)

                for subject_idx, subject_name in enumerate(self.subject_names):
                    if subject_idx in self.config['camera_sees_subjects'][cam_name]:
                        gaze_vectors = data[subject_idx, 0, :,
                                       :]  # Extract all frames at once
                        dx, dy = vis_ut.reproject_gaze_to_camera_view_vectorized(cam_rotation,
                                                                          gaze_vectors, image_width)
                        projected_data_camera_dict[cam_name][subject_idx, :, 0] = dx
                        projected_data_camera_dict[cam_name][subject_idx, :, 1] = dy
            # Create a dictionary to store reprojected data for each camera
            projected_data_algs.append(projected_data_camera_dict)
        return projected_data_algs

    def log_2d_data(self, entity_path, head_points, data_points,color):
        radii = self.get_radii('camera_view')
        rr.log(entity_path,
               rr.Arrows2D(origins=np.array(head_points).reshape(-1, 2),
                           vectors=np.array(data_points).reshape(-1,
                                                                 2),
                           colors=np.array(color),
                           radii=radii))

    def log_3d_data(self, entity_path, head_points, data_points, color):
        radii = self.get_radii('3d')
        rr.log(entity_path,
               rr.Arrows3D(origins=np.array(head_points).reshape(-1, 3),
                                                       vectors=np.array(data_points).reshape(-1,
                                                                                              3)/2, #divided by two to make it shorter in visualization
                                                       colors=np.array(color).reshape(-1, 3),
                                                       radii=radii))

    def get_look_at_color(self, sub_idx, alg_idx, look_to_subject, frame_idx):
        look_to_label = f'look_at_{look_to_subject}'
        look_to_ind = self.look_at_labels.index(look_to_label)
        is_look_at = self.look_at_data[sub_idx, 0, frame_idx, look_to_ind]
        if is_look_at:
            color_index = 0
        else:
            color_index = 1
        return self.config['media']["gaze_interaction"]['appearance']["colors"][alg_idx][color_index]

    def get_camera_view_middle_point_dict(self):
        camera_view_middle_point = {}
        for cam_idx, cam_name in enumerate(self.camera_names):
            subjects_middle_points = []
            for subject_idx, subject in enumerate(self.subject_names):
                ## calculate the middle point of face
                mean_face = np.mean(self.landmarks_2d[:,:,:,:4,:], axis=3)
                subjects_middle_points.append(mean_face[subject_idx, cam_idx, :])
            camera_view_middle_point[cam_name] = subjects_middle_points
        return camera_view_middle_point

    def visualize(self, frame_idx):
        for canvas in self.canvas_list:
            if canvas == '3D_Canvas':
                for alg_idx, alg_data in enumerate(self.canvas_data["3d"]):
                    alg_name = self.algorithm_list[alg_idx]
                    for subject_idx, subject in enumerate(self.subject_names):
                        subject_gaze_individual = alg_data[subject_idx, 0, frame_idx]
                        #subject_filtered_gaze_data = ut.apply_savgol_filter(subject_gaze_individual)
                        subject_eyes_middle_3d_data = self.eyes_middle_3d_data[subject_idx, 0, frame_idx]
                        entity_path = self.logger.generate_component_entity_path(
                            self.component_name, is_3d=True, alg_name=alg_name,
                            subject_name=subject)
                        # gaze interaction defines color
                        if self.look_at_data.all() != None:
                            if subject_idx + 1 < len(self.subject_names)-1: #look at subject either one forward or one backward in index
                                look_to_subject = self.subject_names[subject_idx + 1]
                            else:
                                look_to_subject = self.subject_names[subject_idx - 1]
                            color = self.get_look_at_color(subject_idx, alg_idx, look_to_subject, frame_idx)
                        else:
                            color = self.config['media'][self.component]['appearance']["colors"][alg_idx]
                        self.log_3d_data(entity_path, subject_eyes_middle_3d_data, subject_gaze_individual, color)
            else:
                cam_name = canvas
                for alg_idx, alg_data in enumerate(self.canvas_data["3d"]):
                    alg_name = self.algorithm_list[alg_idx]
                    for subject_idx, subject in enumerate(self.subject_names):
                        camera_data = self.projected_gaze_data_algs[alg_idx][canvas]
                        frame_data = camera_data[subject_idx,frame_idx]
                        subject_eyes_mid = self.camera_view_middle_point_subjects[canvas][subject_idx][frame_idx][:2]
                        entity_path = self.logger.generate_component_entity_path(
                            self.component_name, is_3d=False, alg_name=alg_name,
                            subject_name=subject, cam_name=cam_name)
                        # gaze interaction defines color
                        if self.look_at_data.all() != None:
                            if subject_idx + 1 < len(self.subject_names)-1: #look at subject either one forward or one backward in index
                                look_to_subject = self.subject_names[subject_idx + 1]
                            else:
                                look_to_subject = self.subject_names[subject_idx - 1]
                            color = self.get_look_at_color(subject_idx, alg_idx, look_to_subject, frame_idx)
                        else:
                            color = self.config['media'][self.component_name]['appearance']["colors"][alg_idx]
                        self.log_2d_data(entity_path, subject_eyes_mid, frame_data, color)

class GazeInteractionComponent(Component):
    def __init__(self, visualizer_config, io, logger, component_name):
        super().__init__(visualizer_config, io, logger, component_name)
        self.camera_names = self.get_camera_names_from_data()
        self.subject_names = self.get_subject_names()
    def get_subject_names(self):
        # data description axis0 gives subject information
        # again the results will be read from first algorithm
        subject_names = self.algorithms_results[0]['data_description'].item()['distance_gaze_3d']['axis0']
        return subject_names

    def get_camera_names_from_data(self):
        camera_names = self.algorithms_results[0]['data_description'].item()['distance_gaze_3d']['axis1']
        return camera_names

    def get_algorithms_labels(self, data_name):
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]['data_description'].item()[data_name]['axis3'])
        return algorithm_labels

    def get_lookat_data(self):
        # read from first algorithm
        data_name = 'gaze_look_at_3d'
        data_labels = self.get_algorithms_labels(data_name)[0] #0 first algorithm
        data = self.canvas_data[data_name][0] # 0 first alg
        return (data, data_labels)

class ProximityComponent(Component):
    def __init__(self, visualizer_config, io, logger, component_name, eyes_middle_3d_data=None, eyes_middle_2d_data=None):
        super().__init__(visualizer_config, io, logger, component_name)
        self.camera_names = self.get_camera_names_from_data()
        self.subject_names = self.get_subject_names()
        self.eyes_middle_3d_data, cams = eyes_middle_3d_data
        if self.eyes_middle_3d_data.all():
            self.middle_point_3d = self.get_middle_point_3d()
        self.eyes_middle_2d_data, middle_eye_camera_names = eyes_middle_2d_data
        self.camera_view_middle_point_dict = self.get_camera_view_middle_point_dict()

    def get_subject_names(self):
        subject_names = self.algorithms_results[0]['data_description'].item()['body_distance_2d']['axis0']
        return subject_names

    def get_camera_names_from_data(self):
        camera_names = self.algorithms_results[0]['data_description'].item()['body_distance_2d']['axis1']
        return camera_names

    def get_algorithms_labels(self):
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]['data_description'].item()['body_distance_2d']['axis3'])
        return algorithm_labels

    def get_middle_point_3d(self):
        first_subject_eyes_middle_data = self.eyes_middle_3d_data[0, 0, :].mean(axis=0)
        second_subject_eyes_middle_data = self.eyes_middle_3d_data[1, 0, :].mean(axis=0)
        middle_point = (first_subject_eyes_middle_data+second_subject_eyes_middle_data)/2
        return middle_point

    def get_camera_view_middle_point_dict(self):
        camera_view_middle_point = {}
        for cam in self.camera_names:
            camera_idx = self.camera_names.index(cam)
            first_subject_eyes_middle_data = self.eyes_middle_2d_data[0, camera_idx, :].mean(axis=0)
            second_subject_eyes_middle_data = self.eyes_middle_2d_data[1, camera_idx, :].mean(axis=0)
            middle_point = (first_subject_eyes_middle_data + second_subject_eyes_middle_data) / 2
            camera_view_middle_point[cam] = middle_point

        return camera_view_middle_point

    def log_2d_data(self, entity_path, data_points,alg_idx, camera):
        color = self.get_alg_color(alg_idx)
        radii = self.get_radii('camera_view')
        mid_point = self.camera_view_middle_point_dict[camera]
        proximity_start = np.array([mid_point[0] - (data_points/2), mid_point[1]-100])
        proximity_end = np.array([mid_point[0]  + (data_points/2), mid_point[1]-100])
        rr.log(
            entity_path,
            rr.LineStrips2D(np.vstack((proximity_start, proximity_end)),
                            colors=color, radii=radii, labels="Proximity"))

    def log_3d_data(self, entity_path, data_points, alg_idx):
        color = self.get_alg_color(alg_idx)
        radii = self.get_radii('3d')
        proximity_start = np.array(
            [self.middle_point_3d[0] - (data_points/2), self.middle_point_3d[1] -0.5,
             self.middle_point_3d[2]])
        proximity_end = np.array(
            [self.middle_point_3d[0] + (data_points/2), self.middle_point_3d[1] -0.5,
             self.middle_point_3d[2]])
        rr.log(
            entity_path,
            rr.LineStrips3D(np.vstack((proximity_start, proximity_end)),
                            colors=color, radii=radii, labels="Proximity"))

    def visualize(self, frame_idx):
        for canvas in self.canvas_list:
            if canvas == '3D_Canvas':
                for alg_idx, alg_data in enumerate(self.canvas_data["body_distance_3d"]):
                    alg_name = self.algorithm_list[alg_idx]
                    frame_proximity = alg_data[:,0,frame_idx,0][0]
                    entity_path = self.logger.generate_component_entity_path(
                        self.component_name, is_3d=True, alg_name=alg_name)
                    self.log_3d_data(entity_path, frame_proximity, alg_idx)

            else:
                cam_name = canvas
                for alg_idx, alg_data in enumerate(self.canvas_data["body_distance_2d"]):
                    alg_name = self.algorithm_list[alg_idx]
                    camera_idx = self.camera_names.index(canvas)
                    frame_proximity = alg_data[:,camera_idx, frame_idx,0][0]
                    entity_path = self.logger.generate_component_entity_path(
                        self.component_name, is_3d=False, alg_name=alg_name, cam_name=cam_name)
                    self.log_2d_data(entity_path, frame_proximity, alg_idx, cam_name)

class KinematicsComponent(Component):
    def __init__(self, visualizer_config, io, logger, component_name):
        super().__init__(visualizer_config, io, logger, component_name)
        self.camera_names = self.get_camera_names_from_data()
        self.subject_names = self.get_subject_names()

    def get_subject_names(self):
        subject_names = self.algorithms_results[0]['data_description'].item()['velocity_body_2d']['axis0']
        return subject_names

    def get_camera_names_from_data(self):
        camera_names = self.algorithms_results[0]['data_description'].item()['velocity_body_2d']['axis1']
        return camera_names

    def get_joints_movement_by_bodypart(self, alg_idx):
        bodypart_motion = []
        labels = self.get_algorithms_labels()[alg_idx]
        for bodypart, joints in self.config['media'][self.component_name]['joints'].items():
            data = self.algorithms_data[alg_idx]
            joint_indices = [labels.index(joint) for joint in joints]
            #THRESHOLD = 0.3
            #data[..., :, 0][data[..., : , 0]< THRESHOLD] = 0
            selected_data = data[:, :, :, joint_indices, :]
            bodypart_motion.append(np.nanmean(selected_data, axis=-2))
        bodypart_motion = np.concatenate(bodypart_motion, axis=-1)
        return bodypart_motion
    def get_algorithms_labels(self):
        # axis 3 gives labels information, this might be different for each algorithm
        algorithm_labels = []
        for i, alg in enumerate(self.algorithm_list):
            algorithm_labels.append(
                self.algorithms_results[i]['data_description'].item()['velocity_body_2d']['axis3'])
        return algorithm_labels

    def log_1d_data(self, entity_path, data_points):
        rr.log(entity_path,
               rr.Scalar(np.round(data_points, decimals=2)))


    def visualize(self, frame_idx):
        for data_name, data in self.canvas_data.items():
            for alg_idx, alg_data in enumerate(data):
                alg_name = self.algorithm_list[alg_idx]
                for subject_idx, subject in enumerate(self.subject_names):
                    joints_bodypart_motion = self.get_joints_movement_by_bodypart(alg_idx)
                    for idx, bodypart in enumerate(
                            self.config['media'][self.component_name]['joints'].keys()):
                        if data_name == "velocity_body_3d":
                            frame_bodypart_data = joints_bodypart_motion[subject_idx, 0, frame_idx][idx]
                            entity_path = self.logger.generate_component_entity_path(
                                self.component_name, is_3d=True, alg_name=alg_name,
                                subject_name=subject, bodypart=bodypart)
                            self.log_1d_data(entity_path, frame_bodypart_data)

                        elif data_name == "velocity_body_2d":
                            for camera_idx, camera in enumerate(self.camera_names):
                                frame_kinematic_2d = alg_data[subject_idx, camera_idx, frame_idx][idx]
                                entity_path = self.logger.generate_component_entity_path(
                                    self.component_name, is_3d=False, alg_name=alg_name,
                                    subject_name=subject,cam_name=camera, bodypart=bodypart)
                                self.log_1d_data(entity_path, frame_kinematic_2d[0])







