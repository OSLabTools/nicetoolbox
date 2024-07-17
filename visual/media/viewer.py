import numpy as np
import rerun as rr

class Viewer:
    def __init__(self, visualizer_config):
        self.visualizer_config = visualizer_config
        self.canvas_list = self.create_canvas_list()
        self.is_camera_position = visualizer_config['media']['visualize']['camera_position']
        self.fps =visualizer_config['fps']
        self.create_canvas_roots()

    @staticmethod
    def spawn(app_id="NICE Toolbox Visualization"):
        rr.init(app_id)
        rr.spawn()
        rr.set_time_seconds("time", 0)

    def go_to_timestamp(self, frame_idx):
        frame_time = frame_idx * (1 / self.fps)
        if frame_time == 0:
            frame_time =0.00001
        rr.set_time_seconds("time", frame_time)

    def create_canvas_list(self):
        canvas_list = []
        for component in self.visualizer_config['media']['visualize']['components']:
            for canvases in self.visualizer_config['media'][component]['canvas'].values():
                canvas_list.extend(canvases)
        return list(set(canvas_list))

    def create_canvas_roots(self):
        if '3D_Canvas' in self.canvas_list:
            self.ROOT3D = '3D_Canvas'
            if self.is_camera_position:
                self.CAMERAS_ROOT = '3D_Canvas/cameras'
                self.IMAGES_ROOT = '3D_Canvas/cameras'
            else:
                self.CAMERAS_ROOT = None
                self.IMAGES_ROOT = 'cameras'
        elif self.is_camera_position:
            self.CAMERAS_ROOT = '3D_Canvas/cameras'
            self.IMAGES_ROOT = '3D_Canvas/cameras'
            self.ROOT3D = None
        else:
            self.IMAGES_ROOT = 'cameras'
            self.ROOT3D = None
            self.CAMERAS_ROOT = None
    def get_camera_pos_entity_path(self, camera_name):
        return f'{self.CAMERAS_ROOT}/{camera_name}'
    def get_root3d_entity_path(self):
        return self.ROOT3D
    def get_images_entity_path(self, camera_name):
        return f'{self.IMAGES_ROOT}/{camera_name}'

    def get_is_camera_position(self):
        return self.is_camera_position

    def generate_component_entity_path(self, component, is_3d=True, cam_name = None, alg_name=None, subject_name=None, bodypart=None):
        if (component=='body_joints')| (component=='hand_joints')| (component=='face_landmarks')| (component=='gaze_individual'):
            if is_3d:
                entity_path = f"{self.ROOT3D}/{component}/{alg_name}/{subject_name}"
            else:
                entity_path = f"{self.IMAGES_ROOT}/{cam_name}/{component}/{alg_name}/{subject_name}"
        elif component=='proximity':
            if is_3d:
                entity_path = f"{self.ROOT3D}/{component}/{alg_name}"
            else:
                entity_path = f"{self.IMAGES_ROOT}/{cam_name}/{component}/{alg_name}"
        elif component=='kinematics':
            if is_3d:
                entity_path = f"{alg_name}_{subject_name}/{bodypart}"
            else:
                entity_path = f"{alg_name}_{subject_name}_{cam_name}/{bodypart}"

        else:
            raise ValueError(f"ERROR in generate_component_entity_path(): Component {component} did not implemented")

        return entity_path


    def log_camera(self, camera_calibration, entity_path):
        # intrinsic camera matrix
        K = np.array(camera_calibration['intrinsic_matrix'])
        rr.log(entity_path,
               rr.Pinhole(
                   resolution=camera_calibration['image_size'],
                   image_from_camera=K[:3, :3].flatten(),
               ))
        translation = np.array(camera_calibration['translation']).reshape(3,)
        R = camera_calibration['rotation_matrix']
        R_inv = np.linalg.inv(R)
        center = np.matmul(R_inv, -translation)
        #center *= 1000
        rr.log(entity_path, rr.Transform3D(mat3x3=R_inv, translation=center.flatten()))

    def log_image(self, image, entity_path, img_quality=75):
        rr.log(entity_path, rr.Image(image).compress(jpeg_quality=img_quality))

    def check_multiview(self):
        if self.visualizer_config['media']['multi_view'] == False:
            if "3D_Canvas" in self.canvas_list:
                raise ValueError("ERROR: multi-view parameter in Visualizer_config set false,\n "
                                 "But 3D_Canvas found in components, canvas lists.\n"
                                 "If you don't have multiple cameras, delete 3D_Canvas in all canvases\n"
                                 "If you have multiple cameras, change multi-view parameter as true\n")







