"""
Viewer module for visualizing the components.

This module defines the Viewer class, which is responsible for visualizing the
components of the NICE toolbox.

Classes:
    Viewer: Class for visualizing the components.
"""

import numpy as np
import rerun as rr


class Viewer:
    """
    Class for visualizing the components.

    Attributes:
        config (dict): Configuration settings for the viewer.
    """

    def __init__(self, visualizer_config: dict):
        """
        Initializes the Viewer class with the given configuration.

        Args:
            config (dict): Configuration settings for the viewer.
        """

        self.visualizer_config = visualizer_config

        canvas_list = []
        for component in self.visualizer_config["media"]["visualize"]["components"]:
            for canvases in self.visualizer_config["media"][component][
                "canvas"
            ].values():
                canvas_list.extend(canvases)
        self.canvas_list = list(set(canvas_list))

        self.is_camera_position = visualizer_config["media"]["visualize"][
            "camera_position"
        ]
        self.fps = self.visualizer_config["dataset_properties"]["fps"]
        self._create_canvas_roots()

    @staticmethod
    def spawn(app_id: str = "NICE Toolbox Visualization") -> None:
        """
        Spawns a rerun application with the given app_id.

        Args:
            app_id (str): The ID of the visualization application. Defaults to
                "NICE Toolbox Visualization".
        """
        rr.init(app_id, spawn=True)
        rr.set_time_seconds("time", 0)

    def go_to_timestamp(self, frame_idx: int) -> None:
        """
        Go to the specified frame index in the rerun viewer.

        Args:
            frame_idx (int): The index of the frame to go to.
        """

        frame_time = frame_idx * (1 / self.fps)
        if frame_time == 0:
            frame_time = 0.00001
        rr.set_time_seconds("time", frame_time)

    def get_start_frame(self):
        """
        Returns the start frame for visualization specified in the visualizer config.

        Returns:
            int: The start frame for visualization.
        """
        return self.visualizer_config["media"]["visualize"]["start_frame"]

    def get_end_frame(self) -> int:
        """
        Returns the end frame for visualization specified in the visualizer config.

        Returns:
            int: The end frame for visualization.
        """
        end_frame = self.visualizer_config["media"]["visualize"]["end_frame"]
        if end_frame == -1:
            end_frame = self.visualizer_config["video"]["video_length"]
        return end_frame

    def get_step(self) -> int:
        """
        Returns the frame step size for visualization specified in the visualizer
        config.

        Only every step-th frame will be visualized.

        Returns:
            int: The step for visualization.
        """
        return self.visualizer_config["media"]["visualize"]["visualize_interval"]

    def get_video_start(self) -> int:
        """
        Returns the start time of the video in seconds.

        Returns:
            int: The start time of the video in seconds.
        """
        return self.visualizer_config["video"]["video_start"]

    def _create_canvas_roots(self) -> None:
        """
        Creates the root paths for the 3D canvas, cameras, and images.

        The root paths are used to log the cameras and images in the rerun viewer.
        """
        if "3D_Canvas" in self.canvas_list:
            self.ROOT3D = "3D_Canvas"
            if self.is_camera_position:
                self.CAMERAS_ROOT = "3D_Canvas/cameras"
                self.IMAGES_ROOT = "3D_Canvas/cameras"
            else:
                self.CAMERAS_ROOT = None
                self.IMAGES_ROOT = "cameras"
        elif self.is_camera_position:
            self.CAMERAS_ROOT = "3D_Canvas/cameras"
            self.IMAGES_ROOT = "3D_Canvas/cameras"
            self.ROOT3D = None
        else:
            self.IMAGES_ROOT = "cameras"
            self.ROOT3D = None
            self.CAMERAS_ROOT = None

    def get_camera_pos_entity_path(self, camera_name: str) -> str:
        """
        Returns the entity path for the camera position.

        Args:
            camera_name (str): The name of the camera.

        Returns:
            str: The entity path for the camera position.
        """
        return f"{self.CAMERAS_ROOT}/{camera_name}"

    def get_images_entity_path(self, camera_name):
        """
        Returns the entity path for the images of a specific camera.

        Args:
            camera_name (str): The name of the camera.
        Returns:
            str: The entity path for the images of the specified camera.
        """
        return f"{self.IMAGES_ROOT}/{camera_name}"

    def get_is_camera_position(self):
        """
        Returns a boolean indicating whether to display the camera position in the
        viewer.

        Returns:
            bool: Whether to display the camera position in the viewer.
        """
        return self.is_camera_position

    def generate_component_entity_path(
        self,
        component: str,
        is_3d: bool = True,
        cam_name: str = None,
        alg_name: str = None,
        subject_name: str = None,
        bodypart: str = None,
    ) -> str:
        """
        Generates the entity path for a given component.

        Args:
            component (str): The name of the component.
            is_3d (bool, optional): Flag indicating if the component is 3D.
                Defaults to True.
            cam_name (str, optional): The name of the camera. Defaults to None.
            alg_name (str, optional): The name of the algorithm. Defaults to None.
            subject_name (str, optional): The name of the subject. Defaults to None.
            bodypart (str, optional): The name of the body part. Defaults to None.
        Returns:
            str: The generated entity path.
        Raises:
            ValueError: If the component is not implemented.
        """
        if (
            (component == "body_joints")
            | (component == "hand_joints")
            | (component == "face_landmarks")
            | (component == "gaze_individual")
            | (component == "emotion_individual")
            | (component == "head_orientation")
        ):
            if is_3d:
                entity_path = f"{self.ROOT3D}/{component}/{alg_name}/{subject_name}"
            else:
                entity_path = (
                    f"{self.IMAGES_ROOT}/{cam_name}/{component}/{alg_name}/"
                    f"{subject_name}"
                )
        elif component == "proximity":
            if is_3d:
                entity_path = f"{self.ROOT3D}/{component}/{alg_name}"
            else:
                entity_path = f"{self.IMAGES_ROOT}/{cam_name}/{component}/{alg_name}"
        elif component == "kinematics":
            if is_3d:
                entity_path = f"{alg_name}_{subject_name}/{bodypart}"
            else:
                entity_path = f"{alg_name}_{subject_name}_{cam_name}/{bodypart}"

        else:
            raise ValueError(
                f"ERROR in generate_component_entity_path(): Component {component} "
                "did not implemented"
            )

        return entity_path

    def log_camera(self, camera_calibration, entity_path) -> None:
        """
        Logs the camera calibration in the viewer.

        Args:
            camera_calibration (dict): The camera calibration parameters.
            entity_path (str): The entity path for the camera.
        """
        # intrinsic camera matrix
        K = np.array(camera_calibration["intrinsic_matrix"])
        rr.log(
            entity_path,
            rr.Pinhole(
                resolution=camera_calibration["image_size"],
                image_from_camera=K[:3, :3].flatten(),
            ),
        )
        translation = np.array(camera_calibration["translation"]).reshape(3)
        R = camera_calibration["rotation_matrix"]
        R_inv = np.linalg.inv(R)
        center = np.matmul(R_inv, -translation)
        # center *= 1000
        rr.log(entity_path, rr.Transform3D(mat3x3=R_inv, translation=center.flatten()))

    def log_image(
        self, image: np.array, entity_path: str, img_quality: int = 75
    ) -> None:
        """
        Logs an image in the viewer.

        Args:
            image (np.array): The image to log.
            entity_path (str): The entity path for the image.
            img_quality (int, optional): The quality of the image. Defaults to 75.
        """
        rr.log(entity_path, rr.Image(image).compress(jpeg_quality=img_quality))

    def check_multiview(self) -> None:
        """
        Checks the consistency of the multi-view parameter in the visualizer config.

        Raises:
            ValueError: If the multi-view parameter is set to False but a 3D canvas is
                present.
        """
        if (self.visualizer_config["media"]["multi_view"] is False) and (
            "3D_Canvas" in self.canvas_list
        ):
            raise ValueError(
                "ERROR: multi-view parameter in Visualizer_config set false,\n "
                "But 3D_Canvas found in components, canvas lists.\n"
                "If you don't have multiple cameras, delete 3D_Canvas in all "
                "canvases\nIf you have multiple cameras, change multi-view "
                "parameter as true\n"
            )
