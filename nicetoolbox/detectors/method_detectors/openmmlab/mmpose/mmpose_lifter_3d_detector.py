import pickle

import numpy as np

from nicetoolbox_core.data.array_schema import BBOX_2D_CONF, VECTOR_2D_CONF_PER_LABEL, VECTOR_3D_CONF_PER_LABEL
from nicetoolbox_core.data.loaded_array import NpzArray
from nicetoolbox_core.entrypoint import SubprocessResult
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from .....configs.schemas.detectors_instances_configs import MMPoseLifter3DAlgorithmConfig
from ....detector_inputs import NpzDetectorInput
from ....detector_outputs import DetectorOutput, NpzDetectorOutput
from ....utils.draw_2d import RenderContext
from ....utils.draw_3d import draw_skeleton_3d
from ...base_method import BaseMethod


# TODO: supports only motionbert!
class MMPoseLifter3D(BaseMethod):
    algorithm_type = "mmpose_lifter_3d"
    components = ["body_joints_local"]

    inputs = [
        NpzDetectorInput("body_joints", "pose_2d", schema=VECTOR_2D_CONF_PER_LABEL),
        NpzDetectorInput("person_bounding_box", "person_bboxes", schema=BBOX_2D_CONF),
    ]
    outputs = [
        NpzDetectorOutput("body_joints_local", "3d", schema=VECTOR_3D_CONF_PER_LABEL),
    ]

    def _initialize_detector(self) -> MMPoseLifter3DAlgorithmConfig.RuntimeConfig:
        # restrict input to coco only
        pose_2d = self.loaded_inputs["pose_2d"]
        upstream_cfg = pose_2d.upstream_config
        if upstream_cfg.keypoint_mapping != "coco_wholebody":
            raise NotImplementedError(
                f"mmpose_lifter_3d currently only supports upstream keypoint_mapping='coco_wholebody', "
                f"got {upstream_cfg.keypoint_mapping!r}."
            )
        person_bboxes = self.loaded_inputs["person_bboxes"]

        self.subjects = self.data.subjects_descr
        self.frame_names = self.data.frame_labels
        self.cameras = self.detector_config.camera_names
        self.mapping = getattr(self.predictions_mapping.human_pose, self.detector_config.keypoint_mapping)

        # visualization context
        self.render_context = RenderContext(
            dataloader=ImagePathsByFrameIndexLoader(
                config=self.data.get_input_recipes(), expected_cameras=self.cameras
            ),
            fps=self.data.fps,
            video_start=self.data.video_start_frame_index,
            save_frames=self.save_frames,
        )

        base_runtime = super()._initialize_detector()
        return MMPoseLifter3DAlgorithmConfig.RuntimeConfig(
            **base_runtime.model_dump(),
            pose_2d_npz=str(pose_2d.file_path),
            pose_2d_npz_key=pose_2d.npz_key,
            person_bbox_npz=str(person_bboxes.file_path),
            person_bbox_npz_key=person_bboxes.npz_key,
        )

    def post_inference(self, result_manifest: SubprocessResult) -> DetectorOutput:
        with open(result_manifest.get_raw_output(), "rb") as raw_file:
            per_camera_outputs = pickle.load(raw_file)

        labels = self._resolve_target_body_labels()
        axes = VECTOR_3D_CONF_PER_LABEL.make_axes(self.subjects, self.cameras, self.frame_names, labels=labels)
        pose_3d = np.full(axes.shape, np.nan, dtype=float)

        for cam_idx, cam_name in enumerate(self.cameras):
            for frame_idx, persons in enumerate(per_camera_outputs[cam_name]):
                for person in persons:
                    s = person["subject_idx"]
                    kp = np.asarray(person["keypoints_3d"])
                    sc = np.asarray(person["score"])
                    pose_3d[s, cam_idx, frame_idx, :, :3] = kp
                    pose_3d[s, cam_idx, frame_idx, :, 3] = sc

        out = DetectorOutput()
        out.add_array("body_joints_local", "3d", NpzArray(pose_3d, axes))
        return out

    def visualization(self, out: DetectorOutput) -> None:
        connections = self.mapping.connections.body_joints
        pose_3d = out.get("body_joints_local", "3d")
        draw_skeleton_3d(pose_3d, connections, self.render_context, self.viz_folders["body_joints_local"])

    def _resolve_target_body_labels(self) -> list[str]:
        """Ordered label names for the lifter output — MotionBERT always outputs h36m body joints."""
        body_dict: dict[str, int] = self.mapping.keypoints_index.body
        return [name for name, _ in sorted(body_dict.items(), key=lambda kv: kv[1])]
