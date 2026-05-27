"""Camera manager: per-robot frustum display and on-demand rendering."""

from typing import Callable, Dict, List, Optional

import numpy as np
import viser
from viser.transforms import SO3

from .config import CameraConfig, CameraSystemConfig, load_cameras

# Extrinsics are stored in ROS body convention (X fwd, Y left, Z up).
# Right-multiply by this to get OpenCV convention (Z fwd, X right, Y down),
# which is what viser uses for camera frustums and rendering.
_R_ROS_TO_CV = np.array([
    [0,  0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0,  0, 0, 1],
], dtype=np.float64)


class CameraManager:
    """Manages one robot's mounted cameras: frustum visualization + on-demand rendering."""

    def __init__(
        self,
        server: viser.ViserServer,
        camera_system_cfg: CameraSystemConfig,
        robot_controller,
        robot_name: str,
        get_base_pose: Callable[[], Optional[np.ndarray]],
    ):
        self.server = server
        self.cfg = camera_system_cfg
        self.robot = robot_controller
        self.robot_name = robot_name
        self._get_base_pose = get_base_pose

        self.cameras: List[CameraConfig] = load_cameras(camera_system_cfg.config_path)
        self.camera_map: Dict[str, CameraConfig] = {c.name: c for c in self.cameras}

        self.frustum_handles: Dict[str, viser.CameraFrustumHandle] = {}
        self.world_poses: Dict[str, np.ndarray] = {}

        if camera_system_cfg.show_frustums:
            self._create_frustums()

    def _create_frustums(self):
        for cam in self.cameras:
            T_world_cam = self._compute_world_pose(cam)
            self.world_poses[cam.name] = T_world_cam

            position = T_world_cam[:3, 3]
            wxyz = SO3.from_matrix(T_world_cam[:3, :3]).wxyz

            handle = self.server.scene.add_camera_frustum(
                name=f"/robots/{self.robot_name}/cameras/{cam.name}",
                fov=cam.fov_y,
                aspect=cam.width / cam.height,
                scale=self.cfg.frustum_scale,
                wxyz=wxyz,
                position=position,
                color=(100, 180, 255),
            )
            self.frustum_handles[cam.name] = handle

    def _compute_world_pose(self, cam: CameraConfig) -> np.ndarray:
        """T_world_cam = T_world_base @ T_base_link(URDF FK) @ T_link_cam(extrinsics)."""
        T_world_base = self._get_base_pose()
        if T_world_base is None:
            T_world_base = np.eye(4, dtype=np.float64)
        T_base_link = self.robot.get_link_pose(cam.mount)
        T_link_cam = cam.extrinsics @ _R_ROS_TO_CV
        return T_world_base @ T_base_link @ T_link_cam

    def update_frustums(self):
        """Recompute all camera world poses and update frustum positions.

        Call after each robot state change (IK solve) or base move.
        """
        self.robot.update_kin()
        for cam in self.cameras:
            T_world_cam = self._compute_world_pose(cam)
            self.world_poses[cam.name] = T_world_cam

            if cam.name in self.frustum_handles:
                handle = self.frustum_handles[cam.name]
                handle.position = T_world_cam[:3, 3]
                handle.wxyz = SO3.from_matrix(T_world_cam[:3, :3]).wxyz

    def render_camera(
        self, client: viser.ClientHandle, name: str
    ) -> Optional[np.ndarray]:
        """Render a snapshot from the named camera's current world viewpoint."""
        cam = self.camera_map.get(name)
        if cam is None:
            return None

        T_world_cam = self.world_poses.get(name)
        if T_world_cam is None:
            return None

        position = T_world_cam[:3, 3]
        wxyz = SO3.from_matrix(T_world_cam[:3, :3]).wxyz

        render_h = cam.height // 2
        render_w = cam.width // 2

        return client.get_render(
            height=render_h,
            width=render_w,
            wxyz=wxyz,
            position=position,
            fov=cam.fov_y,
        )

    def set_frustums_visible(self, visible: bool):
        for handle in self.frustum_handles.values():
            handle.visible = visible
