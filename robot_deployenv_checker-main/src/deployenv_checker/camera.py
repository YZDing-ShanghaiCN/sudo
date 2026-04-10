"""Camera manager: frustum display, world pose computation, on-demand rendering."""

import math
from typing import Dict, List, Optional

import numpy as np
import viser
from viser.transforms import SO3, SE3

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


# Group cameras by mount location for UI organization
CAMERA_GROUPS = {
    "Chest": ["chest_left_camera", "chest_right_camera"],
    "Left Hand": ["left_hand_left_camera", "left_hand_right_camera", "left_hand_center_camera"],
    "Right Hand": ["right_hand_left_camera", "right_hand_right_camera", "right_hand_center_camera"],
}


class CameraManager:
    """Manages robot-mounted cameras: frustum visualization and on-demand rendering."""

    def __init__(
        self,
        server: viser.ViserServer,
        camera_system_cfg: CameraSystemConfig,
        robot_controller,
    ):
        self.server = server
        self.cfg = camera_system_cfg
        self.robot = robot_controller

        # Load camera configs
        self.cameras: List[CameraConfig] = load_cameras(camera_system_cfg.config_path)
        self.camera_map: Dict[str, CameraConfig] = {c.name: c for c in self.cameras}

        # Viser frustum handles
        self.frustum_handles: Dict[str, viser.CameraFrustumHandle] = {}

        # Current world poses (updated after each robot move)
        self.world_poses: Dict[str, np.ndarray] = {}

        if camera_system_cfg.show_frustums:
            self._create_frustums()

    def _create_frustums(self):
        """Create camera frustum visualizations in the scene."""
        for cam in self.cameras:
            # Compute initial world pose
            T_world_cam = self._compute_world_pose(cam)
            self.world_poses[cam.name] = T_world_cam

            # Extract position and quaternion from 4x4 matrix
            position = T_world_cam[:3, 3]
            wxyz = SO3.from_matrix(T_world_cam[:3, :3]).wxyz

            handle = self.server.scene.add_camera_frustum(
                name=f"/cameras/{cam.name}",
                fov=cam.fov_y,
                aspect=cam.width / cam.height,
                scale=self.cfg.frustum_scale,
                wxyz=wxyz,
                position=position,
                color=(100, 180, 255),
            )
            self.frustum_handles[cam.name] = handle

    def _compute_world_pose(self, cam: CameraConfig) -> np.ndarray:
        """Compute camera world pose: T_world_cam = T_world_link @ T_link_cam.

        Extrinsics are in ROS body convention; right-multiply by _R_ROS_TO_CV
        to convert to OpenCV/viser convention before composing with FK.
        """
        T_world_link = self.robot.get_link_pose(cam.mount)
        T_link_cam = cam.extrinsics @ _R_ROS_TO_CV
        return T_world_link @ T_link_cam

    def update_frustums(self):
        """Recompute all camera world poses and update frustum positions.

        Call this after each robot state change (IK solve).
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
        """Render a snapshot from the given camera's current viewpoint.

        Uses Viser's client.get_render() with the camera's FoV and world pose.
        Returns an RGB numpy array (H, W, 3).
        """
        cam = self.camera_map.get(name)
        if cam is None:
            return None

        T_world_cam = self.world_poses.get(name)
        if T_world_cam is None:
            return None

        position = T_world_cam[:3, 3]
        wxyz = SO3.from_matrix(T_world_cam[:3, :3]).wxyz

        # Render at reduced resolution for speed
        render_h = cam.height // 2
        render_w = cam.width // 2

        render = client.get_render(
            height=render_h,
            width=render_w,
            wxyz=wxyz,
            position=position,
            fov=cam.fov_y,
        )
        return render

    def set_frustums_visible(self, visible: bool):
        for handle in self.frustum_handles.values():
            handle.visible = visible

    def get_camera_groups(self) -> Dict[str, List[str]]:
        """Return camera names grouped by mount location."""
        return CAMERA_GROUPS
