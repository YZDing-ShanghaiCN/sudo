"""Tool manager: per-robot tool meshes attached to URDF links via FK."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import viser
from viser.transforms import SO3

from .config import RobotConfig, ToolConfig


def _wxyz_to_matrix(wxyz: np.ndarray, position: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = SO3(wxyz).as_matrix()
    T[:3, 3] = position
    return T


class ToolManager:
    """Manages one robot's mounted tool meshes: load + track URDF link pose each tick."""

    def __init__(
        self,
        server: viser.ViserServer,
        robot_cfg: RobotConfig,
        robot_controller,
        get_base_pose: Callable[[], Optional[np.ndarray]],
    ):
        self.server = server
        self.robot = robot_controller
        self._get_base_pose = get_base_pose
        self._robot_name = robot_cfg.name

        self._tools: List[Tuple[str, ToolConfig]] = [
            (eef.name, eef.tool)
            for eef in robot_cfg.end_effectors
            if eef.tool is not None
        ]
        self._handles: Dict[str, viser.MeshHandle] = {}

        for eef_name, tool in self._tools:
            try:
                mesh = trimesh.load(tool.mesh_path)
            except Exception as e:
                print(f"[ToolManager] failed to load {tool.mesh_path}: {e}")
                continue
            if tool.scale != 1.0:
                mesh.apply_scale(tool.scale)
            handle = self.server.scene.add_mesh_trimesh(
                f"/robots/{self._robot_name}/tools/{eef_name}",
                mesh,
            )
            self._handles[eef_name] = handle

        self.update_poses()

    def update_poses(self):
        T_world_base = self._get_base_pose()
        if T_world_base is None:
            T_world_base = np.eye(4, dtype=np.float64)
        self.robot.update_kin()
        for eef_name, tool in self._tools:
            handle = self._handles.get(eef_name)
            if handle is None:
                continue
            T_base_link = self.robot.get_link_pose(tool.mount_link)
            T_link_tool = _wxyz_to_matrix(
                np.array(tool.wxyz, dtype=np.float64),
                np.array(tool.position, dtype=np.float64),
            )
            T_world_tool = T_world_base @ T_base_link @ T_link_tool
            handle.position = T_world_tool[:3, 3]
            handle.wxyz = SO3.from_matrix(T_world_tool[:3, :3]).wxyz
