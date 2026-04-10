"""Scene manager: config-driven Viser scene with objects and gizmos."""

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF
from typing import Dict, Optional

from .config import SceneConfig, ObjectConfig, RobotConfig


class SceneManager:
    """Config-driven scene manager built on Viser."""

    def __init__(self, server: viser.ViserServer, config: SceneConfig):
        self.server = server
        self.config = config
        self.object_handles: Dict[str, viser.TransformControlsHandle] = {}
        self._eef_poses: Dict[str, np.ndarray] = {}  # cached on each drag
        self._object_poses: Dict[str, np.ndarray] = {}  # cached on each drag
        self._object_update_cbs: Dict[str, list] = {}  # registered callbacks
        self.robot_visual: Optional[ViserUrdf] = None
        self.robot_collision: Optional[ViserUrdf] = None
        self.wall_handle = None

    def setup_scene(self):
        """Build the full scene from config."""
        self._setup_robot()
        self._setup_objects()
        if self.config.workspace.show_bounds:
            self._setup_workspace_bounds()
        self._setup_grid()

    def _setup_robot(self):
        """Load robot URDF into the scene."""
        cfg = self.config.robot

        # Add robot root transform
        robot_controls = self.server.scene.add_transform_controls(
            name="/robot",
            position=np.array(cfg.position),
            wxyz=np.array(cfg.wxyz),
            scale=cfg.scale,
            disable_rotations=True,
            disable_sliders=True,
            disable_axes=True,
        )

        # Visual URDF
        self.robot_visual = ViserUrdf(
            self.server,
            URDF.load(cfg.urdf_visual),
            root_node_name="/robot/visual",
        )

        # Collision URDF
        self.robot_collision = ViserUrdf(
            self.server,
            URDF.load(cfg.urdf_collision),
            root_node_name="/robot/collision",
        )
        self.robot_collision.show_visual = False

        # Spawn EEF gizmos
        for eef in cfg.end_effectors:
            handle = self.server.scene.add_transform_controls(
                name=f"/{eef.name}",
                position=np.array(eef.position),
                wxyz=np.array(eef.wxyz),
                scale=eef.scale,
            )
            self.object_handles[eef.name] = handle

            # Seed initial cached pose
            self._eef_poses[eef.name] = np.hstack(
                [np.array(eef.wxyz)[[1, 2, 3, 0]], np.array(eef.position)]
            )

            # Update cached pose whenever the gizmo is dragged in the browser
            def make_update_cb(name, handle):
                @handle.on_update
                def _(_):
                    h = self.object_handles[name]
                    self._eef_poses[name] = np.hstack([h.wxyz[[1, 2, 3, 0]], h.position])

            make_update_cb(eef.name, handle)

            # Load gripper mesh as child
            try:
                mesh = trimesh.load(eef.mesh_path)
                self.server.scene.add_mesh_trimesh(f"{eef.name}/mesh", mesh)
            except Exception:
                pass

    def _setup_objects(self):
        """Spawn environment objects with gizmos."""
        for obj in self.config.objects:
            handle = self.server.scene.add_transform_controls(
                name=obj.name,
                position=np.array(obj.position),
                wxyz=np.array(obj.wxyz),
                scale=obj.scale,
            )
            if not obj.draggable:
                handle.disable_rotations = True
                handle.disable_sliders = True
                handle.disable_axes = True

            try:
                mesh = trimesh.load(obj.mesh_path)
                self.server.scene.add_mesh_trimesh(f"{obj.name}/mesh", mesh)
            except Exception:
                # Placeholder cube
                self.server.scene.add_mesh_simple(
                    name=f"{obj.name}/mesh",
                    vertices=np.array([
                        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                    ]) * 0.1,
                    faces=np.array([
                        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                        [0, 4, 7], [0, 7, 3], [1, 5, 6], [1, 6, 2],
                        [0, 1, 5], [0, 5, 4], [3, 2, 6], [3, 6, 7],
                    ]),
                    color=(100, 200, 255),
                )

            self.object_handles[obj.name] = handle

            # Seed initial cached pose
            self._object_poses[obj.name] = np.hstack(
                [np.array(obj.wxyz)[[1, 2, 3, 0]], np.array(obj.position)]
            )

            # Update cached pose and notify registered callbacks on gizmo drag
            def make_obj_update_cb(name, handle):
                @handle.on_update
                def _(_):
                    h = self.object_handles[name]
                    self._object_poses[name] = np.hstack([h.wxyz[[1, 2, 3, 0]], h.position])
                    for cb in self._object_update_cbs.get(name, []):
                        cb(h.position, h.wxyz)

            make_obj_update_cb(obj.name, handle)

    def _setup_workspace_bounds(self):
        """Draw workspace bounding box."""
        bounds = self.config.workspace.bounds
        self.wall_handle = self._add_bounding_box("/workspace_bounds", bounds)

    def _setup_grid(self):
        """Add ground grid."""
        self.server.scene.add_grid(
            "ground_grid", 2, 2, cell_size=0.1,
            section_color=(0, 0, 0), cell_color=(0, 0, 0),
        )

    def update_robot_state(self, q_viz: np.ndarray):
        """Update robot visualization with new joint positions."""
        if self.robot_visual:
            self.robot_visual.update_cfg(q_viz)
        if self.robot_collision and self.robot_collision.show_visual:
            self.robot_collision.update_cfg(q_viz)

    def get_object_pose(self, name: str) -> Optional[np.ndarray]:
        """Get object gizmo pose as [qx, qy, qz, qw, tx, ty, tz]."""
        return self._object_poses.get(name)

    def register_object_update_cb(self, name: str, cb):
        """Register a callback invoked when the object gizmo is dragged.

        cb(position: np.ndarray, wxyz: np.ndarray)
        """
        self._object_update_cbs.setdefault(name, []).append(cb)

    def get_eef_pose(self, name: str) -> Optional[np.ndarray]:
        """Get EEF gizmo pose as [qx, qy, qz, qw, tx, ty, tz].

        Returns the pose cached by the on_update callback — updated on every drag.
        """
        return self._eef_poses.get(name)

    def update_wall(self, bounds_flat):
        """Update workspace bounds visualization."""
        if self.wall_handle is not None:
            self.server.scene.remove_by_name("/workspace_bounds")
        self.wall_handle = self._add_bounding_box("/workspace_bounds", bounds_flat)

    def _add_bounding_box(self, name, bounds, color=(255, 100, 100)):
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin],
            [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax],
            [xmax, ymax, zmax], [xmin, ymax, zmax],
        ])
        edge_indices = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        edges = corners[edge_indices]
        return self.server.scene.add_line_segments(
            name=name, points=edges, line_width=2.0, colors=color,
        )
