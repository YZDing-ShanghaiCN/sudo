"""Scene manager: config-driven Viser scene with N robots, draggable bases, and objects."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf
from viser.transforms import SO3
from yourdfpy import URDF

from .config import RobotConfig, SceneConfig


def _wxyz_to_matrix(wxyz: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Compose a 4x4 transform from wxyz quaternion + translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = SO3(wxyz).as_matrix()
    T[:3, 3] = position
    return T


class SceneManager:
    """Config-driven Viser scene supporting multiple robots with draggable bases."""

    def __init__(self, server: viser.ViserServer, config: SceneConfig):
        self.server = server
        self.config = config

        # Per-robot scene state, keyed by robot name
        self.base_handles: Dict[str, viser.TransformControlsHandle] = {}
        self.eef_handles: Dict[Tuple[str, str], viser.TransformControlsHandle] = {}
        self.robot_visuals: Dict[str, ViserUrdf] = {}
        self.robot_collisions: Dict[str, ViserUrdf] = {}

        # Cached poses
        self._eef_poses: Dict[Tuple[str, str], np.ndarray] = {}  # local to robot base
        self._base_poses: Dict[str, np.ndarray] = {}  # 4x4 world-from-base
        self._base_update_cbs: Dict[str, List[Callable]] = {}

        # Object state (world-frame)
        self.object_handles: Dict[str, viser.TransformControlsHandle] = {}
        self._object_poses: Dict[str, np.ndarray] = {}
        self._object_update_cbs: Dict[str, list] = {}

        self.wall_handle = None

    # ---- setup ----

    def setup_scene(self):
        """Build the full scene from config."""
        for robot_cfg in self.config.robots:
            self._setup_robot(robot_cfg)
        self._setup_objects()
        if self.config.workspace.show_bounds:
            self._setup_workspace_bounds()
        self._setup_grid()

    def _setup_robot(self, cfg: RobotConfig):
        """Add one robot: draggable base gizmo with URDFs and EEF gizmos parented to it."""
        base_path = f"/robots/{cfg.name}/base"

        # Draggable base gizmo (rotations + sliders + axes ENABLED).
        # Use base_gizmo_scale rather than cfg.scale so the handle is large
        # enough to grab without affecting URDF/EEF visual scale.
        base_handle = self.server.scene.add_transform_controls(
            name=base_path,
            position=np.array(cfg.position, dtype=np.float64),
            wxyz=np.array(cfg.wxyz, dtype=np.float64),
            scale=cfg.base_gizmo_scale,
            line_width=4.0,
        )
        self.base_handles[cfg.name] = base_handle
        self._base_poses[cfg.name] = _wxyz_to_matrix(
            base_handle.wxyz, base_handle.position
        )

        # Cache base pose + fire callbacks on drag
        def _on_base_update(name=cfg.name):
            h = self.base_handles[name]
            self._base_poses[name] = _wxyz_to_matrix(h.wxyz, h.position)
            for cb in self._base_update_cbs.get(name, []):
                cb(h.position, h.wxyz)

        @base_handle.on_update
        def _(_):
            _on_base_update()

        # URDFs as children of the base — Viser composes the base transform automatically.
        self.robot_visuals[cfg.name] = ViserUrdf(
            self.server,
            URDF.load(cfg.urdf_visual),
            root_node_name=f"{base_path}/visual",
        )
        self.robot_collisions[cfg.name] = ViserUrdf(
            self.server,
            URDF.load(cfg.urdf_collision),
            root_node_name=f"{base_path}/collision",
        )
        self.robot_collisions[cfg.name].show_visual = False

        # EEF gizmos as children of the base — gizmo pose is reported in robot-local frame,
        # which is exactly what the IK consumes.
        for eef in cfg.end_effectors:
            eef_path = f"{base_path}/{eef.name}"
            handle = self.server.scene.add_transform_controls(
                name=eef_path,
                position=np.array(eef.position, dtype=np.float64),
                wxyz=np.array(eef.wxyz, dtype=np.float64),
                scale=eef.scale,
            )
            self.eef_handles[(cfg.name, eef.name)] = handle

            self._eef_poses[(cfg.name, eef.name)] = np.hstack(
                [np.array(eef.wxyz)[[1, 2, 3, 0]], np.array(eef.position)]
            )

            def _make_eef_cb(robot_name, eef_name, h):
                @h.on_update
                def _(_):
                    self._eef_poses[(robot_name, eef_name)] = np.hstack(
                        [h.wxyz[[1, 2, 3, 0]], h.position]
                    )

            _make_eef_cb(cfg.name, eef.name, handle)

            # Gripper mesh as child of the gizmo
            try:
                mesh = trimesh.load(eef.mesh_path)
                self.server.scene.add_mesh_trimesh(f"{eef_path}/mesh", mesh)
            except Exception:
                pass

    def _setup_objects(self):
        """Spawn environment objects with gizmos (world-frame)."""
        for obj in self.config.objects:
            handle = self.server.scene.add_transform_controls(
                name=obj.name,
                position=np.array(obj.position),
                wxyz=np.array(obj.wxyz),
                scale=obj.axis_size,
            )
            if not obj.draggable:
                handle.disable_rotations = True
                handle.disable_sliders = True
                handle.disable_axes = True

            try:
                mesh = trimesh.load(obj.mesh_path)
                self.server.scene.add_mesh_trimesh(f"{obj.name}/mesh", mesh)
            except Exception:
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

            self._object_poses[obj.name] = np.hstack(
                [np.array(obj.wxyz)[[1, 2, 3, 0]], np.array(obj.position)]
            )

            def make_obj_update_cb(name, handle):
                @handle.on_update
                def _(_):
                    h = self.object_handles[name]
                    self._object_poses[name] = np.hstack([h.wxyz[[1, 2, 3, 0]], h.position])
                    for cb in self._object_update_cbs.get(name, []):
                        cb(h.position, h.wxyz)

            make_obj_update_cb(obj.name, handle)

    def _setup_workspace_bounds(self):
        bounds = self.config.workspace.bounds
        self.wall_handle = self._add_bounding_box("/workspace_bounds", bounds)

    def _setup_grid(self):
        self.server.scene.add_grid(
            "ground_grid", 2, 2, cell_size=0.1,
            section_color=(0, 0, 0), cell_color=(0, 0, 0),
        )

    # ---- per-robot updates ----

    def update_robot_state(self, robot_name: str, q_viz: np.ndarray):
        """Update one robot's URDF visualization."""
        rv = self.robot_visuals.get(robot_name)
        if rv is not None:
            rv.update_cfg(q_viz)
        rc = self.robot_collisions.get(robot_name)
        if rc is not None and rc.show_visual:
            rc.update_cfg(q_viz)

    # ---- queries ----

    def get_eef_pose(self, robot_name: str, eef_name: str) -> Optional[np.ndarray]:
        """Return EEF gizmo pose in robot-local frame as [qx, qy, qz, qw, tx, ty, tz]."""
        return self._eef_poses.get((robot_name, eef_name))

    def get_base_pose(self, robot_name: str) -> Optional[np.ndarray]:
        """Return 4x4 world-from-base transform of the robot's base gizmo."""
        return self._base_poses.get(robot_name)

    def register_base_update_cb(self, robot_name: str, cb: Callable):
        """Register a callback fired when the named robot's base gizmo is dragged.

        cb(position: np.ndarray, wxyz: np.ndarray)
        """
        self._base_update_cbs.setdefault(robot_name, []).append(cb)

    def get_object_pose(self, name: str) -> Optional[np.ndarray]:
        return self._object_poses.get(name)

    def register_object_update_cb(self, name: str, cb):
        self._object_update_cbs.setdefault(name, []).append(cb)

    # ---- workspace bounds ----

    def update_wall(self, bounds_flat):
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
