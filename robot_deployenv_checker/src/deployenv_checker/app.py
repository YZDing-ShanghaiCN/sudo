"""Main orchestrator: wires config -> scene -> N robots -> per-robot cameras -> GUI."""

import json
import time
from typing import Dict, Optional, Tuple

import numpy as np
import viser

from .camera import CameraManager
from .config import load_config
from .gui import GuiBuilder
from .robot import RobotController
from .scene import SceneManager
from .tool import ToolManager


class App:
    """Deploy environment checker application supporting multiple robots."""

    def __init__(self, config_path: str, port: int = 8080):
        self.cfg = load_config(config_path)
        self.port = port
        self._scene: SceneManager = None
        self._robots: Dict[str, RobotController] = {}
        self._camera_mgrs: Dict[str, CameraManager] = {}
        self._tool_mgrs: Dict[str, ToolManager] = {}
        self._gui: GuiBuilder = None

        # Trajectory replay state (drives the first robot in self.cfg.robots)
        self._traj_q: Optional[np.ndarray] = None      # shape (N, 16)
        self._traj_t: Optional[np.ndarray] = None      # shape (N,) seconds
        self._traj_playing: bool = False
        self._traj_play_started_wall: float = 0.0
        self._traj_play_started_t: float = 0.0
        self._traj_idx: int = 0
        self._traj_speed: float = 1.0

    def move_to_eef_targets(self, robot_name: str):
        """Snap one robot's joints to match its current EEF gizmo positions via direct IK."""
        robot = self._robots.get(robot_name)
        if robot is None:
            return
        robot_cfg = next(
            (r for r in self.cfg.robots if r.name == robot_name), None
        )
        if robot_cfg is None:
            return

        for eef in robot_cfg.end_effectors:
            pose = self._scene.get_eef_pose(robot_name, eef.name)
            if pose is not None:
                robot.snap_to_eef(eef.frame, pose)

        robot.update_kin()
        self._scene.update_robot_state(robot_name, robot.q_viz())
        robot.update_col()
        self._gui.update_collision_status(robot_name, robot.check_collision())
        cam_mgr = self._camera_mgrs.get(robot_name)
        if cam_mgr is not None:
            cam_mgr.update_frustums()
        tool_mgr = self._tool_mgrs.get(robot_name)
        if tool_mgr is not None:
            tool_mgr.update_poses()

    def _target_robot(self) -> Optional[RobotController]:
        if not self.cfg.robots:
            return None
        return self._robots.get(self.cfg.robots[0].name)

    def load_trajectory_json(self, raw: bytes) -> Tuple[bool, str]:
        """Parse a trajectory JSON and snap the target robot to frame 0."""
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as e:
            return False, f"Invalid JSON: {e}"

        traj = data.get("q_trajectory")
        if not isinstance(traj, list) or not traj:
            return False, "JSON has no non-empty 'q_trajectory' list."

        # Drop frames where IK failed (q_full is null in the source JSON).
        usable = [wp for wp in traj if wp.get("q_full") is not None]
        dropped = len(traj) - len(usable)
        if not usable:
            return False, "All waypoints have null q_full (no replayable frames)."

        try:
            q_arr = np.array([wp["q_full"] for wp in usable], dtype=np.float64)
            t_arr = np.array([wp["t_global"] for wp in usable], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as e:
            return False, f"Bad waypoint shape: {e}"

        if q_arr.ndim != 2 or q_arr.shape[1] != 16:
            return False, f"q_full must be length 16; got shape {q_arr.shape}."
        if np.any(np.diff(t_arr) < 0):
            return False, "t_global is not monotonically non-decreasing."

        self._traj_playing = False
        self._traj_q = q_arr
        self._traj_t = t_arr
        self._traj_idx = 0

        robot = self._target_robot()
        if robot is not None:
            robot.agent.q[:] = self._traj_q[0]

        suffix = f" (dropped {dropped} failed frame{'s' if dropped != 1 else ''})" if dropped else ""
        return True, f"Loaded {len(t_arr)} frames, duration {t_arr[-1]:.2f}s{suffix}"

    def play_trajectory(self):
        if self._traj_q is None or self._traj_t is None:
            return
        if self._traj_idx >= len(self._traj_t) - 1:
            self._traj_idx = 0
        self._traj_play_started_wall = time.time()
        self._traj_play_started_t = float(self._traj_t[self._traj_idx])
        self._traj_playing = True

    def pause_trajectory(self):
        self._traj_playing = False

    def stop_trajectory(self):
        self._traj_playing = False
        if self._traj_q is None:
            return
        self._traj_idx = 0
        robot = self._target_robot()
        if robot is not None:
            robot.agent.q[:] = self._traj_q[0]

    def set_trajectory_speed(self, speed: float):
        """Update playback speed; re-anchor mid-play so the current frame doesn't jump."""
        speed = max(float(speed), 1e-3)
        if self._traj_playing and self._traj_t is not None:
            now = time.time()
            current_t = self._traj_play_started_t + (now - self._traj_play_started_wall) * self._traj_speed
            self._traj_play_started_wall = now
            self._traj_play_started_t = current_t
        self._traj_speed = speed

    def _advance_trajectory(self):
        if not self._traj_playing or self._traj_q is None or self._traj_t is None:
            return
        elapsed = time.time() - self._traj_play_started_wall
        target_t = self._traj_play_started_t + elapsed * self._traj_speed
        n = len(self._traj_t)
        while self._traj_idx + 1 < n and self._traj_t[self._traj_idx + 1] <= target_t:
            self._traj_idx += 1
        robot = self._target_robot()
        if robot is not None:
            robot.agent.q[:] = self._traj_q[self._traj_idx]
        if self._traj_idx >= n - 1:
            self._traj_playing = False
        if self._gui is not None:
            self._gui.update_trajectory_status(
                self._traj_idx, n, float(self._traj_t[self._traj_idx])
            )

    def run(self):
        server = viser.ViserServer(port=self.port)

        # Scene first — needed for base-pose lookups
        self._scene = SceneManager(server, self.cfg)
        self._scene.setup_scene()

        # One controller + one camera manager per robot
        for robot_cfg in self.cfg.robots:
            controller = RobotController(robot_cfg)
            self._robots[robot_cfg.name] = controller

            if robot_cfg.cameras is not None:
                name = robot_cfg.name  # capture for closure
                cam_mgr = CameraManager(
                    server,
                    robot_cfg.cameras,
                    controller,
                    robot_name=name,
                    get_base_pose=lambda n=name: self._scene.get_base_pose(n),
                )
                self._camera_mgrs[name] = cam_mgr

                # Refresh frustum world poses when the base gizmo is dragged
                self._scene.register_base_update_cb(
                    name, lambda *_args, m=cam_mgr: m.update_frustums()
                )

            if any(eef.tool is not None for eef in robot_cfg.end_effectors):
                name = robot_cfg.name
                tool_mgr = ToolManager(
                    server,
                    robot_cfg,
                    controller,
                    get_base_pose=lambda n=name: self._scene.get_base_pose(n),
                )
                self._tool_mgrs[name] = tool_mgr
                self._scene.register_base_update_cb(
                    name, lambda *_args, m=tool_mgr: m.update_poses()
                )

        self._gui = GuiBuilder(
            server, self._robots, self._camera_mgrs, self._scene, app=self
        )

        # Initial state push
        for name, robot in self._robots.items():
            self._scene.update_robot_state(name, robot.q_viz())
            robot.update_kin()
            robot.update_col()
            self._gui.update_collision_status(name, robot.check_collision())
            cam_mgr = self._camera_mgrs.get(name)
            if cam_mgr is not None:
                cam_mgr.update_frustums()
            tool_mgr = self._tool_mgrs.get(name)
            if tool_mgr is not None:
                tool_mgr.update_poses()

        print(f"Server running at http://localhost:{self.port}")

        while True:
            self._advance_trajectory()
            for name, robot in self._robots.items():
                self._scene.update_robot_state(name, robot.q_viz())
                robot.update_kin()
                robot.update_col()
                self._gui.update_collision_status(name, robot.check_collision())
                tool_mgr = self._tool_mgrs.get(name)
                if tool_mgr is not None:
                    tool_mgr.update_poses()
            time.sleep(10e-3)
