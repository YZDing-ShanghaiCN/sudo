"""Main orchestrator: wires config -> scene -> robot -> cameras -> GUI."""

import time
import viser

from .config import load_config
from .scene import SceneManager
from .robot import RobotController
from .camera import CameraManager
from .gui import GuiBuilder


class App:
    """Deploy environment checker application."""

    def __init__(self, config_path: str, port: int = 8080):
        self.cfg = load_config(config_path)
        self.port = port
        self._robot = None
        self._scene = None
        self._camera_mgr = None
        self._gui = None

    def move_to_eef_targets(self):
        """Snap robot joints to match current EEF gizmo positions via direct IK."""
        if self._robot is None:
            return
        for eef in self.cfg.robot.end_effectors:
            pose = self._scene.get_eef_pose(eef.name)
            if pose is not None:
                self._robot.snap_to_eef(eef.frame, pose)

        self._robot.update_kin()
        self._scene.update_robot_state(self._robot.q_viz())
        self._robot.update_col()
        self._gui.update_collision_status(self._robot.check_collision())
        self._camera_mgr.update_frustums()

    def run(self):
        server = viser.ViserServer(port=self.port)

        self._robot = RobotController(self.cfg.robot)
        self._scene = SceneManager(server, self.cfg)
        self._scene.setup_scene()
        self._camera_mgr = CameraManager(server, self.cfg.cameras, self._robot)
        self._gui = GuiBuilder(server, self._robot, self._camera_mgr, self._scene, self)

        self._scene.update_robot_state(self._robot.q_viz())
        self._camera_mgr.update_frustums()
        self._robot.update_kin()
        self._robot.update_col()
        self._gui.update_collision_status(self._robot.check_collision())

        print(f"Server running at http://localhost:{self.port}")

        while True:
            self._scene.update_robot_state(self._robot.q_viz())
            self._robot.update_kin()
            self._robot.update_col()
            self._gui.update_collision_status(self._robot.check_collision())
            time.sleep(10e-3)
