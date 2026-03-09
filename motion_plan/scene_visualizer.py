"""
Scene Visualizer for Motion Planning Results
This module visualizes the collision scene, planned trajectories, and robot states
using the viser 3D visualization framework.
"""

import numpy as np
 
from typing import Dict, Optional, List, Tuple, Any
import time

import viser
import ampl
import pyampl
from yourdfpy import URDF

from collision_motion_plan import CollisionMotionPlanner, PlanningConfig
from viz_viser import ViserObject


class ViserTrajectoryMarker:
    """Utility class for rendering trajectory waypoints in viser"""

    def __init__(self, server: viser.ViserServer, name: str = "trajectory"):
        self.server = server
        self.name = name
        self.markers: List[viser.SceneNodeHandle] = []

    def add_waypoint(
        self,
        position: np.ndarray,
        color: Tuple[int, int, int] = (255, 128, 0),
        radius: float = 0.02,
    ) -> None:
        """Add a single waypoint marker"""
        marker = self.server.scene.add_icosphere(
            f"/{self.name}/wp_{len(self.markers)}",
            radius=radius,
            color=color,
        )
        marker.position = position
        self.markers.append(marker)

    def add_line_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: Tuple[int, int, int] = (128, 200, 255),
    ) -> None:
        """Add a line segment between two points"""
        # Note: viser doesn't have add_line method in scene API
        # Line visualization is omitted; waypoint markers are sufficient
        pass

    def clear(self) -> None:
        """Clear all trajectory markers"""
        for marker in self.markers:
            marker.remove()
        self.markers.clear()


class SceneVisualizer:
    """
    Visualizer for motion planning scenes with trajectories and collision obstacles.
    Integrates with viser for interactive 3D visualization.
    """

    def __init__(
        self,
        assets_dir: str = "./assets",
        planner: Optional[CollisionMotionPlanner] = None,
        arm_config: Optional[Any] = None,
        arm_urdf: Optional[URDF] = None,
    ):
        """
        Initialize the scene visualizer.

        Args:
            assets_dir: Path to assets directory
            planner: Optional CollisionMotionPlanner instance for integration
            arm_config: Optional arm configuration (uses planner's arm_config if available)
        """
        self.assets_dir = assets_dir
        self.planner = planner or CollisionMotionPlanner(assets_dir=assets_dir)
        
        # Use provided arm_config or get from planner
        if arm_config is not None:
            self.arm_config = arm_config
        elif hasattr(self.planner, 'arm_config'):
            self.arm_config = self.planner.arm_config
        else:
            # Fallback to default
            self.arm_config = pyampl.create_default_arm_config("hillbot_left")
        
        self.server = viser.ViserServer()
        
        # Print server URL
        print(f"\n{'='*70}")
        print(f"Viser Server running at: {self.server.get_port()}")
        print(f"{'='*70}\n")

        # Initialize visualization components
        self.trajectory_markers: Optional[ViserTrajectoryMarker] = None
        self.arm_robot_urdf: Optional[URDF] = arm_urdf
        self.scene_objects: Dict[str, ViserObject] = {}
        self._animation_running: Optional[List[bool]] = None
        self._animation_thread: Optional[Any] = None

        # Setup basic scene
        self._setup_scene()

    def _setup_scene(self) -> None:
        """Setup basic scene elements like ground grid and camera"""
        # Add ground grid
        grid_handler = self.server.scene.add_grid(
            "/ground",
            width=2.0,
            height=2.0,
            cell_size=0.1,
            section_color=(100, 100, 100),
            cell_color=(200, 200, 200),
            cell_thickness=1,
            section_thickness=2,
        )

        # Setup camera
        @self.server.on_client_connect
        async def _(client: viser.ClientHandle) -> None:
            client.camera.position = (1.5, 1.5, 1.2)
            client.camera.look_at = (0.7, 0.0, 0.7)
            client.camera.fov = 0.05

        # Configure theme
        self.server.gui.configure_theme(
            control_width="large",
            show_logo=False,
            show_share_button=False,
        )

    def add_mesh_object(
        self,
        name: str,
        mesh: Any,
        position: np.ndarray = np.array([0, 0, 0]),
        color: Tuple[int, int, int, int] = (128, 128, 128, 255),
        scale: float = 1.0,
        visible: bool = True,
    ) -> Optional[ViserObject]:
        """
        Add a mesh object to the scene using ViserObject.

        Note: Visualization does not load files. Pass a preloaded mesh object.
        """
        if mesh is None:
            print(f"Warning: Mesh not provided for {name}")
            return None
        if isinstance(mesh, str):
            print("Warning: Visualization does not load meshes from file paths")
            return None

        try:
            viser_obj = ViserObject(
                mesh,
                name=name,
                server=self.server,
                color=color if len(color) == 4 else (*color, 255),
                no_control=True,
            )

            viser_obj.handler.position = position
            if hasattr(viser_obj.handler, "scale"):
                viser_obj.handler.scale = (scale, scale, scale)
            viser_obj.handler.visible = visible

            self.scene_objects[name] = viser_obj
            return viser_obj
        except Exception as e:
            print(f"Error adding mesh {name}: {e}")
            return None

    def add_cube_obstacle(
        self,
        name: str,
        size: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        position: np.ndarray = np.array([0.7, 0.0, 0.7]),
        color: Tuple[int, int, int, int] = (200, 100, 100, 255),
    ) -> Optional[ViserObject]:
        """Add a simple cube obstacle to the scene."""
        try:
            cube_mesh = trimesh.creation.box(extents=size)
            viser_obj = ViserObject(
                cube_mesh,
                name=name,
                server=self.server,
                color=color if len(color) == 4 else (*color, 255),
                no_control=True,
            )
            viser_obj.handler.position = position
            self.scene_objects[name] = viser_obj
            return viser_obj
        except Exception as e:
            print(f"Error creating cube obstacle {name}: {e}")
            return None

    def add_crate_obstacles(
        self,
        crate_poses: Dict[str, np.ndarray],
        meshes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add crate obstacles to the scene.

        Args:
            crate_poses: Dictionary mapping crate names to rwt poses
        """
        if meshes is None:
            print("Warning: Crate meshes not provided")
            return

        for crate_name, pose_rwt in crate_poses.items():
            position = pose_rwt[4:7]
            self.add_mesh_object(
                name=crate_name,
                mesh=meshes.get(crate_name),
                position=position,
                color=(128, 128, 255, 200),
            )

    def add_robot_arm(
        self,
        urdf: Optional[URDF] = None,
        pose_base: Optional[np.ndarray] = None,
        state_ref: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add robot arm URDF to the scene using ViserObject.
        Follows the same pattern as viz_pick_dev.py

        Args:
            config: Optional joint configuration to display
        """
        try:
            if urdf is not None:
                self.arm_robot_urdf = urdf
            if self.arm_robot_urdf is None:
                raise ValueError("Warning: URDF not provided for robot arm")
            
            viser_obj = ViserObject(
                self.arm_robot_urdf,
                name="arm",
                server=self.server,
            )
            
            # Set pose, disable control, and update configuration
            # Following the pattern from viz_pick_dev.py lines 239-241
            if pose_base is not None:
                viser_obj.set_control(pose_base)
            viser_obj.disable_control()
            if state_ref is not None:
                viser_obj.urdf.update_cfg(state_ref)
            
            self.scene_objects["robot_arm"] = viser_obj
        except Exception as e:
            raise ValueError(f"ERROR: Could not load URDF: {e}")

    def add_robot_torso(
        self,
        mesh: Any,
        position: np.ndarray = np.array([0, 0, 0]),
        color: Tuple[int, int, int, int] = (170, 170, 170, 255),
    ) -> None:
        """
        Add robot torso mesh to the scene using ViserObject.
        
        Args:
            mesh: The mesh object for the torso
            position: Position vector [x, y, z]
            color: Color tuple (r, g, b, a)
        """
        if mesh is None:
            print("Warning: Torso mesh not provided")
            return
        try:
            viser_obj = ViserObject(
                mesh,
                name="torso",
                server=self.server,
                color=color,
            )
            viser_obj.disable_control()
            self.scene_objects["torso"] = viser_obj
        except Exception as e:
            print(f"Warning: Could not load torso: {e}")

    def add_static_obstacles(self, meshes: Optional[Dict[str, Any]] = None) -> None:
        """Add static environment obstacles like desk."""
        if meshes is None:
            print("Warning: Static obstacle meshes not provided")
            return
        self.add_mesh_object(
            name="desk",
            mesh=meshes.get("desk"),
            position=np.array([0.7, 0.0, 0.7]),
            color=(128, 255, 128, 200),
        )

    def visualize_trajectory(
        self,
        trajectory: np.ndarray,
        show_waypoints: bool = True,
        show_lines: bool = True,
        color: Tuple[int, int, int] = (255, 128, 0),
    ) -> None:
        """
        Visualize a planned trajectory in the scene.

        Args:
            trajectory: Array of joint configurations (N x 7)
            show_waypoints: Show waypoint markers
            show_lines: Show connecting lines between waypoints
            color: Color for trajectory visualization
        """
        if self.trajectory_markers is not None:
            self.trajectory_markers.clear()

        self.trajectory_markers = ViserTrajectoryMarker(
            self.server, name="planned_trajectory"
        )

        # Get TCP positions for all waypoints
        tcp_positions = []
        for config in trajectory:
            self.planner.agent.fk_rwt = config
            tcp_pose = self.planner.agent.fk_rwt[-1]  # Last element is TCP rwt
            tcp_position = tcp_pose[4:7]  # Extract position from rwt
            tcp_positions.append(tcp_position)

        tcp_positions = np.array(tcp_positions)

        # Visualize waypoints
        if show_waypoints:
            for i, position in enumerate(tcp_positions):
                # Color gradient from start (green) to end (red)
                progress = i / len(tcp_positions)
                rgb_color = (
                    int(255 * progress),
                    int(255 * (1 - progress)),
                    100,
                )
                self.trajectory_markers.add_waypoint(position, color=rgb_color)

        # Visualize connecting lines
        if show_lines and len(tcp_positions) > 1:
            for i in range(len(tcp_positions) - 1):
                self.trajectory_markers.add_line_segment(
                    tcp_positions[i],
                    tcp_positions[i + 1],
                    color=(100, 150, 200),
                )

    def show_start_pose(
        self,
        config: np.ndarray,
        marker_name: str = "start_pose",
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """
        Display the start configuration marker.

        Args:
            config: Joint configuration
            marker_name: Name for the marker
            color: Color (default: green)
        """
        self.planner.agent.fk_rwt = config
        tcp_pose = self.planner.agent.fk_rwt[-1]
        tcp_position = tcp_pose[4:7]

        # Add larger marker for start pose
        marker = self.server.scene.add_icosphere(
            f"/{marker_name}",
            radius=0.01,
            color=color,
        )
        marker.position = tcp_position

        # Add label
        label = self.server.scene.add_label(
            f"/{marker_name}_label",
            text="Start",
            position=tcp_position + np.array([0.05, 0, 0]),
        )

        self.scene_objects[marker_name] = marker
        self.scene_objects[f"{marker_name}_label"] = label

    def show_end_pose(
        self,
        config: np.ndarray,
        marker_name: str = "end_pose",
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> None:
        """
        Display the end configuration marker.

        Args:
            config: Joint configuration
            marker_name: Name for the marker
            color: Color (default: red)
        """
        self.planner.agent.fk_rwt = config
        tcp_pose = self.planner.agent.fk_rwt[-1]
        tcp_position = tcp_pose[4:7]

        # Add larger marker for end pose
        marker = self.server.scene.add_icosphere(
            f"/{marker_name}",
            radius=0.001,
            color=color,
        )
        marker.position = tcp_position

        # Add label
        label = self.server.scene.add_label(
            f"/{marker_name}_label",
            text="Goal",
            position=tcp_position + np.array([0.05, 0, 0]),
        )

        self.scene_objects[marker_name] = marker
        self.scene_objects[f"{marker_name}_label"] = label

    def play_trajectory_animation(
        self,
        trajectory: np.ndarray,
        speed: float = 1.0,
        loop: bool = True,
    ) -> None:
        """
        Play trajectory animation with robot following the path.

        Args:
            trajectory: Array of joint configurations
            speed: Animation speed multiplier (>1 is faster)
            loop: Whether to loop the animation
        """
        if self.arm_robot_urdf is None:
            print("Robot arm not loaded. Call add_robot_arm() first.")
            return

        play_button = self.server.gui.add_button("Play Animation")
        stop_button = self.server.gui.add_button("Stop Animation")

        # Ensure animation flag list exists
        if self._animation_running is None:
            self._animation_running = [False]
        
        # Stop any previous animation
        self._animation_running[0] = False
        if self._animation_thread is not None and self._animation_thread.is_alive():
             self._animation_thread.join(timeout=0.2)

        def animate() -> None:
            frame_idx = 0
            while self._animation_running[0]:
                if loop:
                    frame_idx = frame_idx % len(trajectory)
                else:
                    if frame_idx >= len(trajectory):
                        self._animation_running[0] = False
                        break

                config = trajectory[frame_idx]
                if "robot_arm" in self.scene_objects:
                    self.scene_objects["robot_arm"].urdf.update_cfg(config)
                frame_idx += 1
                time.sleep(0.05 / speed)  # 20 FPS base rate

        def start_animation() -> None:
            if self._animation_thread is not None and self._animation_thread.is_alive():
                return
            
            self._animation_running[0] = True
            import threading
            self._animation_thread = threading.Thread(target=animate, daemon=True)
            self._animation_thread.start()

        @play_button.on_click
        def _(_) -> None:
            print("play button clicked")
            start_animation()

        @stop_button.on_click
        def _(_) -> None:
            self._animation_running[0] = False


    def stop_animations(self) -> None:
        """Stop any running animations."""
        if self._animation_running is not None:
            self._animation_running[0] = False

    def load_and_visualize_trajectory_file(
        self,
        json_path: str,
        crate_poses: Optional[Dict[str, np.ndarray]] = None,
    ) -> bool:
        """
        Load trajectory from JSON file and visualize it.

        Args:
            json_path: Path to trajectory JSON file
            crate_poses: Optional crate poses for scene setup

        Returns:
            True if successful, False otherwise
        """
        print("Visualization does not load files. Provide data directly instead.")
        return False

    def interactive_scene_with_gui(
        self,
        crate_poses: Optional[Dict[str, np.ndarray]] = None,
        meshes: Optional[Dict[str, Any]] = None,
        arm_urdf: Optional[URDF] = None,
    ) -> None:
        """
        Create an interactive scene with GUI controls for planning and visualization.

        Args:
            crate_poses: Initial crate poses
        """
        # Setup scene
        if crate_poses:
            self.add_crate_obstacles(crate_poses, meshes=meshes)
        else:
            # Default crate poses
            workbench_center = np.array([0.7, 0.0, 0.7])
            crate_center = workbench_center + np.array([0.0, 0.0, 0.01])
            crate_poses = {
                "crate_0": np.array([0.0, 0.0, 0.0, 1.0,
                                    crate_center[0], crate_center[1] - 0.2, crate_center[2]]),
                "crate_1": np.array([0.0, 0.0, 0.0, 1.0,
                                    crate_center[0], crate_center[1] + 0.2, crate_center[2]]),
            }
            self.add_crate_obstacles(crate_poses, meshes=meshes)

        self.add_static_obstacles(meshes=meshes)
        if meshes is not None:
            self.add_robot_torso(meshes.get("torso"))
        self.add_robot_arm(
            urdf=arm_urdf,
            pose_base=self.planner.agent.pose_base,
            state_ref=self.planner.state_from,
        )

        # GUI Controls
        bn_load = self.server.gui.add_button("Load Trajectory File")
        bn_plan = self.server.gui.add_button("Plan Motion")
        bn_animate = self.server.gui.add_button("Play Animation")
        hl_file = self.server.gui.add_text(
            "Trajectory File",
            "planned_trajectory.json",
        )

    def run(self) -> None:
        """Keep the visualizer running - blocking call"""
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_animations()

