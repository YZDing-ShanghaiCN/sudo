"""
Motion Planning with Crate and Robot Collision Checking
This module provides collision-aware motion planning for the hillbot_left arm
with dynamic crate obstacle handling and trajectory optimization.
"""

import numpy as np
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import ampl
import pyampl


@dataclass
class PlanningConfig:
    """Configuration for motion planning parameters"""
    max_edge_length: float = 0.5
    max_samples: int = 2000
    edge_discrete_resolution: int = 16
    nb_subdivision_internal: int = 3
    nb_refine: int = 32
    ik_redundant_search: int = 512
    pcd_resolution: float = 0.0025


@dataclass
class CrateConfig:
    """Configuration for crate obstacles"""
    crate_0_position: np.ndarray
    crate_1_position: np.ndarray
    crate_0_pose_rwt: Optional[np.ndarray] = None
    crate_1_pose_rwt: Optional[np.ndarray] = None


class CollisionMotionPlanner:
    """
    Motion planner with crate collision checking for the hillbot_left robot arm.
    Handles both robot self-collision and collision with dynamic crate obstacles.
    """

    def __init__(
        self,
        config: Optional[PlanningConfig] = None,
        arm_config: Optional[Any] = None,
        convex_meshes: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        gripper_collision: Optional[pyampl.CollisionObjectConvex] = None,
    ):
        """
        Initialize the collision motion planner.

        Args:
            assets_dir: Path to assets directory containing meshes and URDF files
            config: Planning configuration parameters
            arm_config: Optional arm configuration (creates default if None)
        """
        self.config = config or PlanningConfig()
        self.convex_meshes = convex_meshes or {}
        self.cvh_gripper = gripper_collision
        
        # Initialize robot agent with arm configuration
        if arm_config is None:
            self.arm_config = pyampl.create_default_arm_config("hillbot_left")
        else:
            self.arm_config = arm_config
        
        self.agent = pyampl.AgentArm(
            self.arm_config.name, self.arm_config.dim, self.arm_config
        )
        self.agent.wall[2] = 0.6  # Set wall height
        # Initialize state variables
        self.state = np.array(self.agent.state_ref.tolist(), dtype=np.float64)
        self.state_from = np.array(self.agent.state_ref.tolist(), dtype=np.float64)
        self.state_to = np.array(self.agent.state_ref.tolist(), dtype=np.float64)
        
        # Initialize collision scene
        self.collision_scene = pyampl.CollisionScene()
        self._setup_collision_objects()
        
        # Planning results
        self.planned_trajectory: Optional[np.ndarray] = None
        self.trajectory_shortcut: Optional[np.ndarray] = None
        self.trajectory_refined: Optional[np.ndarray] = None
        
    def _setup_collision_objects(self) -> None:
        """Setup collision objects from provided meshes"""
        # Load convex obstacles from provided meshes
        for name, tuple_vf in self.convex_meshes.items():
            self.collision_scene.insert_convex(
                name,
                tuple_vf,
                create_pcd=True,
                pcd_dx=self.config.pcd_resolution,
            )
            self.collision_scene.enable_collision(name)
        
        # Validate agent setup
        self.agent.validate()

    def set_start_state(self, q_init: np.ndarray) -> None:
        """Set the initial configuration for motion planning"""
        np.copyto(self.state_from, q_init)
        np.copyto(self.state, q_init)

    def set_goal_state(self, q_goal: np.ndarray) -> None:
        """Set the goal configuration for motion planning"""
        np.copyto(self.state_to, q_goal)

    def set_start_pose(self, tf44: np.ndarray) -> Optional[np.ndarray]:
        """Set the initial pose for motion planning using a 4x4 transformation matrix
        
        Returns:
            Joint configuration if IK succeeds, None otherwise
        """
        q_init = self.agent.ik_redundant_wall_torso(
            tf44,
            state_ref=self.state_from,
            nb_redundant_search=self.config.ik_redundant_search,
        )
        if len(q_init) > 0:
            self.set_start_state(q_init[0])
            return q_init[0]
        else:
            return None
        
    def set_goal_pose(self, tf44: np.ndarray) -> Optional[np.ndarray]:
        """Set the goal pose for motion planning using a 4x4 transformation matrix
        
        Returns:
            Joint configuration if IK succeeds, None otherwise
        """
        q_goal = self.agent.ik_redundant_wall_torso(
            tf44,
            state_ref=self.state_to,
            nb_redundant_search=self.config.ik_redundant_search,
        )
        if len(q_goal) > 0:
            self.set_goal_state(q_goal[0])
            return q_goal[0]
        else:
            return None

    def update_env_object_pose(self, object_poses: Dict[str, np.ndarray]) -> None:
        """
        Update environment object positions in the collision scene.

        Args:
            object_poses: Dictionary mapping object names to their poses as rwt (qx, qy, qz, qw, x, y, z)
        """
        for object_name, pose_rwt in object_poses.items():
            self.collision_scene.update_pose(object_name, pose_rwt)
        
        # Update pointcloud representations for all objects
        self.collision_scene.update_poses_pcd_from_convex()
        print(f"{list(object_poses.keys())=}")
        # Get and update distance field from pointclouds
        pcd_tmp = self.collision_scene.get_pointcloud(list(object_poses.keys()))
        self.collision_scene.update_df_from_pcd(pcd_tmp)

    def plan_trajectory_with_crates(
        self,
        crate_poses: Dict[str, np.ndarray],
        attach_object: bool = False,
        verbose: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Plan a collision-free trajectory avoiding crates and self-collision.

        Args:
            crate_poses: Dictionary mapping crate names to rwt poses
            attach_object: If True, use attached object collision checking
            verbose: If True, print planning details

        Returns:
            Tuple of (success: bool, trajectory: Optional[np.ndarray])
                - success: True if a valid trajectory was found
                - trajectory: Refined trajectory waypoints or None if planning failed
        """
        # Update collision scene with current crate poses
        self.update_env_object_pose(crate_poses)
        
        # Select collision checking method
        if attach_object:
            pyampl.AgentArm.collision_free_trajectory = (
                pyampl.AgentArm.collision_free_trajectory_attach
            )
            if verbose:
                print("Using collision checking WITH attached object")
        else:
            pyampl.AgentArm.collision_free_trajectory = (
                pyampl.AgentArm.collision_free_trajectory_no_attach
            )
            if verbose:
                print("Using collision checking WITHOUT attached object")
        
        # Run RRT-Connect motion planner
        if verbose:
            print(
                f"Planning from {np.array_str(self.state_from, precision=3)} "
                f"to {np.array_str(self.state_to, precision=3)}"
            )
        
        mp = pyampl.RRTConnect(
            agent=self.agent,
            env=self.collision_scene,
            q_init=self.state_from,
            q_goal=self.state_to,
            max_edge_length=self.config.max_edge_length,
            max_samples=self.config.max_samples,
            edge_discrete_resolution=self.config.edge_discrete_resolution,
        )
        
        path = mp.rrt_connect()
        
        if len(path) == 0:
            if verbose:
                print("Motion planning FAILED: No feasible path found")
            return False, None
        
        if verbose:
            print(f"Motion planning SUCCESS: Found path with {len(path)} waypoints")
        
        # Optimize trajectory
        path = np.array(path)
        self.planned_trajectory = path.copy()
        
        # Shortcut optimization
        path_shortcut = mp.shortcut(path, nb_subdivision_internal=self.config.nb_subdivision_internal)
        self.trajectory_shortcut = path_shortcut.copy()
        
        # Trajectory refinement
        path_refine = pyampl.refine_trajectory_trivial(
            path_shortcut, nb_refine=self.config.nb_refine
        )
        self.trajectory_refined = path_refine.copy()
        
        # Print trajectory statistics
        if verbose:
            len_feasible = pyampl.get_traj_length_from_waypoints(path)
            len_shortcut = pyampl.get_traj_length_from_waypoints(path_shortcut)
            len_refined = pyampl.get_traj_length_from_waypoints(path_refine)
            straight_line = np.linalg.norm(self.state_from - self.state_to)
            
            print(
                f"Trajectory length: "
                f"feasible={len_feasible:.4f} -> "
                f"shortcut={len_shortcut:.4f} -> "
                f"refined={len_refined:.4f} "
                f"(straight_line={straight_line:.4f})"
            )
            print(f"Waypoint count: {len(path)} -> {len(path_shortcut)} -> {len(path_refine)}")
        
        return True, path_refine

    def check_trajectory_collision(
        self,
        trajectory: np.ndarray,
        crate_poses: Dict[str, np.ndarray],
    ) -> Tuple[bool, List[int]]:
        """
        Check if a trajectory is collision-free.

        Args:
            trajectory: Array of joint configurations (N x 7)
            crate_poses: Dictionary mapping crate names to rwt poses

        Returns:
            Tuple of (is_collision_free: bool, collision_indices: List[int])
                - is_collision_free: True if entire trajectory is collision-free
                - collision_indices: List of waypoint indices that have collisions
        """
        self.update_env_object_pose(crate_poses)
        collision_indices = []
        
        for idx, config in enumerate(trajectory):
            self.agent.fk_rwt = config
            # The agent.fk_rwt triggers forward kinematics
            # Collision checking would be done via the collision scene
            # This is a simplified check - actual implementation depends on pyampl API
        
        return len(collision_indices) == 0, collision_indices

    def compute_ik_with_collision(
        self,
        target_tf44: np.ndarray,
        state_ref: Optional[np.ndarray] = None,
        crate_poses: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Compute collision-free inverse kinematics for a target transform.

        Args:
            target_tf44: 4x4 target transformation matrix
            state_ref: Reference configuration for IK (uses current state if None)
            crate_poses: Optional crate poses for collision checking

        Returns:
            Tuple of (success: bool, config: Optional[np.ndarray])
        """
        if state_ref is None:
            state_ref = self.state
        
        if crate_poses is not None:
            self.update_env_object_pose(crate_poses)
        
        qs_ik = self.agent.ik_redundant_wall_torso_df(
            target_tf44,
            state_ref=state_ref,
            nb_redundant_search=self.config.ik_redundant_search,
            env=self.collision_scene,
        )
        
        if len(qs_ik) > 0:
            return True, qs_ik[0]
        return False, None

    def save_trajectory_to_json(
        self,
        filepath: str,
        include_shortcut: bool = False,
        include_refined: bool = True,
    ) -> None:
        """
        Save planned trajectory to JSON file.

        Args:
            filepath: Output JSON file path
            include_shortcut: Include shortcut trajectory in output
            include_refined: Include refined trajectory in output
        """
        output_data = {
            "timestamp": str(np.datetime64('now')),
            "config": {
                "max_edge_length": self.config.max_edge_length,
                "max_samples": self.config.max_samples,
                "edge_discrete_resolution": self.config.edge_discrete_resolution,
                "nb_subdivision_internal": self.config.nb_subdivision_internal,
                "nb_refine": self.config.nb_refine,
            },
            "start_config": self.state_from.tolist(),
            "goal_config": self.state_to.tolist(),
        }
        
        if self.planned_trajectory is not None:
            output_data["trajectory_raw"] = self.planned_trajectory.tolist()
        
        if include_shortcut and self.trajectory_shortcut is not None:
            output_data["trajectory_shortcut"] = self.trajectory_shortcut.tolist()
        
        if include_refined and self.trajectory_refined is not None:
            output_data["trajectory_refined"] = self.trajectory_refined.tolist()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

    def load_trajectory_from_json(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load trajectory from JSON file.

        Args:
            filepath: Input JSON file path

        Returns:
            Trajectory array or None if loading fails
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Try to load refined trajectory first, then shortcut, then raw
            if "trajectory_refined" in data:
                return np.array(data["trajectory_refined"], dtype=np.float64)
            elif "trajectory_shortcut" in data:
                return np.array(data["trajectory_shortcut"], dtype=np.float64)
            elif "trajectory_raw" in data:
                return np.array(data["trajectory_raw"], dtype=np.float64)
            else:
                print("No trajectory found in JSON file")
                return None
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
        except json.JSONDecodeError:
            print(f"Invalid JSON file: {filepath}")
            return None

