"""
Unit tests for motion planning with collision checking.
Run with: pytest test_planning.py -v
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

# Add motion_plan folder to path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import trimesh
import ampl
import pyampl

from collision_motion_plan import CollisionMotionPlanner, PlanningConfig


ASSETS_DIR = os.environ.get("MOTION_PLAN_ASSETS_DIR", "./assets")


def _load_convex_meshes(assets_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    convex_dir = os.path.join(assets_dir, "mesh", "scene_00", "obstacle", "convex")
    if not os.path.exists(convex_dir):
        return {}

    convex_meshes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for filename in sorted(os.listdir(convex_dir)):
        filepath = os.path.join(convex_dir, filename)
        if not os.path.isfile(filepath):
            continue
        name, _ = os.path.splitext(filename)
        mesh = trimesh.load(filepath, process=False)
        convex_meshes[name] = (mesh.vertices, mesh.faces)
    return convex_meshes


def _load_gripper_collision(assets_dir: str) -> Optional[pyampl.CollisionObjectConvex]:
    gripper_path = os.path.join(assets_dir, "mesh", "scene_00", "tool", "gripper.ply")
    if not os.path.exists(gripper_path):
        return None
    return pyampl.CollisionObjectConvex(ampl.read_trimesh(gripper_path))


@pytest.fixture(scope="module")
def planner():
    """Create a CollisionMotionPlanner instance shared across tests in this module."""
    convex_meshes = _load_convex_meshes(ASSETS_DIR)
    gripper_collision = _load_gripper_collision(ASSETS_DIR)
    return CollisionMotionPlanner(
        assets_dir=ASSETS_DIR,
        convex_meshes=convex_meshes,
        gripper_collision=gripper_collision,
    )


@pytest.fixture(scope="module")
def crate_poses():
    """Standard crate poses for testing."""
    workbench_center = np.array([0.7, 0.0, 0.7])
    crate_center = workbench_center + np.array([0.0, 0.0, 0.01])
    return {
        "crate_0": np.array([
            0.0, 0.0, 0.0, 1.0,
            crate_center[0], crate_center[1] - 0.2, crate_center[2],
        ]),
        "crate_1": np.array([
            0.0, 0.0, 0.0, 1.0,
            crate_center[0], crate_center[1] + 0.2, crate_center[2],
        ]),
    }


# ---------------------------------------------------------------------------
# Test: collision-free trajectory planning with joint configurations
# ---------------------------------------------------------------------------

class TestCollisionFreeWithQpos:
    """Plan with collision-free start/goal joint configurations."""

    def test_plan_trajectory_succeeds(self, planner, crate_poses):
        q_start = np.array(
            [1.65, -1.88, 0.615, -1.514, -2.371, -0.5, -1.04], dtype=np.float64
        )
        q_goal = np.array(
            [1.43, -0.82, 0.65, -1.75, -2.09, 0.07, -0.36], dtype=np.float64
        )

        planner.set_start_state(q_start)
        planner.set_goal_state(q_goal)

        success, trajectory = planner.plan_trajectory_with_crates(
            crate_poses=crate_poses,
            attach_object=False,
            verbose=False,
        )

        assert success, "Planning should succeed from collision-free start"
        assert trajectory is not None, "Trajectory should be returned"
        assert trajectory.ndim == 2, "Trajectory should be a 2D array"
        assert trajectory.shape[1] == 7, "Each waypoint should have 7 joint values"
        assert len(trajectory) >= 2, "Trajectory should have at least start and goal"

    def test_start_and_goal_endpoints(self, planner, crate_poses):
        q_start = np.array(
            [1.65, -1.88, 0.615, -1.514, -2.371, -0.5, -1.04], dtype=np.float64
        )
        q_goal = np.array(
            [1.43, -0.82, 0.65, -1.75, -2.09, 0.07, -0.36], dtype=np.float64
        )

        planner.set_start_state(q_start)
        planner.set_goal_state(q_goal)

        success, trajectory = planner.plan_trajectory_with_crates(
            crate_poses=crate_poses,
            attach_object=False,
            verbose=False,
        )

        assert success
        np.testing.assert_allclose(
            trajectory[0], q_start, atol=0.1,
            err_msg="First waypoint should be close to start config",
        )
        np.testing.assert_allclose(
            trajectory[-1], q_goal, atol=0.1,
            err_msg="Last waypoint should be close to goal config",
        )


# ---------------------------------------------------------------------------
# Test: set_start_pose (IK from 4x4 transform)
# ---------------------------------------------------------------------------

class TestSetStartPose:
    """Verify set_start_pose returns joint config or None."""

    def test_valid_pose_returns_joint_config(self, planner):
        tf44 = np.array([
            [0.3072087,  0.8747070,  0.3748473, 0.77],
            [0.9083399, -0.1520567, -0.3896117, 0.16],
            [-0.2837980,  0.4601808, -0.8412445, 0.87],
            [0, 0, 0, 1.0],
        ], dtype=np.float64)

        q = planner.set_start_pose(tf44)

        assert q is not None, "IK should find a solution for a valid reachable pose"
        assert len(q) == 7, "Returned q pose should have 7 joint values"
        np.testing.assert_array_equal(
            q, planner.state_from,
            err_msg="Returned q should match planner.state_from",
        )
        np.testing.assert_array_equal(
            planner.state, planner.state_from,
            err_msg="planner.state should equal planner.state_from",
        )

    def test_unreachable_pose_returns_none(self, planner):
        tf44_unreachable = np.eye(4, dtype=np.float64)
        tf44_unreachable[0, 3] = 10.0
        tf44_unreachable[1, 3] = 10.0
        tf44_unreachable[2, 3] = 10.0

        state_before = planner.state_from.copy()
        q = planner.set_start_pose(tf44_unreachable)

        assert q is None, "IK should return None for an unreachable pose"
        np.testing.assert_array_equal(
            planner.state_from, state_before,
            err_msg="state_from should not change when IK fails",
        )


# ---------------------------------------------------------------------------
# Test: update_env_object_pose
# ---------------------------------------------------------------------------

class TestUpdateEnvObjectPose:
    """Verify collision scene is updated with new object poses."""

    def test_update_does_not_raise(self, planner, crate_poses):
        # Should execute without error
        planner.update_env_object_pose(crate_poses)

    def test_ik_respects_updated_objects(self, planner, crate_poses):
        planner.update_env_object_pose(crate_poses)

        tf44 = np.array([
            [0.3072087,  0.8747070,  0.3748473, 0.77],
            [0.9083399, -0.1520567, -0.3896117, 0.16],
            [-0.2837980,  0.4601808, -0.8412445, 0.87],
            [0, 0, 0, 1.0],
        ], dtype=np.float64)

        q = planner.set_start_pose(tf44)
        # IK may or may not succeed depending on collision; just verify type
        assert q is None or len(q) == 7
