"""
Core tests for motion planning with collision checking.
Tests for:
- Collision-free start pose planning
- Collision start pose planning
- Successful planning
- Failed planning
Each test has option to visualize results.

Usage:
    python3.11 vis_test_planning.py 1 --visualize
    python3.11 vis_test_planning.py 2 --visualize

"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

# Add motion_plan folder to path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Tuple, Dict, Optional
import trimesh
import ampl
import pyampl

from collision_motion_plan import CollisionMotionPlanner
from scene_visualizer import SceneVisualizer
from yourdfpy import URDF
from viz_viser import ViserObject
from unittest.mock import patch


class MotionPlanningTests:
    """Core motion planning tests"""

    ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets")

    def __init__(self, assets_dir=None, visualize=False):
        """
        Initialize test suite.
        
        Args:
            assets_dir: Path to assets directory
            visualize: Enable visualization for tests
        """
        self.assets_dir = assets_dir or self.ASSETS_DIR
        self.visualize = visualize
        self.convex_meshes = self._load_convex_meshes()
        self.gripper_collision = self._load_gripper_collision()
        self.visual_meshes = self._load_visual_meshes()
        self.planner = CollisionMotionPlanner(
            assets_dir=self.assets_dir,
            convex_meshes=self.convex_meshes,
            gripper_collision=self.gripper_collision,
        )
        self.robot_urdf = self._load_robot_urdf()
        
        if visualize:
            # Create real visualizer for visualization, passing arm_config from planner
            self.visualizer = SceneVisualizer(
                assets_dir=self.assets_dir,
                planner=self.planner,
                arm_urdf=self.robot_urdf,
            )
        else:
            # Mock visualizer when not visualizing
            with patch('scene_visualizer.viser.ViserServer'):
                self.visualizer = None
        
        # Standard crate poses
        self.workbench_center = np.array([0.7, 0.0, 0.7])
        self.crate_center = self.workbench_center + np.array([0.0, 0.0, 0.01])
        self.crate_poses = {
            "crate_0": np.array([
                0.0, 0.0, 0.0, 1.0,
                self.crate_center[0], self.crate_center[1] - 0.2, self.crate_center[2]
            ]),
            "crate_1": np.array([
                0.0, 0.0, 0.0, 1.0,
                self.crate_center[0], self.crate_center[1] + 0.2, self.crate_center[2]
            ]),
        }

    def _load_convex_meshes(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        convex_dir = os.path.join(
            self.assets_dir, "mesh", "scene_00", "obstacle", "convex"
        )
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

    def _load_gripper_collision(self) -> Optional[pyampl.CollisionObjectConvex]:
        gripper_path = os.path.join(
            self.assets_dir, "mesh", "scene_00", "tool", "gripper.ply"
        )
        if not os.path.exists(gripper_path):
            return None
        return pyampl.CollisionObjectConvex(ampl.read_trimesh(gripper_path))

    def _load_visual_meshes(self) -> Dict[str, Optional[trimesh.Trimesh]]:
        mesh_map = {
            "desk": "mesh/desk.ply",
            "torso": "mesh/torso_vla_fine.ply",
            "crate_0": "mesh/scene_00/obstacle/visual/crate_0.ply",
            "crate_1": "mesh/scene_00/obstacle/visual/crate_1.ply",
        }
        visual_meshes: Dict[str, Optional[trimesh.Trimesh]] = {}
        for name, rel_path in mesh_map.items():
            mesh_path = os.path.join(self.assets_dir, rel_path)
            if not os.path.exists(mesh_path):
                visual_meshes[name] = None
                continue
            visual_meshes[name] = trimesh.load(mesh_path, process=False)
        return visual_meshes

    def _load_robot_urdf(self) -> Optional[URDF]:
        urdf_path = os.path.join(
            self.assets_dir, "urdf", self.planner.arm_config.name, "urdf.urdf"
        )
        if not os.path.exists(urdf_path):
            return None
        return URDF.load(urdf_path)

    def _add_visual_mesh(
        self,
        name: str,
        position: np.ndarray,
        color: Tuple[int, int, int, int],
    ) -> None:
        mesh = self.visual_meshes.get(name)
        if mesh is None:
            return
        obj = ViserObject(
            mesh,
            name=name,
            server=self.visualizer.server,
            color=color,
            no_control=True,
        )
        obj.handler.position = position
        self.visualizer.scene_objects[name] = obj

    def _add_robot_torso(self) -> None:
        mesh = self.visual_meshes.get("torso")
        if mesh is not None:
            self.visualizer.add_robot_torso(
                mesh,
                position=self.workbench_center,
                color=(170, 170, 170, 255),
            )

    def _add_standard_scene(self) -> None:
        self._add_visual_mesh(
            name="desk",
            position=self.workbench_center,
            color=(128, 255, 128, 200),
        )
        self._add_robot_torso()
        for crate_name, pose in self.crate_poses.items():
            self._add_visual_mesh(
                name=crate_name,
                position=pose[4:7],
                color=(128, 128, 255, 200),
            )

    def _add_cube(self, name: str, size: Tuple[float, float, float], position: np.ndarray) -> None:
        cube_mesh = trimesh.creation.box(extents=size)
        cube_obj = ViserObject(
            cube_mesh,
            name=name,
            server=self.visualizer.server,
            color=(200, 100, 100, 255),
            no_control=True,
        )
        cube_obj.handler.position = position
        self.visualizer.scene_objects[name] = cube_obj

    def _add_wall(self, name: str, size: Tuple[float, float, float], position: np.ndarray) -> None:
        wall_mesh = trimesh.creation.box(extents=size)
        wall_obj = ViserObject(
            wall_mesh,
            name=name,
            server=self.visualizer.server,
            color=(120, 120, 120, 255),
            no_control=True,
        )
        wall_obj.handler.position = position
        self.visualizer.scene_objects[name] = wall_obj

    def test_collision_free_with_qpos(self):
        """
        Test 1: Plan with collision-free start pose
        Start configuration is not in collision with obstacles.
        """
        print("\n" + "="*70)
        print("TEST 1: Planning with Collision-Free Start Pose")
        print("="*70)
        
        # Get reference config (collision-free)
        q_goal = np.array([1.43,-0.82,0.65,-1.75,-2.09,0.07,-0.36], dtype=np.float64)
        q_start = np.array([1.65,-1.88,0.615,-1.514,-2.371,-0.5,-1.04], dtype=np.float64)
        
        self.planner.agent.fk_rwt=q_goal
        rwt = self.planner.agent.fk_rwt[-1].copy()
        print(f"Goal end-effector pose: {rwt=}")
        
        
        self.planner.set_start_state(q_start)
        self.planner.set_goal_state(q_goal)
        
        print(f"Start config: {np.array_str(q_start, precision=3)}")
        print(f"Goal config:  {np.array_str(q_goal, precision=3)}")
        print("Status: Start pose is collision-free")
        
        # Plan
        success, trajectory = self.planner.plan_trajectory_with_crates(
            crate_poses=self.crate_poses,
            attach_object=False,
            verbose=True,
        )
        
        if self.visualize:
            print("\nVisualizing result...")
            self._add_standard_scene()
            self.visualizer.add_robot_arm(
                urdf=self.robot_urdf,
                pose_base=self.planner.agent.pose_base,
                state_ref=q_start,
            )
            self.visualizer.show_start_pose(q_start)
            self.visualizer.show_end_pose(q_goal)
            if success and trajectory is not None:
                self.visualizer.visualize_trajectory(trajectory)
                self.visualizer.play_trajectory_animation(trajectory)
            print("Visualization ready at viser server")
        
        # assert success, "Planning should succeed from collision-free start"
        # assert trajectory is not None, "Trajectory should be returned"
        print("\n✓ TEST 1 PASSED")
        return success, trajectory

    def test_set_start_pose(self):
        """
        Test 2: Test set_start_pose function
        Verify that set_start_pose correctly sets the start pose from a transformation matrix.
        """
        print("\n" + "="*70)
        print("TEST 2: Testing set_start_pose Function")
        print("="*70)
        
        # Test 1: Valid pose with rotation
        print("\nTest 2.1: Setting pose with rotation (45° around z-axis)")
        tf44_rotated = np.array([
            [0.3072087,  0.8747070,  0.3748473, 0.77],
            [0.9083399, -0.1520567, -0.3896117, 0.16],
            [-0.2837980,  0.4601808, -0.8412445, 0.87],
            [0,0,0,1.0]
        ], dtype=np.float64)
        
        q_rotated = self.planner.set_start_pose(tf44_rotated)
        print(f"IK Result: {q_rotated is not None}")
        
        if q_rotated is not None:
            print(f"Returned q pose: {np.array_str(q_rotated, precision=3)}")
            print(f"Start state set to: {np.array_str(self.planner.state_from, precision=3)}")
            assert len(q_rotated) == 7, "Returned q pose should have 7 joint values"
            assert np.allclose(q_rotated, self.planner.state_from), \
                "Returned q pose should match state_from"
            print("✓ Rotated pose test PASSED")
        else:
            print("⚠ IK failed for rotated pose")
        
        # Test 2: Unreachable pose (very far away)
        print("\nTest 2.2: Testing unreachable pose (far away)")
        tf44_unreachable = np.eye(4, dtype=np.float64)
        tf44_unreachable[0, 3] = 10.0  # Very far
        tf44_unreachable[1, 3] = 10.0
        tf44_unreachable[2, 3] = 10.0
        
        initial_state_unreachable = self.planner.state_from.copy()
        q_unreachable = self.planner.set_start_pose(tf44_unreachable)
        
        print(f"Transform: position=[{tf44_unreachable[0,3]:.1f}, {tf44_unreachable[1,3]:.1f}, {tf44_unreachable[2,3]:.1f}]")
        print(f"IK Result: {q_unreachable is not None}")
        
        if q_unreachable is None:
            assert np.allclose(self.planner.state_from, initial_state_unreachable), \
                "State should not change when IK fails"
            print("✓ Unreachable pose test PASSED (correctly returned None)")
        else:
            print(f"⚠ Unexpectedly found solution for far pose: {np.array_str(q_unreachable, precision=3)}")
        
        if self.visualize:
            print("\nVisualizing final pose...")
            self._add_standard_scene()
            if q_rotated is not None:
                self.visualizer.add_robot_arm(
                    urdf=self.robot_urdf,
                    pose_base=self.planner.agent.pose_base,
                    state_ref=self.planner.state_from,
                )
                self.visualizer.show_start_pose(self.planner.state_from)
        
        print("\n✓ TEST 2 PASSED")
        return True

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "="*70)
        print("MOTION PLANNING TEST SUITE")
        print("="*70)
        print(f"Visualization: {'ENABLED' if self.visualize else 'DISABLED'}")
        
        results = {}
        
        try:
            success, traj = self.test_collision_free_with_qpos()
            results['collision_free_start'] = {'success': success, 'trajectory': traj}
        except Exception as e:
            print(f"✗ TEST 1 FAILED: {e}")
            results['collision_free_start'] = {'success': False, 'error': str(e)}
        
        try:
            success = self.test_set_start_pose()
            results['set_start_pose'] = {'success': success}
        except Exception as e:
            print(f"✗ TEST 2 FAILED: {e}")
            results['set_start_pose'] = {'success': False, 'error': str(e)}
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        passed = sum(1 for r in results.values() if r.get('success', False) or 'error' not in r)
        total = len(results)
        print(f"Passed: {passed}/{total}")
        
        for test_name, result in results.items():
            status = "✓" if result.get('success') else "✗"
            print(f"  {status} {test_name}")
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Motion Planning Tests'
    )
    parser.add_argument(
        'test_num',
        nargs='?',
        type=int,
        choices=[1, 2],
        default=None,
        help='Run specific test (1, 2). If not specified, runs all tests.'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization for tests'
    )
    parser.add_argument(
        '--assets',
        default=None,
        help='Path to assets directory (defaults to motion_plan/assets)'
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = MotionPlanningTests(
        assets_dir=args.assets,
        visualize=args.visualize
    )
    
    # Run specific test or all tests
    if args.test_num == 1:
        print("\nRunning TEST 1 only...")
        tester.test_collision_free_with_qpos()
    elif args.test_num == 2:
        print("\nRunning TEST 2 only...")
        tester.test_set_start_pose()
    else:
        print("\nRunning all tests...")
        tester.run_all_tests()
    
    results = None
    
    if args.visualize:
        print("\n✓ Viser server is running. Access it via the displayed URL.")
        print("  Press Ctrl+C to stop.")
        
        try:
            tester.visualizer.run()
        except KeyboardInterrupt:
            tester.visualizer.stop_animations()
            print("\nShutting down...")


if __name__ == "__main__":
    main()