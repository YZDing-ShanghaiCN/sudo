"""Test that EEF gizmo on_update callbacks fire and update _eef_poses."""

import dataclasses
import numpy as np
import pytest

from deployenv_checker.config import EEFConfig, RobotConfig, SceneConfig
from deployenv_checker.scene import SceneManager


# ---------------------------------------------------------------------------
# Minimal viser fakes — no network, no event loop required
# ---------------------------------------------------------------------------

class FakeHandle:
    """Fake TransformControlsHandle that records on_update callbacks."""

    def __init__(self, wxyz, position):
        self.wxyz = np.array(wxyz, dtype=np.float64)
        self.position = np.array(position, dtype=np.float64)
        self._callbacks = []
        self.disable_rotations = False
        self.disable_sliders = False
        self.disable_axes = False

    def on_update(self, func):
        self._callbacks.append(func)
        return func

    def fire(self, wxyz=None, position=None):
        """Simulate a client drag: update pose then call all callbacks."""
        if wxyz is not None:
            self.wxyz = np.array(wxyz, dtype=np.float64)
        if position is not None:
            self.position = np.array(position, dtype=np.float64)

        @dataclasses.dataclass
        class FakeEvent:
            target: object

        event = FakeEvent(target=self)
        for cb in self._callbacks:
            cb(event)


class FakeScene:
    def __init__(self):
        self._handles: dict[str, FakeHandle] = {}

    def add_transform_controls(self, name, position, wxyz, scale, **kwargs):
        handle = FakeHandle(wxyz=wxyz, position=position)
        self._handles[name] = handle
        return handle

    def add_mesh_trimesh(self, *args, **kwargs):
        pass

    def add_frame(self, *args, **kwargs):
        return FakeHandle(wxyz=[1, 0, 0, 0], position=[0, 0, 0])

    def add_grid(self, *args, **kwargs):
        pass

    def add_line_segments(self, *args, **kwargs):
        return object()


class FakeServer:
    def __init__(self):
        self.scene = FakeScene()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scene(eef_names=("left_tool0", "right_tool0")):
    """Build a SceneManager backed by fake viser objects."""
    eefs = [
        EEFConfig(
            name=name,
            frame=f"FRAME_{name.upper()}",
            mesh_path="/nonexistent/gripper.glb",
            position=[0.7, 0.3 * (i + 1), 0.9],
            wxyz=[1.0, 0.0, 0.0, 0.0],
            scale=0.15,
        )
        for i, name in enumerate(eef_names)
    ]
    robot_cfg = RobotConfig(
        urdf_visual="/nonexistent/urdf.urdf",
        urdf_collision="/nonexistent/urdf.urdf",
        end_effectors=eefs,
    )
    cfg = SceneConfig(robot=robot_cfg)
    server = FakeServer()

    # Patch out URDF loading — we only test the gizmo/callback logic
    import unittest.mock as mock
    with mock.patch("deployenv_checker.scene.ViserUrdf"), \
         mock.patch("deployenv_checker.scene.URDF"):
        scene = SceneManager(server, cfg)
        scene._setup_robot()

    return scene, server


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEEFDragCallback:

    def test_callback_registered_for_each_eef(self):
        """on_update callback must be registered on every EEF gizmo handle."""
        scene, server = _make_scene()
        for name in ("left_tool0", "right_tool0"):
            handle = server.scene._handles[f"/{name}"]
            assert len(handle._callbacks) == 1, (
                f"Expected 1 on_update callback on /{name}, "
                f"got {len(handle._callbacks)}"
            )

    def test_drag_updates_eef_pose(self):
        """Firing the on_update callback must update _eef_poses."""
        scene, server = _make_scene(eef_names=("left_tool0",))

        new_wxyz = np.array([0.0, 1.0, 0.0, 0.0])   # qw,qx,qy,qz
        new_pos  = np.array([1.1, 2.2, 3.3])

        handle = server.scene._handles["/left_tool0"]
        handle.fire(wxyz=new_wxyz, position=new_pos)

        pose = scene.get_eef_pose("left_tool0")
        assert pose is not None

        # Pose is stored as [qx, qy, qz, qw, tx, ty, tz]
        # handle.wxyz is [qw, qx, qy, qz] → reordered [qx,qy,qz,qw] = [1,0,0,0]
        expected = np.hstack([new_wxyz[[1, 2, 3, 0]], new_pos])
        np.testing.assert_allclose(pose, expected)

    def test_initial_pose_seeded_from_config(self):
        """_eef_poses must be pre-seeded with the config pose before any drag."""
        scene, _ = _make_scene(eef_names=("left_tool0",))

        pose = scene.get_eef_pose("left_tool0")
        assert pose is not None
        assert pose.shape == (7,)

    def test_independent_callbacks_per_eef(self):
        """Dragging one gizmo must not update the other EEF's pose."""
        scene, server = _make_scene()

        right_pose_before = scene.get_eef_pose("right_tool0").copy()

        handle_left = server.scene._handles["/left_tool0"]
        handle_left.fire(wxyz=[0, 0, 1, 0], position=[9, 9, 9])

        right_pose_after = scene.get_eef_pose("right_tool0")
        np.testing.assert_array_equal(right_pose_before, right_pose_after)
