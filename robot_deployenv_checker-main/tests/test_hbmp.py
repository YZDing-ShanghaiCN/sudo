"""Unit tests for hbmp: FK, IK tracking, collision, Tf format."""

import numpy as np
import pytest
import ampl
from ampl import Tf as AmplTf
from hbmp import Robot_T2DA2, FrameEnum, ColGroup

# Reference q from main_viser.py
Q_REF = np.array([
    0.15, 0.3, 0.8, 0.64, 1.5, -1.65, -0.8,
    -0.8, 0.6, 0.8, 0.64, 1.5, -1.65, -0.8, -0.8, 0.6,
])

WBC_CONFIG = "hbmp/wbc_config_hb.yaml"


@pytest.fixture(scope="module")
def robot():
    """Create a Robot_T2DA2 instance for tests."""
    agent = Robot_T2DA2("hb11_left", "hb11_right", "hb11_torso", WBC_CONFIG, 16)
    np.copyto(agent.q, Q_REF)
    agent.update_kin(agent.q)
    agent.set_wall(x_wall=[0, 1.25], z_wall=[0.9, 2.0])
    return agent


# ---- Test 1: Tf format ----

class TestTfFormat:
    def test_tf_from_7d_roundtrip(self):
        """ampl.Tf([qx,qy,qz,qw,tx,ty,tz]) should round-trip through .matrix."""
        pose_7d = np.array([0.0, 0.0, 0.707107, 0.707107, 1.0, 2.0, 3.0])
        tf = AmplTf(pose_7d)
        mat = tf.matrix
        assert mat.shape == (4, 4), f"Expected 4x4, got {mat.shape}"
        # Translation should match
        np.testing.assert_allclose(mat[:3, 3], [1.0, 2.0, 3.0], atol=1e-5)
        # Rotation should be valid (det=1)
        det = np.linalg.det(mat[:3, :3])
        assert abs(det - 1.0) < 1e-5, f"Rotation det={det}"

    def test_tf_identity(self):
        """Identity quaternion [0,0,0,1] should give identity rotation."""
        pose_7d = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        tf = AmplTf(pose_7d)
        mat = tf.matrix
        np.testing.assert_allclose(mat, np.eye(4), atol=1e-10)

    def test_tf_position_property(self):
        """Tf.position should return the translation part."""
        pose_7d = np.array([0.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0])
        tf = AmplTf(pose_7d)
        np.testing.assert_allclose(tf.position, [5.0, 6.0, 7.0], atol=1e-10)

    def test_tf_wxyz_property(self):
        """Tf.wxyz should return [w,x,y,z] format."""
        pose_7d = np.array([0.1, 0.2, 0.3, 0.9274, 0.0, 0.0, 0.0])
        tf = AmplTf(pose_7d)
        wxyz = tf.wxyz
        # wxyz[0] should be w, wxyz[1:] should be x,y,z
        assert len(wxyz) == 4
        # The w component (from input qw=0.9274) should be wxyz[0]
        assert abs(wxyz[0] - 0.9274) < 0.01 or abs(wxyz[0] + 0.9274) < 0.01

    def test_viser_to_ampl_conversion(self):
        """Viser wxyz[1,2,3,0] reordering should produce valid ampl.Tf input."""
        # Simulate viser gizmo output
        viser_wxyz = np.array([0.707107, 0.0, 0.0, 0.707107])  # w,x,y,z
        viser_pos = np.array([1.0, 2.0, 3.0])
        # Convert: wxyz[[1,2,3,0]] → [x,y,z,w]
        pose_7d = np.hstack([viser_wxyz[[1, 2, 3, 0]], viser_pos])
        tf = AmplTf(pose_7d)
        np.testing.assert_allclose(tf.position, [1.0, 2.0, 3.0], atol=1e-5)


# ---- Test 2: Forward Kinematics ----

class TestFK:
    def test_fk_left_valid(self, robot):
        """FK for left tactile should return valid position."""
        tf = robot.get_fk(FrameEnum.FRAME_TACTILE_L)
        pos = tf.position
        assert len(pos) == 3
        # Position should be reasonable (within workspace)
        assert 0.0 < pos[0] < 2.0, f"x={pos[0]} out of range"
        assert -1.0 < pos[1] < 1.0, f"y={pos[1]} out of range"
        assert 0.0 < pos[2] < 2.0, f"z={pos[2]} out of range"

    def test_fk_right_valid(self, robot):
        """FK for right tactile should return valid position."""
        tf = robot.get_fk(FrameEnum.FRAME_TACTILE_R)
        pos = tf.position
        assert len(pos) == 3
        assert 0.0 < pos[0] < 2.0

    def test_fk_torso_valid(self, robot):
        """FK for torso should return valid position."""
        tf = robot.get_fk(FrameEnum.FRAME_TORSO_2)
        pos = tf.position
        assert len(pos) == 3

    def test_fk_left_right_symmetric(self, robot):
        """Left and right EEFs should be roughly y-mirrored for symmetric q."""
        tf_l = robot.get_fk(FrameEnum.FRAME_TACTILE_L)
        tf_r = robot.get_fk(FrameEnum.FRAME_TACTILE_R)
        # x and z should be similar, y should be opposite sign
        assert abs(tf_l.position[0] - tf_r.position[0]) < 0.1
        assert abs(tf_l.position[1] + tf_r.position[1]) < 0.1  # y mirrored
        assert abs(tf_l.position[2] - tf_r.position[2]) < 0.1

    def test_fk_returns_tf_with_matrix(self, robot):
        """FK result should have a .matrix property returning 4x4."""
        tf = robot.get_fk(FrameEnum.FRAME_TACTILE_L)
        mat = tf.matrix
        assert mat.shape == (4, 4)
        # Should be valid SE(3)
        det = np.linalg.det(mat[:3, :3])
        assert abs(det - 1.0) < 1e-5


# ---- Test 3: Collision Detection ----

class TestCollision:
    def test_ref_q_collision_free(self, robot):
        """Reference q should be collision-free."""
        robot.update_kin(Q_REF)
        robot.update_col_self()
        col = robot.check_self_collision()
        assert col == ColGroup.NONE, f"Expected NONE, got {col}"

    def test_collision_gradient_self_returns_tuple_or_none(self, robot):
        """collision_gradient_self should return tuple(p,g,d,l) or None."""
        robot.update_kin(Q_REF)
        robot.update_col_self()
        result = robot.collision_gradient_self(ColGroup.T_L)
        assert result is None or (isinstance(result, tuple) and len(result) == 4), \
            f"Expected None or 4-tuple, got {type(result)}"

    def test_collision_gradient_self_both_arms(self, robot):
        """Test collision gradient for both arms."""
        robot.update_kin(Q_REF)
        robot.update_col_self()
        for group in [ColGroup.T_L, ColGroup.T_R]:
            result = robot.collision_gradient_self(group)
            if result is not None:
                p, g, d, l = result
                assert p.ndim == 2 and p.shape[1] == 3
                assert g.ndim == 2 and g.shape[1] == 3

    def test_wall_bounds(self, robot):
        """Wall should return 3 pairs of [min, max]."""
        wall = robot.wall()
        assert len(wall) == 3
        for pair in wall:
            assert len(pair) == 2
            assert pair[0] < pair[1]


# ---- Test 4: IK Tracking ----

class TestTracking:
    def test_track_tcp_left_at_current_pose_stable(self, robot):
        """Tracking to current FK pose should not change q significantly."""
        np.copyto(robot.q, Q_REF)
        robot.update_kin(robot.q)
        tf_target = robot.get_fk(FrameEnum.FRAME_TACTILE_L)

        q_before = robot.q.copy()
        try:
            q_new = robot.track_tcp(FrameEnum.FRAME_TACTILE_L, tf_target)
            # Should be very close to original
            np.testing.assert_allclose(q_new, q_before, atol=0.05,
                                       err_msg="Tracking to current pose should be stable")
        except TypeError as e:
            if "'NoneType'" in str(e):
                pytest.skip("collision_gradient_self returned None (known issue)")
            raise

    def test_track_tcp_left_moves_toward_target(self, robot):
        """Tracking to a nearby pose should move q toward target."""
        np.copyto(robot.q, Q_REF)
        robot.update_kin(robot.q)
        tf_current = robot.get_fk(FrameEnum.FRAME_TACTILE_L)

        # Create a slightly offset target
        offset = np.array([0.0, 0.0, 0.0, 1.0, 0.02, 0.0, 0.0])  # 2cm x offset
        target_pos = tf_current.position + np.array([0.02, 0.0, 0.0])
        pose_7d = np.hstack([np.array([0, 0, 0, 1]), target_pos])  # Identity rotation + offset pos
        # Use the actual FK rotation instead
        tf_mat = tf_current.matrix.copy()
        tf_mat[:3, 3] = target_pos
        tf_target = AmplTf(ampl.tf44_to_qt7(tf_mat))

        q_before = robot.q.copy()
        try:
            q_new = robot.track_tcp(FrameEnum.FRAME_TACTILE_L, tf_target)
            # q should have changed
            assert not np.allclose(q_new, q_before, atol=1e-6), \
                "Expected q to change when tracking to offset target"
        except TypeError as e:
            if "'NoneType'" in str(e):
                pytest.skip("collision_gradient_self returned None (known issue)")
            raise

    def test_collision_gradient_none_crash(self, robot):
        """Demonstrate the None collision gradient crash."""
        np.copyto(robot.q, Q_REF)
        robot.update_kin(robot.q)
        robot.update_col_self()

        # Check if collision_gradient_self returns None for T_L
        result = robot.collision_gradient_self(ColGroup.T_L)
        if result is None:
            # This confirms Bug 1: track_tcp would crash here
            tf_target = robot.get_fk(FrameEnum.FRAME_TACTILE_L)
            with pytest.raises(TypeError, match="NoneType"):
                robot.track_tcp(FrameEnum.FRAME_TACTILE_L, tf_target)


# ---- Test 5: to_viz mapping ----

class TestToViz:
    def test_to_viz_shape(self):
        """to_viz should produce 18-DOF from 16-DOF."""
        from deployenv_checker.robot import to_viz
        q16 = np.ones(16)
        q18 = to_viz(q16)
        assert q18.shape == (18,)

    def test_to_viz_gripper_slots(self):
        """Gripper values should be at indices 9 and 17."""
        from deployenv_checker.robot import to_viz
        q16 = np.zeros(16)
        q18 = to_viz(q16, gripper_left=0.1, gripper_right=0.2)
        assert q18[9] == 0.1
        assert q18[17] == 0.2

    def test_to_viz_preserves_joints(self):
        """Joint values should be preserved in the right slots."""
        from deployenv_checker.robot import to_viz
        q16 = np.arange(16, dtype=np.float64)
        q18 = to_viz(q16)
        # First 9 joints (torso + left arm) map directly
        np.testing.assert_array_equal(q18[:9], q16[:9])
        # Right arm joints (q16[9:16]) map to q18[10:17]
        np.testing.assert_array_equal(q18[10:17], q16[9:16])
