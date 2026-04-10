"""Robot wrapper: registry + IK controller + link pose queries."""

import numpy as np
from yourdfpy import URDF
from hbmp import Robot_T2DA2, FrameEnum, ColGroup
from ampl import Tf as AmplTf

# Map EEF frame names to FrameEnum
EEF_FRAME_MAP = {
    "FRAME_TACTILE_L": FrameEnum.FRAME_TACTILE_L,
    "FRAME_TACTILE_R": FrameEnum.FRAME_TACTILE_R,
}

# Arm joint slices in the 16-DOF q vector (torso=0:2, left=2:9, right=9:16)
ARM_SLICE = {
    FrameEnum.FRAME_TACTILE_L: slice(2, 9),
    FrameEnum.FRAME_TACTILE_R: slice(9, 16),
}


def to_viz(q16: np.ndarray, gripper_left: float = 0.05, gripper_right: float = 0.05) -> np.ndarray:
    """Convert 16-DOF q to 18-DOF visualization q (with gripper slots)."""
    q_viz = np.zeros(18, dtype=np.float64)
    np.copyto(q_viz[:9], q16[:9])
    np.copyto(q_viz[9 + 1: 16 + 1], q16[9:16])
    q_viz[2 + 7] = gripper_left
    q_viz[-1] = gripper_right
    return q_viz


class RobotController:
    """Wraps Robot_T2DA2 with IK and link pose queries."""

    def __init__(self, robot_cfg):
        self.cfg = robot_cfg
        self.agent = Robot_T2DA2(
            "hb11_left", "hb11_right", "hb11_torso",
            robot_cfg.params.wbc_config,
            robot_cfg.params.ndof,
        )

        if robot_cfg.initial_q:
            np.copyto(self.agent.q, np.array(robot_cfg.initial_q, dtype=np.float64))

        self.agent.update_kin(self.agent.q)
        self.agent.set_wall(x_wall=[0, 1.25], z_wall=[0.9, 2.0])

        self._urdf_fk = URDF.load(robot_cfg.urdf_visual)

    @property
    def q(self) -> np.ndarray:
        return self.agent.q

    def q_viz(self) -> np.ndarray:
        return to_viz(self.agent.q)

    def get_link_pose(self, link_name: str) -> np.ndarray:
        """Get 4x4 world pose of a robot link via URDF FK."""
        self._urdf_fk.update_cfg(self.q_viz())
        return self._urdf_fk.get_transform(link_name)

    def snap_to_eef(self, frame_name: str, pose_7d: np.ndarray) -> bool:
        """Directly solve IK and set joint state to match target EEF pose.

        pose_7d: [qx, qy, qz, qw, tx, ty, tz] — as returned by scene.get_eef_pose().
        Picks the valid IK solution closest to current joint state.
        Returns True if a valid solution was found and applied.
        """
        frame = EEF_FRAME_MAP.get(frame_name)
        if frame is None:
            return False

        ik_result = self.agent.get_ik(AmplTf(pose_7d), frame)
        if not ik_result:
            return False

        arm_sl = ARM_SLICE[frame]
        q_arm_current = self.agent.q[arm_sl]
        best_q = None
        best_dist = float("inf")

        for _, (iks_batch, ikq_batch) in ik_result.items():
            valid_idx = np.where(iks_batch)[0]
            if len(valid_idx) == 0:
                continue
            valid_qs = ikq_batch[valid_idx]
            dists = np.linalg.norm(valid_qs - q_arm_current, axis=1)
            i_best = np.argmin(dists)
            if dists[i_best] < best_dist:
                best_dist = dists[i_best]
                best_q = valid_qs[i_best]

        if best_q is None:
            return False

        self.agent.q[arm_sl] = best_q
        return True

    def update_kin(self):
        self.agent.update_kin(self.agent.q)

    def update_col(self):
        self.agent.update_col_self()

    def check_collision(self) -> ColGroup:
        return self.agent.check_self_collision()

    def get_fk(self, frame: FrameEnum):
        return self.agent.get_fk(frame)

    def wall(self):
        return self.agent.wall()

    def set_wall(self, **kwargs):
        self.agent.set_wall(**kwargs)

    def get_arm_bound(self, frame: FrameEnum, which: str):
        return self.agent.get_arm_bound(frame, which)

    def get_limits(self):
        return self.agent.get_limits()
