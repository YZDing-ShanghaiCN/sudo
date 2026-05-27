import numpy as np
from ampl import Tf, spatial_to_mixed_linear
from .col import *
from .common import *
from .base import Kin, FrameEnum
from pywbc import WBCManager
from .util import (
    spatial_to_mixed_linear_fast,
    spatial_to_mixed_linear_projection,
    MAGIC_NUMBER_COLLISION_OBB_OFFSET,
)


class WBC9(WBCManager):
    DOF: int = 9

    def __init__(self, config_path: str, backend: Kin):
        super().__init__(config_path)
        self._dir_elbo_pull = np.array([0.0, 0.0, -1.0], dtype=DTypeDouble)
        # self.set_elbo_pull()
        self._v_goal: np.ndarray = np.zeros(3, dtype=DTypeDouble)
        self._w_goal: np.ndarray = np.zeros(3, dtype=DTypeDouble)
        self._tf_diff: Tf = Tf()
        self._backend = backend
        self._q_curr = np.zeros(WBC9.DOF, dtype=DTypeDouble)
        self._dq = np.zeros(WBC9.DOF, dtype=DTypeDouble)
        self._dq_prev = np.zeros(WBC9.DOF, dtype=DTypeDouble)
        self._v_goal_relax = np.full(3, 0.1, dtype=DTypeDouble)

        constraint = next(
            (obj for obj in self.config.constraints if obj.name == "self_collision"),
            None,
        )
        print(constraint.dim)
        self._J_col = np.zeros((constraint.dim, WBC9.DOF), order="F", dtype=DTypeDouble)
        self._d_col = np.zeros(constraint.dim, dtype=DTypeDouble)
        # print(constraint.dim)
        self._J_debug = None
        self._p_mocap = None

    def set_elbo_pull(self, dir_pull: np.ndarray):
        np.copyto(self._dir_elbo_pull, dir_pull / np.linalg.norm(dir_pull))

    def set_goal(self, tf_curr: Tf, tf_target: Tf):
        self._tf_diff = tf_target @ tf_curr.inv()
        self._v_goal = tf_target.position - tf_curr.position
        self._w_goal = self._tf_diff.log_so3()
        # _twist = ampl.transform_se3_downlog(
        #     (tf_target.matrix @ np.linalg.inv(tf_curr.matrix))
        # )
        # self._v_goal = (tf_target.matrix[:3, 3] - tf_curr.matrix[:3, 3]).flatten()
        # self._w_goal = _twist[3:]

    def update_collision_gradient(
        self,
        frame: FrameEnum,
        ps: np.ndarray,
        gs: np.ndarray,
        ds: np.ndarray,
        ls: np.ndarray,
    ):
        J_s = self._backend.get_jacobian(frame)
        ls[:] += 2
        spatial_to_mixed_linear(J_s, ps, -gs, ls, self._J_col)
        self._d_col = ds - MAGIC_NUMBER_COLLISION_OBB_OFFSET

    def update_state(self, q_curr: np.ndarray):
        np.copyto(self._q_curr, q_curr)

    def update_controller(self, frame: FrameEnum = FrameEnum.FRAME_TACTILE_L):
        J_s = self._backend.get_jacobian(frame)
        J_v_mixed = spatial_to_mixed_linear_fast(
            J_s, self._backend.get_fk(frame).position.astype(DTypeDouble)
        )
        J_v_elbo = np.zeros_like(J_v_mixed)
        J_v_elbo[:, : 2 + 2] = spatial_to_mixed_linear_fast(
            J_s, self._backend.get_fk(frame).position.astype(DTypeDouble)
        )[:, : 2 + 2]

        # J_v_elbo[:, :2] = 0
        # self._J_debug = J_v_elbo.copy()
        # self._p_debug = (
        #     self._backend.get_fk("left_elbo").position.astype(WBC9.dtype).copy()
        # )

        # J_v_elbo[:, 2 + 3 :] = 0

        self.task_objects["l_vel"].update(J_v_mixed, self._v_goal)
        self.task_objects["a_vel"].update(J_v_mixed, self._w_goal)
        self.task_objects["ddq_damp"].update(self._dq_prev)
        self.task_objects["dddq_damp"].update(self._dq_prev)
        self.constraint_objects["safety"].update(self._q_curr)
        self.constraint_objects["l_vel_goal"].set_active(False)
        if "l_vel_goal" in self.constraint_objects:
            self.constraint_objects["l_vel_goal"].update(
                J_v_mixed[:3],
                self._v_goal - self._v_goal_relax,
                self._v_goal + self._v_goal_relax,
            )

        self.task_objects["gravity_bias"].update(J_v_elbo[:3], self._dir_elbo_pull)

        if "self_collision" in self.constraint_objects:
            self.constraint_objects["self_collision"].update(self._J_col, self._d_col)
            if np.min(self._d_col) > MAGIC_NUMBER_COLLISION_OBB_OFFSET:
                self.constraint_objects["self_collision"].set_active(False)
            else:
                self.constraint_objects["self_collision"].set_active(True)
        # print("print(J_v_elbo)")
        # print(J_v_elbo.flags)
        # print(self.constraint_objects["l_vel_goal"].is_active())
        # print(self._q_curr)

    def solve(self) -> int:
        self._dq = self.controller.solve()
        if np.isnan(self._dq).any():
            return -6
        # print((self.controller.solve()),)
        # print(self._J_point_3d @ self._dq)
        # if (solver.sate)

        return self.controller.solver_status()
