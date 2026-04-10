import numpy as np
import ampl
from ampl import Tf
import pywbc
from .col import *
from .common import *
from .base import Col, Kin, ColGroup, FrameEnum, RobotInterface
from .wbc import WBC9


class Col_T2DA2_AMPL(Col):
    def __init__(self, **kwargs):

        from .data_hb11 import (
            DICT_PACKED_SPHERES_HB11_LEFT as DICT_SPHS_L,
            DICT_PACKED_SPHERES_HB11_RIGHT as DICT_SPHS_R,
            OBB3_TORSO_2 as DICT_OBB_T,
            AAABB_WORKSPACE as DICT_AAABB_W,
        )

        # from .data_tianji import (
        #     DICT_PACKED_SPHERES_TIANJI_LEFT as DICT_SPHS_L,
        #     DICT_PACKED_SPHERES_TIANJI_RIGHT as DICT_SPHS_R,
        # )

        self._debug_col = None
        self._obb_T = CollisionObjectOBB(
            DICT_OBB_T["center"],
            DICT_OBB_T["half_extents"],
            DICT_OBB_T["u"],
            DICT_OBB_T["v"],
            DICT_OBB_T["w"],
        )
        self._aaabb_W = CollisionObjectAAABB(
            xyz_min=DICT_AAABB_W["xyz_min"], xyz_max=DICT_AAABB_W["xyz_max"]
        )
        self._vsph_L = CollisionObjectVSph([v for _, v in DICT_SPHS_L.items()])
        self._vsph_R = CollisionObjectVSph([v for _, v in DICT_SPHS_R.items()])
        self._ids_sph_LR = [3, 4, 5, 6]

    def update_pose(
        self,
        fk_rwt_T: np.ndarray = None,
        fk_rwt_L: np.ndarray = None,
        fk_rwt_R: np.ndarray = None,
    ) -> None:
        if fk_rwt_T is not None:
            self._obb_T.pose = fk_rwt_T[1]
        if fk_rwt_R is not None:
            self._vsph_R.set_pose(np.copy(fk_rwt_R), self._ids_sph_LR)
        if fk_rwt_L is not None:
            self._vsph_L.set_pose(np.copy(fk_rwt_L), self._ids_sph_LR)
        return

    def collision_free_self(self, group: ColGroup) -> ColGroup:
        col_status = ColGroup.NONE
        if group & ColGroup.T_R:
            for i in self._ids_sph_LR:
                if ampl.collision_check_obb_vsph(
                    self._obb_T.entity_move, self._vsph_R.entity_move, i
                ):
                    col_status = col_status | ColGroup.T_R
                    break
        if group & ColGroup.T_L:
            for i in self._ids_sph_LR:
                if ampl.collision_check_obb_vsph(
                    self._obb_T.entity_move, self._vsph_L.entity_move, i
                ):
                    col_status = col_status | ColGroup.T_L
                    break
        if group & ColGroup.L_R:
            for iL in self._ids_sph_LR:
                for iR in self._ids_sph_LR:
                    if ampl.collision_check_vsph_vsph(
                        self._vsph_L.entity_move, iL, self._vsph_R.entity_move, iR
                    ):
                        col_status = col_status | ColGroup.L_R
                        break

        if group & ColGroup.W_L:
            for iL in self._ids_sph_LR:
                if ampl.collision_check_aaabb_vsph(
                    self._aaabb_W.entity_move, self._vsph_L.entity_move, iL
                ):
                    col_status = col_status | ColGroup.W_L
                    break
        if group & ColGroup.W_R:
            for iR in self._ids_sph_LR:
                if ampl.collision_check_aaabb_vsph(
                    self._aaabb_W.entity_move, self._vsph_R.entity_move, iR
                ):
                    col_status = col_status | ColGroup.W_R
                    break
        return col_status

    # def collision_free_self(self, group: ColGroup) -> ColGroup:
    #     col_status = ColGroup.NONE
    #     if group & ColGroup.W_L:
    #         for i in self._ids_sph_LR:
    #             if ampl.collision_check_obb_vsph(
    #                 self._obb_T.entity_move, self._vsph_R.entity_move, i
    #             ):
    #                 col_status = col_status | ColGroup.T_R
    #                 break
    #     return col_status

    # def collision_gradient_self_(self, group: ColGroup):
    #     if group & ColGroup.T_L:
    #         vsph = self._vsph_L
    #         for i in [3, 4, 5]:  # self._ids_sph_LR:
    #             ampl.collision_gradient_obb_vsph_pd(
    #                 self._obb_T.entity_move, vsph.entity_move, i
    #             )
    #         ampl.collision_visualize_object(
    #             vsph.entity_move, vsph.centers, vsph.grads, vsph.dists, vsph.labels
    #         )
    #         dict_topk = pywbc.top_k(vsph.dists, vsph.labels, [3, 4, 5], 1)
    #         if dict_topk:
    #             mask = list(dict_topk.values())
    #             c = vsph.centers[mask].reshape((-1, 3))
    #             b = c + vsph.grads[mask].reshape((-1, 3))
    #             return np.hstack([b, c])
    #     return None
    @classmethod
    def _collision_detail_self(
        cls, vsph: CollisionObjectVSph, ids_arm: list[int], top_k: int = 1
    ):
        ampl.collision_detail(
            vsph.entity_move, vsph.centers, vsph.grads, vsph.dists, vsph.labels
        )
        dict_topk = pywbc.top_k(vsph.dists, vsph.labels, ids_arm, top_k)
        if dict_topk:
            mask = list(dict_topk.values())
            p = vsph.centers[mask].reshape((-1, 3))
            g = vsph.grads[mask].reshape((-1, 3))
            d = vsph.dists[mask].flatten()
            l = vsph.labels[mask].flatten()
            return p, g, d, l

    def collision_gradient_self(self, group: ColGroup):
        ids_self = self._ids_sph_LR
        if group == ColGroup.T_L or group == ColGroup.W_L:
            vsph = self._vsph_L
        elif group == ColGroup.T_R or group == ColGroup.W_R:
            vsph = self._vsph_R
        # if group == ColGroup.W_L:
        #     vsph = self._vsph_L
        #     for i in ids_self:  # self._ids_sph_LR:
        #         ampl.collision_gradient_aaabb_vsph_pd(
        #             self._obb_T.entity_move, vsph.entity_move, i
        #         )
        #     ampl.collision_detail(
        #         vsph.entity_move, vsph.centers, vsph.grads, vsph.dists, vsph.labels
        #     )
        #     return None

        for i in ids_self:  # self._ids_sph_LR:
            ampl.collision_gradient_aaabb_vsph_pd(
                self._aaabb_W.entity_move, vsph.entity_move, i
            )
        info_self_wall = Col_T2DA2_AMPL._collision_detail_self(vsph, ids_self, 1)
        # return info_self_wall
        for i in ids_self:  # self._ids_sph_LR:
            ampl.collision_gradient_obb_vsph_pd(
                self._obb_T.entity_move, vsph.entity_move, i
            )
        info_self_torso = Col_T2DA2_AMPL._collision_detail_self(vsph, ids_self, 1)

        if info_self_wall is None:
            return info_self_torso

        if info_self_torso is None:
            return info_self_wall
        # ggg = (
        #     np.vstack([info_self_torso[0], info_self_wall[0]]),
        #     np.vstack([info_self_torso[1], info_self_wall[1]]),
        #     np.vstack([info_self_torso[2], info_self_wall[2]]),
        #     np.vstack([info_self_torso[3], info_self_wall[3]]),
        # )
        # print(ggg)
        return (
            np.vstack([info_self_torso[0], info_self_wall[0]]),
            np.vstack([info_self_torso[1], info_self_wall[1]]),
            np.hstack([info_self_torso[2], info_self_wall[2]]),
            np.hstack([info_self_torso[3], info_self_wall[3]]),
        )

        # elif group == ColGroup.T_R:
        #     vsph = self._vsph_R
        #     for i in ids_self:  # self._ids_sph_LR:
        #         ampl.collision_gradient_obb_vsph_pd(
        #             self._obb_T.entity_move, vsph.entity_move, i
        #         )
        #     ampl.collision_detail(
        #         vsph.entity_move, vsph.centers, vsph.grads, vsph.dists, vsph.labels
        #     )
        #     dict_topk = pywbc.top_k(vsph.dists, vsph.labels, ids_self, 1)
        #     if dict_topk:
        #         mask = list(dict_topk.values())
        #         p = vsph.centers[mask].reshape((-1, 3))
        #         g = vsph.grads[mask].reshape((-1, 3))
        #         d = vsph.dists[mask].flatten()
        #         return p, g, d, vsph.labels[mask].flatten()
        return None


class Kin_T2DA2_AMPL(Kin):
    DOF: int = 16
    DOF_L: int = 7
    DOF_R: int = 7
    DOF_T: int = 2
    I_T: int = 0
    I_L: int = 2
    I_R: int = 9

    def __init__(
        self, name_arm_left: str, name_arm_right: str, name_torso: str, **kwargs
    ):

        self._kin_L = ampl.ArmBase(name_arm_left, ampl.ArmType.Humanoid7, 7)
        self._kin_R = ampl.ArmBase(name_arm_right, ampl.ArmType.Humanoid7, 7)
        self._kin_T = ampl.ArmBase(name_torso, ampl.ArmType.Extern_TZRX, 2)
        self._q8_L = np.zeros((8, 7), dtype=np.float64)
        self._q8_R = np.zeros((8, 7), dtype=np.float64)
        self._q2_T = np.zeros((2, 2), dtype=np.float64)

        self._redundant_L = [
            self._kin_L.joint_limits()[0][6],
            self._kin_L.joint_limits()[1][6],
        ]
        self._redundant_R = [
            self._kin_R.joint_limits()[0][6],
            self._kin_R.joint_limits()[1][6],
        ]

        self._fk_L = np.zeros((7 + 1, 7), dtype=DTypeDouble)
        self._fk_R = np.zeros((7 + 1, 7), dtype=DTypeDouble)
        self._fk_T = np.zeros((2 + 2, 7), dtype=DTypeDouble)
        self._Js_L = np.zeros(
            (6, Kin_T2DA2_AMPL.DOF_T + Kin_T2DA2_AMPL.DOF_L),
            order="F",
            dtype=DTypeDouble,
        )
        self._Js_R = np.zeros(
            (6, Kin_T2DA2_AMPL.DOF_T + Kin_T2DA2_AMPL.DOF_R),
            order="F",
            dtype=DTypeDouble,
        )
        self._Js_T = np.zeros((6, Kin_T2DA2_AMPL.DOF_T), order="F", dtype=DTypeDouble)
        self._q_T = np.zeros(Kin_T2DA2_AMPL.DOF_T, dtype=DTypeDouble)
        self._q_L = np.zeros(Kin_T2DA2_AMPL.DOF_L, dtype=DTypeDouble)
        self._q_R = np.zeros(Kin_T2DA2_AMPL.DOF_R, dtype=DTypeDouble)
        self._qmap_L = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
        self._qmap_R = np.array([0, 1, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint32)

    def get_limits(self):
        return (
            self._kin_T.joint_limits()[0]
            + self._kin_L.joint_limits()[0]
            + self._kin_R.joint_limits()[0],
            self._kin_T.joint_limits()[1]
            + self._kin_L.joint_limits()[1]
            + self._kin_R.joint_limits()[1],
        )

    def update_kin(
        self, q: np.ndarray, pose_base: Tf = None, update_jacobian: bool = False
    ) -> None:
        if pose_base is not None:
            self._kin_T.set_base(pose_base.matrix)

        self._q_T = q[
            Kin_T2DA2_AMPL.I_T : Kin_T2DA2_AMPL.I_T + Kin_T2DA2_AMPL.DOF_T
        ].astype(np.float64)
        self._q_L = q[
            Kin_T2DA2_AMPL.I_L : Kin_T2DA2_AMPL.I_L + Kin_T2DA2_AMPL.DOF_L
        ].astype(np.float64)
        self._q_R = q[
            Kin_T2DA2_AMPL.I_R : Kin_T2DA2_AMPL.I_R + Kin_T2DA2_AMPL.DOF_R
        ].astype(np.float64)
        self._kin_T.fk_links(self._q_T, self._fk_T)
        self._kin_L.set_base(ampl.qt7_to_tf44(self._fk_T[2]))
        self._kin_R.set_base(ampl.qt7_to_tf44(self._fk_T[3]))
        self._kin_L.fk_links(self._q_L, self._fk_L)
        self._kin_R.fk_links(self._q_R, self._fk_R)
        if not update_jacobian:
            return
        self._kin_T.jacobian(self._fk_T, self._Js_L)
        np.copyto(self._Js_R[:, :2], self._Js_L[:, :2])
        self._kin_L.jacobian(self._fk_L, self._Js_L[:, 2:])
        self._kin_R.jacobian(self._fk_R, self._Js_R[:, 2:])

    def get_fk(self, frame: FrameEnum) -> Tf:
        if frame == FrameEnum.FRAME_TACTILE_L:
            return Tf(self._fk_L[-1])
        if frame == FrameEnum.FRAME_TACTILE_R:
            return Tf(self._fk_R[-1])
        if frame == FrameEnum.FRAME_ELBOW_L:
            return Tf(self._fk_L[3])
        if frame == FrameEnum.FRAME_ELBOW_R:
            return Tf(self._fk_R[3])
        if frame == FrameEnum.FRAME_TORSO_2:
            return Tf(self._fk_T[1])

    def get_jacobian(self, frame: FrameEnum) -> np.ndarray:
        if frame == FrameEnum.FRAME_TACTILE_L:
            return self._Js_L
        if frame == FrameEnum.FRAME_TACTILE_R:
            return self._Js_R

    def get_qmap(self, frame: FrameEnum) -> np.ndarray:
        if frame == FrameEnum.FRAME_TACTILE_L:
            return self._qmap_L
        if frame == FrameEnum.FRAME_TACTILE_R:
            return self._qmap_R

    def get_ik(
        self,
        tf_target: Tf,
        frame: FrameEnum,
        nb_redundant_search: int = 512,
        range_redundant_last_joint: Union[List[np.ndarray], np.ndarray] = None,
        which_iks: List[int] = [5, 7],
    ):

        tf44_tool0 = tf_target.matrix.copy(order="C")
        ik_result = {}

        if frame == FrameEnum.FRAME_TACTILE_L:
            q8 = self._q8_L
            if range_redundant_last_joint is None:
                rlo = self._redundant_L[0]
                rhi = self._redundant_L[1]
            else:
                rlo = range_redundant_last_joint[0]
                rhi = range_redundant_last_joint[1]
            kin = self._kin_L

        elif frame == FrameEnum.FRAME_TACTILE_R:
            q8 = self._q8_R
            if range_redundant_last_joint is None:
                rlo = self._redundant_R[0]
                rhi = self._redundant_R[1]
            else:
                rlo = range_redundant_last_joint[0]
                rhi = range_redundant_last_joint[1]
            kin = self._kin_R

        else:
            return {}

        qs_search = np.linspace(
            rlo, rhi, nb_redundant_search + 1, True, dtype=DTypeDouble
        )
        ikq_batch = np.zeros((len(qs_search), Kin_T2DA2_AMPL.DOF_L), dtype=DTypeDouble)
        iks_batch = np.zeros(len(qs_search), dtype=DTypeBool)
        for which_ik in which_iks:
            # print(which_ik)
            for iq, ql in enumerate(qs_search):
                q8[:, -1] = ql
                ik_status = kin.ik(tf44_tool0, q8)
                if (ik_status >> which_ik) & 1:
                    iks_batch[iq] = True
                    np.copyto(ikq_batch[iq], q8[which_ik])
            ik_result[which_ik] = (iks_batch.copy(), ikq_batch.copy())
        return ik_result


class Robot_T2DA2(Col_T2DA2_AMPL, Kin_T2DA2_AMPL, RobotInterface):
    def __init__(
        self,
        name_arm_left: str,
        name_arm_right: str,
        name_torso: str,
        path_to_config_wbc: str,
        ndof: int,
        **kwargs
    ):
        # super().__init__(**kwargs)
        Col_T2DA2_AMPL.__init__(self)
        Kin_T2DA2_AMPL.__init__(self, name_arm_left, name_arm_right, name_torso)
        self._wbc_L = WBC9(path_to_config_wbc, self)
        self._wbc_R = WBC9(path_to_config_wbc, self)
        self.ndof = ndof
        self.q = np.zeros(ndof, dtype=DTypeDouble)
        self._elbow_prior_L = np.array([0.25, 0.5, 0.5])
        self._elbow_prior_R = np.array([0.25, -0.5, 0.5])
        constraint = next(
            (
                obj
                for obj in self._wbc_L.config.constraints
                if obj.type == "joint_position_speed_limits"
            ),
            None,
        )
        self.q_max = constraint.q_max
        self.q_min = constraint.q_min

        # print(path_to_config_wbc)

    def set_wall(
        self,
        x_wall: Union[List[float], np.ndarray] = [0.0, 1.5],
        y_wall: Union[List[float], np.ndarray] = [-1.0, 1.0],
        z_wall: Union[List[float], np.ndarray] = [0.6, 1.8],
    ):
        self._aaabb_W.update_parameter(
            [x_wall[0], y_wall[0], z_wall[0]], [x_wall[1], y_wall[1], z_wall[1]]
        )

    def wall(self):
        tmin = self._aaabb_W.entity_move.xyz_min
        tmax = self._aaabb_W.entity_move.xyz_max
        return ([tmin[0], tmax[0]], [tmin[1], tmax[1]], [tmin[2], tmax[2]])

    def get_arm_bound(self, frame: FrameEnum, which_bound: str = "z_min"):
        if len(self._ids_sph_LR) == 0:
            return None
        if frame == FrameEnum.FRAME_TACTILE_L:
            arm_sphs = self._vsph_L.entity_move
        elif frame == FrameEnum.FRAME_TACTILE_R:
            arm_sphs = self._vsph_R.entity_move

        MAGIC_NUMBER_VALID_RADIUS = -1e-3  # TODO: GET RID OF IT
        key = which_bound[0]
        s = -1.0 if "min" in which_bound else 1.0
        g = arm_sphs.get_group(self._ids_sph_LR[0])
        mask = g["r"] > MAGIC_NUMBER_VALID_RADIUS

        # print(g["r"], "r")

        if s > 0:
            best = np.max(g[key][mask] + s * g["r"][mask])
            for id_l in self._ids_sph_LR[1:]:
                g = arm_sphs.get_group(id_l)
                mask = g["r"] > MAGIC_NUMBER_VALID_RADIUS
                best = max(best, np.max(g["z"][mask] + s * g["r"][mask]))
        else:
            best = np.min(g[key][mask] + s * g["r"][mask])
            for id_l in self._ids_sph_LR[1:]:
                g = arm_sphs.get_group(id_l)
                mask = g["r"] > MAGIC_NUMBER_VALID_RADIUS
                best = min(best, np.min(g["z"][mask] + s * g["r"][mask]))
        return best

    def update_col_self(self):
        self.update_pose(fk_rwt_T=self._fk_T, fk_rwt_R=self._fk_R, fk_rwt_L=self._fk_L)
        return

    #
    #  def get_fk(self, frame: FrameEnum) -> Tf:
    #    return

    def check_self_collision(
        self, colgroup: ColGroup = ColGroup.ALL_SELF | ColGroup.ALL_W
    ) -> ColGroup:
        return self.collision_free_self(colgroup)

    def track_tcp(self, which_hand: FrameEnum, tf_target: Tf, substeps: int = 3):
        q = np.copy(self.q)
        dt_sub = 1.0 / float(substeps)
        if which_hand == FrameEnum.FRAME_TACTILE_L:
            wbc = self._wbc_L
            elbow = self._elbow_prior_L
            frame_elbo = FrameEnum.FRAME_ELBOW_L
            col_group = ColGroup.T_L
        elif which_hand == FrameEnum.FRAME_TACTILE_R:
            wbc = self._wbc_R
            elbow = self._elbow_prior_R
            frame_elbo = FrameEnum.FRAME_ELBOW_R
            col_group = ColGroup.T_R
        else:
            return q
        # print(q)
        # print(self.kin.get_qmap(which_hand))
        for _ in range(substeps):
            self.update_kin(q, None, True)
            self.update_col_self()
            wbc._q_curr = q[self.get_qmap(which_hand)]
            wbc.update_state(wbc._q_curr)
            wbc.set_goal(self.get_fk(which_hand), tf_target)
            wbc._dir_elbo_pull = elbow - self.get_fk(frame_elbo).position
            info_col = self.collision_gradient_self(col_group)
            wbc.update_collision_gradient(
                which_hand, info_col[0], info_col[1], info_col[2], info_col[3]
            )
            wbc.update_controller(which_hand)
            sol_status = wbc.solve()
            #
            if sol_status:
                q[self.get_qmap(which_hand)] = np.clip(
                    wbc._q_curr + wbc._dq * dt_sub, self.q_min, self.q_max
                ).astype(np.float64)
                np.copyto(wbc._dq_prev, wbc._dq)

        return q
