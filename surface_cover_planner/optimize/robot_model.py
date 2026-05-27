"""AMPL agent wrapper for the trajopt post-processor (KinHB11 + SphereBVHScene).

Owns a single `ampl.KinHB11` instance (built by `ampl_motion.create_robot_system("hb11")`)
plus its sphere-BVH collision scene. Exposes FK, analytic position Jacobian,
and — new — per-pair signed distance + analytic Jacobian for self/wall
collisions. Torso (rail + waist) and the off-arm are frozen at `initial_q`
for the lifetime of the wrapper, mirroring `tools/surface_cover_planner/robot_check.py`.

Outside the SudoDeploy docker container the imports here will fail
(`ampl`, `ampl_motion` come from container-only wheels). Module imports are
deferred to the constructor so callers can introspect this file without a
working AMPL install.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Mirrors HBMP_COMPONENT_DOF_INDICES in robot_check.py:47-52. The KinHB11
# q vector is laid out [torso(2), left_arm(7), right_arm(7)] — same as what
# `Robot_T2DA2` used.
HBMP_COMPONENT_DOF_INDICES: dict[str, list[int]] = {
    "rail": [0],
    "waist": [1],
    "left_arm": list(range(2, 9)),
    "right_arm": list(range(9, 16)),
}


@dataclass
class RobotConfig:
    arm: str  # "left_arm" | "right_arm"
    initial_q: Sequence[float]  # length 16
    # The wheel ships robot geometry / wbc data for hb11 internally — these
    # legacy fields are kept so old configs load without churn but are no
    # longer load-bearing. A non-default value emits a one-line deprecation
    # notice from RobotModel.__init__.
    hbmp_arm_left: str = "hb11_left"
    hbmp_arm_right: str = "hb11_right"
    hbmp_torso: str = "hb11_torso"
    wall: dict | None = None
    tool_offset_4x4: np.ndarray | None = None  # (4,4) or None == identity
    q_lo_arm: Sequence[float] | None = None
    q_hi_arm: Sequence[float] | None = None
    joint_vel_limits: Sequence[float] | None = None
    # SDF safety + warning distance for the per-pair self/wall collision
    # constraint. `d_safe` is the hard floor used by the optimizer; `d_warn` is
    # the broad-phase margin (pairs farther than this aren't even returned).
    # `top_k` is the max contacts retained per macro-region (1 mirrors the
    # WBC). Defaults are conservative for the hb11 surface-cover use case.
    collision_d_safe: float = 0.005
    collision_d_warn: float = 0.050
    collision_top_k: int = 1


# `ampl_motion-0.0.2` has type hints `NDArray[np.bool]` that crash on
# NumPy ≥ 1.20. Re-add the alias as a module attribute before importing
# the wheel. NumPy's `__getattr__` raises on the *read* path so
# `hasattr(np, "bool")` returns False, but `setattr` bypasses it — the
# assignment sticks and `NDArray[np.bool]` at import time resolves.
import numpy as _np_for_patch

if not hasattr(_np_for_patch, "bool"):
    _np_for_patch.bool = bool  # type: ignore[attr-defined]
del _np_for_patch


def _ensure_ampl_motion_imports() -> None:
    """Idempotent re-application of the np.bool patch (callers that reload
    numpy or wipe attrs are unlikely but cheap to guard against)."""
    import numpy

    if not hasattr(numpy, "bool"):
        numpy.bool = bool  # type: ignore[attr-defined]


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q_xyzw = R.from_matrix(rot).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


class RobotModel:
    """KinHB11 + SphereBVHScene + per-pair SDF.

    Each FK / Jacobian / collision query assembles a 16-DOF q_full from
    `initial_q` with the active arm slice replaced, calls
    `kin.update_kin(q_full, None, update_jacobian)`, refreshes the colsph
    pose buffer, and reads back what the caller asked for.

    The active arm's index range is `[arm_lo, arm_hi)`; everything outside
    is frozen.
    """

    def __init__(self, cfg: RobotConfig) -> None:
        _ensure_ampl_motion_imports()
        import ampl  # noqa: F401  (verifies the wheel is installed)
        import ampl_motion as am

        if cfg.arm not in ("left_arm", "right_arm"):
            raise ValueError(
                f"cfg.arm must be 'left_arm' or 'right_arm', got {cfg.arm!r}"
            )
        initial_q = np.asarray(cfg.initial_q, dtype=np.float64)
        if initial_q.shape != (16,):
            raise ValueError(
                f"cfg.initial_q must have length 16, got {initial_q.shape}"
            )

        # Legacy YAML fields are no-ops now that the wheel embeds hb11
        # geometry. Warn loudly if a caller is still trying to override.
        for label, val, default in (
            ("hbmp_arm_left", cfg.hbmp_arm_left, "hb11_left"),
            ("hbmp_arm_right", cfg.hbmp_arm_right, "hb11_right"),
            ("hbmp_torso", cfg.hbmp_torso, "hb11_torso"),
        ):
            if val != default:
                print(
                    f"[robot_model] ignoring deprecated robot.{label}={val!r}; "
                    f"ampl_motion embeds the hb11 robot identity"
                )

        self.cfg = cfg
        self._initial_q = initial_q
        self.arm_indices = HBMP_COMPONENT_DOF_INDICES[cfg.arm]
        self._arm_lo = self.arm_indices[0]
        self._arm_hi = self.arm_indices[-1] + 1

        (
            self._meta,
            self._kin,
            self._colcvx,   # unused — kept for future swap to convex SDF
            self._colsph,
            self._wbc,      # unused outside ampl_motion's WBC pipeline
            self._link_to_id,
            self._ignores,
            self._cvx_data,
            self._sph_data,
        ) = am.create_robot_system("hb11")
        am.update_ignore(self._colsph, self._link_to_id, self._ignores)
        am.update_ignore(self._colcvx, self._link_to_id, self._ignores)

        # Frame for tool-tip FK. KinHB11's tactile-center is the wheel's
        # equivalent of HBMP's FRAME_TACTILE_L/R.
        if cfg.arm == "left_arm":
            self._frame = self._kin.Frames.left_link_tactile_center
        else:
            self._frame = self._kin.Frames.right_link_tactile_center

        # Wire collision-topology so `fill_contact_jacobians` can project
        # per-link contact normals onto each DOF. Mirrors
        # `set_colsph_topology_from_wbc` (util_hb.py:316) but pulls from
        # `meta.topology` (loaded from the wheel's meta.json) so we don't
        # need to spin up a WBC controller just for this side effect.
        macro_region_sph = [[]] + list(self._meta.topology.macro_regions)
        joint_to_region_sph = [0] + [s + 1 for s in self._meta.topology.joint_to_region]
        self._colsph.set_region_mapping(macro_region_sph, joint_to_region_sph)

        # Workspace AABB (the new home for the old `wall` field). HBMP's
        # `agent.set_wall(...)` only enforces the box on the *arm* links —
        # the torso sits below `z_min` (typically the surface-cover working
        # plane at z=0.6 m) and would otherwise flag every pose. Match
        # HBMP's behavior: include only `left-*` / `right-*` (the two arm
        # chains, including hand / tactile frames). base_link and torso
        # links are excluded.
        if cfg.wall:
            min3 = [
                float(cfg.wall.get("x", [-np.inf, np.inf])[0]),
                float(cfg.wall.get("y", [-np.inf, np.inf])[0]),
                float(cfg.wall.get("z", [-np.inf, np.inf])[0]),
            ]
            max3 = [
                float(cfg.wall.get("x", [-np.inf, np.inf])[1]),
                float(cfg.wall.get("y", [-np.inf, np.inf])[1]),
                float(cfg.wall.get("z", [-np.inf, np.inf])[1]),
            ]
            includes = [
                name for name in self._link_to_id
                if name.startswith("left-") or name.startswith("right-")
            ]
            am.configure_workspace(
                self._colsph, self._link_to_id, includes, min3=min3, max3=max3
            )
            am.configure_workspace(
                self._colcvx, self._link_to_id, includes, min3=min3, max3=max3
            )

        # Initial kin update so subsequent FK / collision queries operate on
        # a primed state.
        self._kin.update_kin(self._initial_q, None, True)
        self._colsph.update_poses(1, self._kin.get_fk_actuated_frame())

        if cfg.tool_offset_4x4 is None:
            self._tool_offset = None
            self._tool_offset_inv = None
        else:
            tof = np.asarray(cfg.tool_offset_4x4, dtype=np.float64)
            if tof.shape != (4, 4):
                raise ValueError(f"tool_offset_4x4 must be 4x4, got {tof.shape}")
            self._tool_offset = tof
            self._tool_offset_inv = np.linalg.inv(tof)

        self._q_lo = (
            np.full(7, -np.pi, dtype=np.float64)
            if cfg.q_lo_arm is None
            else np.asarray(cfg.q_lo_arm, dtype=np.float64)
        )
        self._q_hi = (
            np.full(7, np.pi, dtype=np.float64)
            if cfg.q_hi_arm is None
            else np.asarray(cfg.q_hi_arm, dtype=np.float64)
        )
        if self._q_lo.shape != (7,) or self._q_hi.shape != (7,):
            raise ValueError("q_lo_arm / q_hi_arm must have length 7")

        if cfg.joint_vel_limits is None:
            self._v_max = np.array([1.0] * 7, dtype=np.float64)
        else:
            v = np.asarray(cfg.joint_vel_limits, dtype=np.float64)
            if v.shape != (7,):
                raise ValueError(f"joint_vel_limits must have length 7, got {v.shape}")
            self._v_max = v

        self._scratch_q_full = initial_q.copy()

        # Pre-allocated buffers for `fill_contact_jacobians`. Mirroring
        # ampl_motion's WBC layout exactly (empirically required — the
        # call's signature claims `J_cbf_out: shape=(*, *)` but a too-small
        # or wrong-order J_cbf_out trips nanobind's overload matcher with
        # an unhelpful "incompatible function arguments" message). The WBC
        # for hb11 allocates `_J_cbf` as (50, 16) F-order and `_d_cbf` as
        # (50,) — generous upper bound covering internal-pair expansion
        # past the user-visible `max_top_k`. We adopt the same sizing.
        self._num_regions = len(self._meta.topology.macro_regions)
        self._max_contact_rows = 50  # matches wbc._J_cbf.shape[0] for hb11
        self._J_cbf_buf = np.zeros(
            (self._max_contact_rows, self._kin.DOF_TOTAL),
            dtype=np.float64,
            order="F",
        )
        self._d_cbf_buf = np.zeros(self._max_contact_rows, dtype=np.float64)
        # Pre-allocated F-order kin-Jacobian buffer. Mirrors
        # `np.copyto(wbc._Js, kin.get_jacobian_actuated_frame())` in
        # `ampl_motion/util_da.py:282`. Allocating once F-order + copyto
        # per call avoids `np.asfortranarray` returning a view nanobind
        # rejects.
        self._J_kin_buf = np.zeros(
            (6, self._kin.DOF_TOTAL), dtype=np.float64, order="F"
        )

    # ---------------- public properties ----------------

    @property
    def initial_q(self) -> np.ndarray:
        return self._initial_q.copy()

    @property
    def q_lo(self) -> np.ndarray:
        return self._q_lo.copy()

    @property
    def q_hi(self) -> np.ndarray:
        return self._q_hi.copy()

    @property
    def v_max(self) -> np.ndarray:
        return self._v_max.copy()

    @property
    def tool_offset_4x4(self) -> np.ndarray | None:
        return None if self._tool_offset is None else self._tool_offset.copy()

    @property
    def collision_d_safe(self) -> float:
        return float(self.cfg.collision_d_safe)

    @property
    def collision_d_warn(self) -> float:
        return float(self.cfg.collision_d_warn)

    @property
    def max_contact_rows(self) -> int:
        return self._max_contact_rows

    @property
    def arm_dof_slice(self) -> tuple[int, int]:
        return self._arm_lo, self._arm_hi

    # ---------------- internal helpers ----------------

    def assemble_q_full(self, q_arm: np.ndarray) -> np.ndarray:
        """Return a 16-DOF state with the active arm slice replaced by q_arm."""
        q_full = self._scratch_q_full
        np.copyto(q_full, self._initial_q)
        q_full[self.arm_indices] = q_arm
        return q_full.copy()

    def _update_state(self, q_arm: np.ndarray, with_jacobian: bool) -> np.ndarray:
        """Update kin + colsph for `q_arm`; return the 16-DOF q_full."""
        q_full = self.assemble_q_full(q_arm)
        self._kin.update_kin(q_full, None, with_jacobian)
        self._colsph.update_poses(1, self._kin.get_fk_actuated_frame())
        return q_full

    def _tool_pose_from_kin(self) -> tuple[np.ndarray, np.ndarray]:
        gripper_fk = np.asarray(
            self._kin.get_fk(self._frame).matrix, dtype=np.float64
        )
        if self._tool_offset is None:
            return gripper_fk[:3, 3].copy(), _rotmat_to_quat_wxyz(gripper_fk[:3, :3])
        T_tool = gripper_fk @ self._tool_offset
        return T_tool[:3, 3].copy(), _rotmat_to_quat_wxyz(T_tool[:3, :3])

    # ---------------- FK ----------------

    def fk_pose(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._update_state(q_arm, with_jacobian=False)
        return self._tool_pose_from_kin()

    def fk_position(self, q_arm: np.ndarray) -> np.ndarray:
        return self.fk_pose(q_arm)[0]

    def fk_quat_wxyz(self, q_arm: np.ndarray) -> np.ndarray:
        return self.fk_pose(q_arm)[1]

    def fk_pos_and_jac(
        self, q_arm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward kinematics + analytic position Jacobian for the active arm.

        Returns:
            pos      : (3,) tool-tip position in world frame.
            quat_wxyz: (4,) tool-tip orientation.
            J_pos    : (3, 7) `d(tool_pos) / d(q_arm)`.

        KinHB11's `get_jacobian(frame)` returns the (6, 7) arm-only Jacobian
        directly (3 lin + 3 ang rows × 7 active-arm cols), which is cleaner
        than Robot_T2DA2's (6, 9) layout that mixed in torso cols.

        Tool offset chain rule:
            T_tool = T_gripper @ T_offset
            d(t_tool)/dq = J_lin_arm − skew(R_gripper · t_offset_local) · J_ang_arm
        """
        self._update_state(q_arm, with_jacobian=True)
        gripper_fk = np.asarray(
            self._kin.get_fk(self._frame).matrix, dtype=np.float64
        )
        J = np.asarray(self._kin.get_jacobian(self._frame), dtype=np.float64)
        J_lin_arm = J[:3, :]  # (3, 7)
        J_ang_arm = J[3:, :]  # (3, 7)

        if self._tool_offset is None:
            return (
                gripper_fk[:3, 3].copy(),
                _rotmat_to_quat_wxyz(gripper_fk[:3, :3]),
                J_lin_arm.copy(),
            )

        T_off = self._tool_offset
        t_off = T_off[:3, 3]
        R_gripper = gripper_fk[:3, :3]
        T_tool = gripper_fk @ T_off
        offset_world = R_gripper @ t_off
        skew = np.array([
            [0.0, -offset_world[2], offset_world[1]],
            [offset_world[2], 0.0, -offset_world[0]],
            [-offset_world[1], offset_world[0], 0.0],
        ])
        J_pos_tool = J_lin_arm - skew @ J_ang_arm
        return (
            T_tool[:3, 3].copy(),
            _rotmat_to_quat_wxyz(T_tool[:3, :3]),
            J_pos_tool,
        )

    def fk_pos_jac_and_collision(
        self, q_arm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Single-`update_kin` evaluation: pos + quat + J_pos + collision-bool.

        Used by trajopt's analytic-gradient path. The collision-bool here is
        the cheap sphere-BVH binary (`compute_collisions(0.0)`) — fine for
        cost-grad bookkeeping where the value is discarded anyway. The
        definitive mesh-based binary lives in `self_collides()` and is what
        `n_collisions_after` reports.
        """
        pos, quat, J = self.fk_pos_and_jac(q_arm)
        coll = (
            len(self._colsph.compute_collisions(0.0)) > 0
            or len(self._colsph.compute_wall_collisions(0.0)) > 0
        )
        return pos, quat, J, bool(coll)

    # ---------------- collision ----------------

    def self_collides(self, q_arm: np.ndarray) -> bool:
        """Binary post-solve audit using the **mesh** check (`colcvx`).

        Matches the semantics of the prior `Robot_T2DA2.check_self_collision()` —
        the definitive answer for hardware sign-off. The sphere-BVH SDF used
        by the trajopt constraint is conservative (sphere ≤ mesh), so a state
        can have `dist_sph < 0` while being mesh-collision-free; the binary
        report needs the mesh check, not the sphere.

        Mirrors `ampl_motion.collision_free_cvx` (util_hb.py:133) but is
        inverted to return `True` when colliding.
        """
        import ampl  # noqa: F401

        self._kin.update_kin(self.assemble_q_full(q_arm), None, False)
        # colcvx needs per-link FK pushed in; copy from kin's actuated-frame buffer.
        fk_actuated = self._kin.get_fk_actuated_frame()  # (16, 7)
        frames = self._kin.get_actuated_frames()
        for i, frame_name in enumerate(frames):
            self._colcvx.update_pose(frame_name, fk_actuated[i])
        if not self._colcvx.is_workspace_safe():
            return True
        narrow_phase_queue = self._colcvx.process_collisions()
        from ampl import collision_check_cvx_cvx
        MAGIC = 0.02
        for ent_a, comp_a, ent_b, comp_b in narrow_phase_queue:
            verts_a = self._colcvx.get_world_vertices(ent_a, comp_a)
            verts_b = self._colcvx.get_world_vertices(ent_b, comp_b)
            if collision_check_cvx_cvx(verts_a, verts_b) < MAGIC:
                return True
        return False

    def self_collision_distances(
        self, q_arm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-pair signed distance + analytic Jacobian for the active arm.

        Returns:
            dists : (max_contact_rows,) signed distance per near-contact pair.
                    Padded with `+inf` when fewer than `max_contact_rows`
                    pairs are within `d_warn` — keeps the trajopt constraint
                    shape fixed across iterations.
            J_arm : (max_contact_rows, 7) `∂dist_i / ∂q_arm[j]` for the
                    active arm's 7 DOFs. Rows for the padded entries are
                    zero (since `+inf - d_safe ≥ 0` trivially).

        The C++ `fill_contact_jacobians` produces `(nc, DOF_TOTAL=16)` rows;
        we slice out the active arm's columns and pad up to max_rows.
        """
        self._update_state(q_arm, with_jacobian=True)
        # Copy into our pre-allocated F-order buffer. The dst's `order='F'`
        # is preserved by `np.copyto` regardless of src layout — mirrors the
        # `np.copyto(wbc._Js, kin.get_jacobian_actuated_frame())` idiom in
        # `ampl_motion/util_da.py:282`.
        np.copyto(self._J_kin_buf, self._kin.get_jacobian_actuated_frame())

        d_warn = float(self.cfg.collision_d_warn)
        pairs = self._colsph.compute_collisions(d_warn)
        wall_hits = self._colsph.compute_wall_collisions(d_warn)

        # Zero the buffers — fill_contact_jacobians writes only the first
        # `nc` rows.
        self._J_cbf_buf.fill(0.0)
        self._d_cbf_buf.fill(np.inf)

        nc = 0
        if pairs or wall_hits:
            nc_ret = self._colsph.fill_contact_jacobians(
                pairs, wall_hits, self._J_kin_buf,
                start_idx=0,
                J_cbf_out=self._J_cbf_buf,
                distances_out=self._d_cbf_buf,
                max_top_k=int(max(1, self.cfg.collision_top_k)),
                col_shift_cbf=0,
                return_contact_regions=False,
            )
            nc = int(nc_ret) if not isinstance(nc_ret, tuple) else int(nc_ret[0])
            # The buffer's tail beyond `nc` may have been touched if the
            # call wrote a partial result; reset it to safe defaults.
            if nc < self._max_contact_rows:
                self._J_cbf_buf[nc:, :] = 0.0
                self._d_cbf_buf[nc:] = np.inf

        # Slice out only the active arm's 7 cols. Off-arm + torso cols would
        # multiply into frozen q's; for the trajopt decision vector those
        # gradients are dead weight.
        J_arm = self._J_cbf_buf[:, self._arm_lo:self._arm_hi].copy()
        return self._d_cbf_buf.copy(), J_arm

    def min_pair_distance(self, q_arm: np.ndarray) -> float:
        """Closest near-contact signed distance for `q_arm`. `+inf` if no pair within d_warn."""
        dists, _ = self.self_collision_distances(q_arm)
        return float(dists.min())
