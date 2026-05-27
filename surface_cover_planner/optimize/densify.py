"""Densify the trajopt output to a control-rate trajectory.

trajopt runs on a coarse N-step decimated trajectory (e.g. N=40 over 39 s).
Downstream consumers want a uniform control-rate stream — typically 30 Hz
matching surface_cover_planner's `hz`. This module fills that gap by linear-
interpolating the optimized `(q_arm, dt)` in joint space onto a uniform
`target_dt` grid and re-evaluating FK / collision at every densified step.

Why joint-space linear interp (not cartesian slerp + IK):

  * trust-constr's decision variables ARE the per-step `q_arm` values, so
    linear interp between consecutive q's is the canonical "what the
    optimizer committed to" reconstruction.
  * Joint velocities are exactly bounded at the dense rate: if `|q[k+1] −
    q[k]| ≤ v_max · dt[k]` holds at the optimization level and dt at the
    dense level is ≤ dt[k], the linear-interp segment also satisfies the
    cap (linear interpolation preserves L∞ slope).
  * No extra IK calls — densification is `O(N_dense)` FK / collision
    evaluations, vs `O(N_dense)` IK calls for the cartesian-slerp route.
  * Cartesian deviation at densified samples is a derived quantity; if
    cart_dev is unacceptable, the fix is at the trajopt level (raise
    `track_pos` weight or tighten `trust_radius`), not at densification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DenseTrajectory:
    """Output of `densify_joint_space`.

    `q_arm` (N_dense, 7), `dt` (N_dense - 1,) uniform = target_dt,
    `t_global` (N_dense,) = i * target_dt.
    `parent_segment[k]` is the index `i` of the original segment
    (`q_arm_orig[i], q_arm_orig[i+1]`) that dense sample k belongs to —
    useful for stratified diagnostics.
    """

    q_arm: np.ndarray
    dt: np.ndarray
    t_global: np.ndarray
    parent_segment: np.ndarray


def densify_joint_space(
    q_arm: np.ndarray,
    dt: np.ndarray,
    target_dt: float,
) -> DenseTrajectory:
    """Linear-interp `q_arm` onto a uniform `target_dt` grid.

    The dense grid spans `[0, sum(dt)]` at spacing `target_dt`; the last
    sample is forced to be exactly the final original waypoint so endpoint
    pinning is preserved.

    Parameters
    ----------
    q_arm : (N, 7) float64
        Optimized joint trajectory.
    dt : (N - 1,) float64
        Optimized per-segment durations (must be positive).
    target_dt : float
        Densified sample spacing in seconds (typically `1 / hz`).
    """
    q_arm = np.asarray(q_arm, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    if q_arm.ndim != 2 or q_arm.shape[1] != 7:
        raise ValueError(f"q_arm must be (N, 7); got {q_arm.shape}")
    if dt.shape != (q_arm.shape[0] - 1,):
        raise ValueError(
            f"dt must have length N-1 = {q_arm.shape[0] - 1}; got {dt.shape}"
        )
    if target_dt <= 0:
        raise ValueError(f"target_dt must be positive; got {target_dt}")
    if (dt <= 0).any():
        raise ValueError(f"dt must be strictly positive; got min {dt.min()}")

    t_orig = np.concatenate([[0.0], np.cumsum(dt)])
    total_time = float(t_orig[-1])
    # Force-include the terminal sample so the endpoint matches the original.
    n_dense = int(np.ceil(total_time / target_dt)) + 1
    t_dense = np.linspace(0.0, total_time, n_dense)

    # Vectorized linear interp per arm DOF using np.interp.
    q_dense = np.column_stack([np.interp(t_dense, t_orig, q_arm[:, j]) for j in range(7)])

    # parent_segment[k]: largest i with t_orig[i] <= t_dense[k]; cap at N-2.
    parent_segment = np.clip(np.searchsorted(t_orig, t_dense, side="right") - 1, 0, q_arm.shape[0] - 2)

    dt_dense = np.diff(t_dense)
    return DenseTrajectory(
        q_arm=q_dense,
        dt=dt_dense,
        t_global=t_dense,
        parent_segment=parent_segment,
    )


def densify_targets(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    dt: np.ndarray,
    target_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-sample the cartesian targets onto the dense time grid.

    Position is linearly interpolated; quaternion via shortest-arc slerp
    (so the densified target stream is what cart_dev should be measured
    against, segment-by-segment).
    """
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_quat = np.asarray(target_quat, dtype=np.float64)
    if target_pos.shape[1] != 3 or target_quat.shape[1] != 4:
        raise ValueError(
            f"target_pos must be (N, 3) and target_quat (N, 4); got "
            f"{target_pos.shape} and {target_quat.shape}"
        )

    t_orig = np.concatenate([[0.0], np.cumsum(dt)])
    total_time = float(t_orig[-1])
    n_dense = int(np.ceil(total_time / target_dt)) + 1
    t_dense = np.linspace(0.0, total_time, n_dense)

    pos_dense = np.column_stack(
        [np.interp(t_dense, t_orig, target_pos[:, j]) for j in range(3)]
    )

    parent = np.clip(
        np.searchsorted(t_orig, t_dense, side="right") - 1, 0, target_pos.shape[0] - 2
    )
    quat_dense = np.empty((t_dense.size, 4), dtype=np.float64)
    for k in range(t_dense.size):
        i = int(parent[k])
        seg_len = t_orig[i + 1] - t_orig[i]
        alpha = 0.0 if seg_len <= 0 else (t_dense[k] - t_orig[i]) / seg_len
        quat_dense[k] = _slerp(target_quat[i], target_quat[i + 1], float(alpha))
    return pos_dense, quat_dense


def project_dense_to_cart(
    q_dense: np.ndarray,
    target_pos_dense: np.ndarray,
    model,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    max_iters: int = 12,
    tol: float = 1e-4,
    step_limit: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Full 3-axis cart-projection of each dense q via finite-difference DLS.

    Same idea as the z-only projection but solves the 3×7 DLS system per
    sample to drive all three cart components to their targets. Uses a
    finite-difference Jacobian (7 FK calls per iter) because the AMPL
    analytic `fk_pos_and_jac` Jacobian doesn't match `d(tool_pos_world)/dq`
    in this build (a known discrepancy: trajopt tolerates it because the
    warm-start + tight trust radius do the heavy lifting; an unanchored
    projection like this one diverges with the analytic Jacobian).

        err = target_pos − fk_pos(q)               ((3,))
        J   = d(fk_pos)/dq via central FD            ((3, 7))
        Δq  = Jᵀ (J Jᵀ + λ²I)⁻¹ err                 (DLS, 3×3 solve)
        Δq  ← clip ‖Δq‖∞ ≤ step_limit
        q   ← clip(q + Δq, q_lo, q_hi)
    """
    q_dense = np.asarray(q_dense, dtype=np.float64)
    target_pos_dense = np.asarray(target_pos_dense, dtype=np.float64)
    if q_dense.ndim != 2 or q_dense.shape[1] != 7:
        raise ValueError(f"q_dense must be (N, 7); got {q_dense.shape}")
    if target_pos_dense.shape != (q_dense.shape[0], 3):
        raise ValueError(
            f"target_pos_dense must be (N, 3) matching q_dense rows; got "
            f"{target_pos_dense.shape} vs {q_dense.shape}"
        )

    fd_eps = 1e-5

    def fd_dfk_dq(q_local: np.ndarray) -> np.ndarray:
        out = np.empty((3, 7), dtype=np.float64)
        for j in range(7):
            qp = q_local.copy(); qp[j] += fd_eps
            qm = q_local.copy(); qm[j] -= fd_eps
            out[:, j] = (model.fk_position(qp) - model.fk_position(qm)) / (2.0 * fd_eps)
        return out

    eye3 = np.eye(3, dtype=np.float64)
    lam_sq = 1e-6
    n = q_dense.shape[0]
    q_out = q_dense.copy()
    iters_used = np.zeros(n, dtype=np.int32)
    final_err = np.zeros((n, 3), dtype=np.float64)
    for k in range(n):
        q = q_out[k]
        for it in range(max_iters):
            pos = model.fk_position(q)
            err = target_pos_dense[k] - pos                 # (3,)
            if float(np.linalg.norm(err)) < tol:
                break
            J = fd_dfk_dq(q)                                # (3, 7)
            jjt = J @ J.T + lam_sq * eye3
            dq = J.T @ np.linalg.solve(jjt, err)
            dq_inf = float(np.max(np.abs(dq)))
            if dq_inf > step_limit:
                dq *= step_limit / dq_inf
            q = np.clip(q + dq, q_lo, q_hi)
        q_out[k] = q
        iters_used[k] = it + 1
        pos_final = model.fk_position(q)
        final_err[k] = target_pos_dense[k] - pos_final
    return q_out, {"iters": iters_used, "final_err": final_err}


def project_dense_z_only(
    q_dense: np.ndarray,
    target_z_dense: np.ndarray,
    model,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    max_iters: int = 8,
    tol: float = 1e-4,
    step_limit: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """Project each dense q so `fk_pos.z(q) = target_z`, xy left free.

    Joint-space linear interpolation between two on-plane trajopt waypoints
    produces an intermediate q whose FK z is *off* the surface (because
    FK is nonlinear in q). Forcing the full 3-vector `fk_pos(q) = target`
    is *over-specifying*: the trajopt waypoints have ≤30 mm of legitimate
    xy drift under the loose xy cap, and the dense `target_pos` is just
    a linear-interp of the warm targets — not what trajopt actually
    committed to in xy. Projecting xy back to the warm target fights the
    trajopt solution.

    Z-only projection is single-equation underdetermined (1 equation, 7
    unknowns) → unique minimum-norm Δq → small joint moves that leave
    xy approximately where the joint-interp put it (close to trajopt's
    xy, well within the 30 mm cap):

        err_z = target_z − fk_pos.z(q)             (scalar)
        J_z   = J_pos[2, :]                        ((7,))
        Δq    = J_z · err_z / (‖J_z‖² + λ²)         (min-norm solution)
        Δq    ← clip ‖Δq‖∞ ≤ step_limit
        q     ← clip(q + Δq, q_lo, q_hi)

    Parameters
    ----------
    q_dense : (N_dense, 7) float64
        Joint-space-interpolated dense trajectory.
    target_z_dense : (N_dense,) float64
        Per-sample target z (typically constant on a fixed-z surface plan).
    model : RobotModel
        Provides `fk_pos_and_jac`.
    q_lo, q_hi : (7,) float64
        Hardware joint position limits.
    max_iters : int, optional
        Per-sample Newton iters. Default 8 — most samples converge in 1-3.
    tol : float, optional
        Stop when `|err_z| < tol` (meters). Default 0.1 mm.
    step_limit : float, optional
        L∞ cap on per-iter joint step (rad). Default 0.02.

    Returns
    -------
    q_projected : (N_dense, 7) float64
    stats : dict
        `{"iters": (N_dense,) int, "final_dz": (N_dense,) float}`.
    """
    q_dense = np.asarray(q_dense, dtype=np.float64)
    target_z_dense = np.asarray(target_z_dense, dtype=np.float64)
    if q_dense.ndim != 2 or q_dense.shape[1] != 7:
        raise ValueError(f"q_dense must be (N, 7); got {q_dense.shape}")
    if target_z_dense.shape != (q_dense.shape[0],):
        raise ValueError(
            f"target_z_dense must be (N,) matching q_dense rows; got "
            f"{target_z_dense.shape} vs {q_dense.shape}"
        )

    # Use a finite-difference Jacobian for the z row, not AMPL's analytic
    # `fk_pos_and_jac`. The AMPL body Jacobian returned by that path does
    # *not* match `d(tool_pos_world)/dq` (both sign and magnitude differ
    # at random q). Trust-constr / IPOPT in the trajopt loop happen to
    # converge anyway because the warm-start is already feasible and the
    # trust radius pins q near the warm — but the projection here has no
    # such anchor, so using the wrong Jacobian makes the iterates diverge
    # in the *wrong direction*. FD costs 7 FK calls per iter and is
    # numerically reliable.
    fd_eps = 1e-5

    def fd_dfkz_dq(q_local: np.ndarray) -> np.ndarray:
        out = np.empty(7, dtype=np.float64)
        for j in range(7):
            qp = q_local.copy(); qp[j] += fd_eps
            qm = q_local.copy(); qm[j] -= fd_eps
            out[j] = (model.fk_position(qp)[2] - model.fk_position(qm)[2]) / (2.0 * fd_eps)
        return out

    n = q_dense.shape[0]
    q_out = q_dense.copy()
    iters_used = np.zeros(n, dtype=np.int32)
    final_dz = np.zeros(n, dtype=np.float64)
    lam_sq = 1e-6
    for k in range(n):
        q = q_out[k]
        for it in range(max_iters):
            pos = model.fk_position(q)
            err_z = float(target_z_dense[k] - pos[2])
            if abs(err_z) < tol:
                break
            J_z = fd_dfkz_dq(q)                          # (7,)
            denom = float(J_z @ J_z) + lam_sq
            dq = J_z * (err_z / denom)
            dq_inf = float(np.max(np.abs(dq)))
            if dq_inf > step_limit:
                dq *= step_limit / dq_inf
            q = np.clip(q + dq, q_lo, q_hi)
        q_out[k] = q
        iters_used[k] = it + 1
        pos_final = model.fk_position(q)
        final_dz[k] = float(target_z_dense[k] - pos_final[2])
    return q_out, {"iters": iters_used, "final_dz": final_dz}


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc spherical linear interp in (w, x, y, z) form."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_t0 = np.sin(theta_0)
    s0 = np.sin((1.0 - t) * theta_0) / sin_t0
    s1 = np.sin(t * theta_0) / sin_t0
    return s0 * q0 + s1 * q1
