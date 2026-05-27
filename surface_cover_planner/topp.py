"""Analytic time-optimal path parameterization for smoothness.

Classical forward-backward TOPP (Bobrow / Pham) over a discrete joint
path, treating each segment as a unit-arc-length interval and the state
`β = (ds/dt)²` as the per-sample decision. Two passes produce per-segment
`dt` such that, for every arm DOF j and every segment k:

    |Δq[k, j]| / dt[k] ≤ v_max[j] · (1 − vel_headroom)
    |Δq[k+1, j] / dt[k+1] − Δq[k, j] / dt[k]|
        / ((dt[k] + dt[k+1]) / 2)            ≤ a_max[j] · (1 − accel_headroom)

The path itself is unchanged; only the time grid is set. Pair with
`resample_to_uniform_grid` + `slerp_wxyz` to produce a control-rate
output.

References:
  Bobrow, Dubowsky, Gibson (1985), "Time-optimal control of robotic
  manipulators along specified paths."
  Pham (2014), "A general, fast, and robust implementation of the
  time-optimal path parameterization algorithm."
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


def topp_smooth(
    q_arm: np.ndarray,
    v_max: np.ndarray,
    a_max: np.ndarray,
    dt_min: float,
    vel_headroom: float = 0.05,
    accel_headroom: float = 0.05,
) -> np.ndarray:
    """Forward-backward TOPP on a 7-DOF joint path; returns per-segment dt.

    Parameters
    ----------
    q_arm : (M+1, 7) float64
        Discrete joint path. Endpoints pin start / end velocity to zero.
    v_max, a_max : (7,) float64
        Per-DOF velocity (rad/s) and acceleration (rad/s²) caps.
    dt_min : float
        Lower bound on each emitted segment (seconds). Guards the
        degenerate β=0 endpoint segments and any zero-displacement
        segments where the path doesn't actually move in joint space.
    vel_headroom, accel_headroom : float
        Fractional slack below the caps (default 5%). Mirrors
        `optimize/retime.py:topp_retime`'s convention.
    """
    q_arm = np.asarray(q_arm, dtype=np.float64)
    if q_arm.ndim != 2 or q_arm.shape[1] != 7:
        raise ValueError(f"q_arm must be (M+1, 7); got {q_arm.shape}")
    v_max = np.asarray(v_max, dtype=np.float64)
    a_max = np.asarray(a_max, dtype=np.float64)
    if v_max.shape != (7,) or (v_max <= 0).any():
        raise ValueError(f"v_max must be strictly-positive length-7; got {v_max}")
    if a_max.shape != (7,) or (a_max <= 0).any():
        raise ValueError(f"a_max must be strictly-positive length-7; got {a_max}")
    if not (0.0 <= vel_headroom < 1.0):
        raise ValueError(f"vel_headroom must be in [0, 1); got {vel_headroom}")
    if not (0.0 <= accel_headroom < 1.0):
        raise ValueError(f"accel_headroom must be in [0, 1); got {accel_headroom}")
    if dt_min <= 0:
        raise ValueError(f"dt_min must be positive; got {dt_min}")

    M = q_arm.shape[0] - 1
    if M < 1:
        return np.zeros(0, dtype=np.float64)

    v_eff = v_max * (1.0 - vel_headroom)
    a_eff = a_max * (1.0 - accel_headroom)

    delta = np.diff(q_arm, axis=0)               # (M, 7)
    abs_delta = np.abs(delta)

    # Per-segment vel-² cap β ≤ min_j (v_eff[j] / |Δq[k, j]|)². Use a tiny
    # floor on |Δq| to avoid divide-by-zero on stationary segments — those
    # segments impose no vel cap, so we sentinel them out via inf.
    seg_beta_cap = np.full(M, np.inf, dtype=np.float64)
    for k in range(M):
        for j in range(7):
            d = abs_delta[k, j]
            if d > _EPS:
                seg_beta_cap[k] = min(seg_beta_cap[k], (v_eff[j] / d) ** 2)

    # MVC at each sample k: the tighter of the two adjacent segments
    # (endpoints only see one side).
    beta_max = np.full(M + 1, np.inf, dtype=np.float64)
    beta_max[0] = seg_beta_cap[0]
    beta_max[M] = seg_beta_cap[M - 1]
    for k in range(1, M):
        beta_max[k] = min(seg_beta_cap[k - 1], seg_beta_cap[k])

    # Corner-acceleration cap at interior samples. The Bobrow forward-backward
    # bounds in-segment acceleration via `Δβ`, but at a path corner (where
    # q'(s) changes from Δq[k-1] to Δq[k] across sample k) the discrete
    # corner acceleration `(Δq[k]/dt[k] − Δq[k-1]/dt[k-1]) / ((dt[k-1]+dt[k])/2)`
    # saturates at `|Δq[k] − Δq[k-1]| · β` under locally-constant β, so
    #     β ≤ a_eff[j] / |Δq[k, j] − Δq[k-1, j]|
    # per DOF. We apply this cap symmetrically at samples k-1, k, k+1 so the
    # forward-backward can't push β at the corner's neighbours past the
    # locally-constant approximation's validity (without the symmetric
    # propagation, β[k-1] / β[k+1] can be much larger than β[k] and the
    # segment-mean velocity jump still violates the bound). Without these
    # corner caps, piecewise-linear paths with IK branch flips accumulate
    # huge corner-spike accel ratios.
    for k in range(1, M):
        for j in range(7):
            corner = abs(delta[k, j] - delta[k - 1, j])
            if corner > _EPS:
                cap = a_eff[j] / corner
                beta_max[k] = min(beta_max[k], cap)
                beta_max[k - 1] = min(beta_max[k - 1], cap)
                if k + 1 <= M:
                    beta_max[k + 1] = min(beta_max[k + 1], cap)

    # Per-segment Δβ cap: |β[k+1] − β[k]| ≤ min_j (2 · a_eff[j] / |Δq[k, j]|).
    dbeta_cap = np.full(M, np.inf, dtype=np.float64)
    for k in range(M):
        for j in range(7):
            d = abs_delta[k, j]
            if d > _EPS:
                dbeta_cap[k] = min(dbeta_cap[k], 2.0 * a_eff[j] / d)

    # Forward / backward passes — iterated up to a small budget so the
    # corner-accel cap (an analytic approximation under locally-constant β)
    # tightens further on paths where adjacent corners overlap and the
    # locally-constant assumption underestimates the discrete corner accel.
    # Each iteration: run forward + backward, reconstruct dt, compute the
    # discrete segment-pair acceleration, and tighten `beta_max` at samples
    # where the discrete cap is violated. Convergence is monotone (caps only
    # shrink) and typically reaches the saturation ratio in 2–4 iterations.
    max_iters = 32
    for _it in range(max_iters):
        beta_fwd = np.zeros(M + 1, dtype=np.float64)
        for k in range(M):
            nxt = beta_fwd[k] + dbeta_cap[k]
            beta_fwd[k + 1] = min(beta_max[k + 1], nxt)
        beta_bwd = np.zeros(M + 1, dtype=np.float64)
        for k in range(M - 1, -1, -1):
            prv = beta_bwd[k + 1] + dbeta_cap[k]
            beta_bwd[k] = min(beta_max[k], prv)
        beta = np.minimum(beta_fwd, beta_bwd)

        sqrt_beta = np.sqrt(np.clip(beta, 0.0, None))
        denom = sqrt_beta[:-1] + sqrt_beta[1:]
        dt = np.empty(M, dtype=np.float64)
        for k in range(M):
            dt[k] = dt_min if denom[k] < _EPS else max(2.0 / denom[k], dt_min)

        if M < 2:
            return dt
        # Discrete corner-accel diagnostic: ratio of segment-mean accel to
        # a_eff per DOF. The corner is at sample k+1 (between segments k and
        # k+1).
        velocities = np.diff(q_arm, axis=0) / dt[:, None]
        dvel = np.diff(velocities, axis=0)
        seg_pair_dt = (dt[:-1] + dt[1:]) / 2.0
        ratios = np.abs(dvel) / (seg_pair_dt[:, None] * a_eff[None, :])
        max_ratio = float(ratios.max()) if ratios.size else 0.0
        if max_ratio <= 1.0 + 1e-6:
            return dt
        # Tighten beta_max at the offending corner sample(s) proportionally.
        # `ratio² > 1` → β was a factor ratio² too large at the corner — shrink
        # by 1/ratio². Apply to k, k-1 (segment-k mean velocity uses β[k] and
        # β[k+1]) so adjacent sqrt(β) values both pull down.
        per_sample_ratio = ratios.max(axis=1)  # (M-1,)
        any_change = False
        for kk in range(M - 1):
            r = float(per_sample_ratio[kk])
            if r <= 1.0 + 1e-6:
                continue
            corner_idx = kk + 1
            shrink = 1.0 / (r * r)
            for s in (corner_idx - 1, corner_idx, corner_idx + 1):
                if 0 <= s <= M:
                    new_cap = beta_max[s] * shrink
                    if new_cap < beta_max[s] - _EPS:
                        beta_max[s] = new_cap
                        any_change = True
        if not any_change:
            return dt
    return dt


def envelope_smooth(
    q_arm: np.ndarray,
    v_max: np.ndarray,
    a_max: np.ndarray,
) -> np.ndarray:
    """Per-segment envelope dt: ``max(t_vel, t_accel)``. No smoothing, no floor.

    For each segment k:
        t_vel[k]   = max_j |Δq[k, j]| / v_max[j]
        t_accel[k] = sqrt(max_j |Δq[k, j]| / a_max[j])
        dt[k]      = max(t_vel[k], t_accel[k])

    No forward-backward propagation: adjacent segments may end up with very
    different speeds, so the discrete corner acceleration between them is
    only *locally* bounded (each segment respects a_max for its own Δq, but
    a velocity jump at the corner can still exceed `a_max · dt_avg`). Use
    `topp.topp_smooth` instead when the discrete corner-accel cap must hold.

    Zero-displacement segments (every |Δq[k, j]| < `_EPS`) get `dt[k] = _EPS`
    so `resample_to_uniform_grid` doesn't divide by zero; they contribute
    essentially nothing to total time and disappear after resampling.

    Mirrors the per-edge formula in ``optimize/local_dp.py`` (`cost_mode =
    'fastest'`) so the sparse-DP cost and the dense-grid timing agree.
    """
    q_arm = np.asarray(q_arm, dtype=np.float64)
    if q_arm.ndim != 2 or q_arm.shape[1] != 7:
        raise ValueError(f"q_arm must be (M+1, 7); got {q_arm.shape}")
    v_max = np.asarray(v_max, dtype=np.float64)
    a_max = np.asarray(a_max, dtype=np.float64)
    if v_max.shape != (7,) or (v_max <= 0).any():
        raise ValueError(f"v_max must be strictly-positive length-7; got {v_max}")
    if a_max.shape != (7,) or (a_max <= 0).any():
        raise ValueError(f"a_max must be strictly-positive length-7; got {a_max}")

    M = q_arm.shape[0] - 1
    if M < 1:
        return np.zeros(0, dtype=np.float64)

    abs_delta = np.abs(np.diff(q_arm, axis=0))                   # (M, 7)
    t_vel = (abs_delta / v_max[None, :]).max(axis=1)             # (M,)
    t_accel = np.sqrt((abs_delta / a_max[None, :]).max(axis=1))  # (M,)
    dt = np.maximum(t_vel, t_accel)
    return np.where(dt < _EPS, _EPS, dt)


def resample_to_uniform_grid(
    dt: np.ndarray,
    target_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (t_dense, parent_segment, alpha) tuple for resampling.

    `parent_segment[k]` is the original segment index `i ∈ [0, M-1]` such
    that `t_orig[i] ≤ t_dense[k] ≤ t_orig[i+1]`; `alpha[k] ∈ [0, 1]` is the
    linear-interp weight inside that segment:
        `t_dense[k] = (1 − alpha[k]) · t_orig[parent] + alpha[k] · t_orig[parent + 1]`.

    The dense grid spans `[0, sum(dt)]` at spacing `target_dt`; the final
    sample is forced onto `sum(dt)` exactly so endpoint pinning is
    preserved (matches `optimize/densify.py:densify_joint_space`).
    """
    dt = np.asarray(dt, dtype=np.float64)
    if dt.ndim != 1:
        raise ValueError(f"dt must be 1-D; got shape {dt.shape}")
    if (dt <= 0).any():
        raise ValueError(f"dt must be strictly positive; got min {dt.min()}")
    if target_dt <= 0:
        raise ValueError(f"target_dt must be positive; got {target_dt}")

    t_orig = np.concatenate([[0.0], np.cumsum(dt)])
    total_time = float(t_orig[-1])
    n_dense = int(np.ceil(total_time / target_dt)) + 1
    t_dense = np.linspace(0.0, total_time, n_dense)

    M = dt.shape[0]
    parent = np.clip(
        np.searchsorted(t_orig, t_dense, side="right") - 1, 0, M - 1
    ).astype(np.int32)
    seg_durations = dt[parent]
    alpha = (t_dense - t_orig[parent]) / np.maximum(seg_durations, _EPS)
    alpha = np.clip(alpha, 0.0, 1.0)
    return t_dense, parent, alpha


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc spherical linear interp in (w, x, y, z) form.

    Pure numpy; mirrors the helper in `optimize/densify.py` so this module
    has no dependency on the optimize subpackage.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + _EPS)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_t0 = np.sin(theta_0)
    s0 = np.sin((1.0 - t) * theta_0) / sin_t0
    s1 = np.sin(t * theta_0) / sin_t0
    return s0 * q0 + s1 * q1
