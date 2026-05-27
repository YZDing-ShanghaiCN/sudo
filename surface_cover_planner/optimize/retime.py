"""TOPP-style fixed-path retime helper.

When the warm-start q-trajectory has segments where `|Δq[k,j]| / dt[k] > v_max[j]`,
the trust-constr solver sees the vel constraint as infeasible at iter 0 and
spends most of the optimization budget pulling `dt` outward. Doing that
analytically before the solve gives the SCO a feasible starting point,
turning the inner trust-constr into a pure smoothness/track-cost minimizer.

This is the canonical TOPP simplification when the path is fixed and only
velocity bounds matter — acceleration / jerk shaping happens later inside
trajopt's cost. See `feedback_trajopt_hard_constraints.md`: weakening
`joint_vel_limits` is *not* a legitimate fix; pre-retiming the warm-start is.
"""

from __future__ import annotations

import numpy as np


def topp_retime(
    q_arm: np.ndarray,
    dt_warm: np.ndarray,
    v_max: np.ndarray,
    dt_min: float,
    vel_headroom: float = 0.10,
) -> np.ndarray:
    """Return per-segment dt that makes the warm-start velocity-feasible.

    `dt_required[k] = max(dt_warm[k], max_j |Δq[k,j]| / (v_max[j] · (1 − h)), dt_min)`
    where `h = vel_headroom` reserves slack below the v_max boundary so the
    downstream SCO solver has room to move `q` slightly without immediately
    breaking the vel cap. Default 5% headroom — pure vel-feasibility (h=0)
    pins vel_ratio at 1.0 exactly, which leaves IPOPT no slack and forces
    a "local infeasibility" exit on the first q-perturbation.

    Parameters
    ----------
    q_arm : (N, 7) float64
        Warm-start arm joints.
    dt_warm : (N - 1,) float64
        Warm-start per-segment time deltas (seconds).
    v_max : (7,) float64
        Per-DOF velocity caps (rad/s). Use `RobotModel.v_max`.
    dt_min : float
        Lower bound on each retimed segment (seconds). Same as the value
        passed to the trajopt bounds box.
    vel_headroom : float, optional
        Fractional slack below v_max (0.0 ≤ h < 1.0). Default 0.05 (5%).
    """
    if q_arm.ndim != 2 or q_arm.shape[1] != 7:
        raise ValueError(f"q_arm must be (N, 7); got {q_arm.shape}")
    if dt_warm.shape != (q_arm.shape[0] - 1,):
        raise ValueError(
            f"dt_warm must have length N-1 = {q_arm.shape[0] - 1}; got {dt_warm.shape}"
        )
    if v_max.shape != (7,) or (v_max <= 0).any():
        raise ValueError(f"v_max must be a strictly-positive length-7 vector; got {v_max}")
    if not (0.0 <= vel_headroom < 1.0):
        raise ValueError(f"vel_headroom must be in [0, 1); got {vel_headroom}")

    v_eff = v_max * (1.0 - vel_headroom)
    delta = np.abs(np.diff(q_arm, axis=0))           # (N-1, 7)
    required = (delta / v_eff[None, :]).max(axis=1)  # (N-1,)
    return np.maximum.reduce([dt_warm, required, np.full_like(dt_warm, dt_min)])


def vel_ratios(
    q_arm: np.ndarray, dt: np.ndarray, v_max: np.ndarray
) -> np.ndarray:
    """Return per-pair, per-DOF velocity ratios (|Δq| / (dt * v_max))."""
    delta = np.abs(np.diff(q_arm, axis=0))
    return delta / (dt[:, None] * v_max[None, :] + 1e-12)
