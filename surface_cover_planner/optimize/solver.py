"""Sequential-convex-optimization-style post-processor for surface_cover_planner.

This module implements a python-only TrajOpt-spirited solver: minimize a
weighted sum of (total time + joint accel-as-jerk-proxy + cartesian tracking
+ collision penalty) over a (q, dt) trajectory pinned to a known feasible
warm-start. The actual solver is `scipy.optimize.minimize(method="trust-constr")`,
not Berkeley TrajOpt's SQP — the 2014 C++ TrajOpt won't build in the
SudoDeploy docker (OpenRAVE / Boost.Python deps).

Decision vector layout (`_pack` / `_unpack`):

    x = [ q[0,0..6], q[1,0..6], …, q[N-1,0..6], dt[0], …, dt[N-2] ]
    len(x) = 7*N + (N - 1)

Cost components (all minimized):

    cost_time   = w_time   * sum(dt)
    cost_jerk   = w_jerk   * sum_k ||(q[k+1] - 2*q[k] + q[k-1]) / dt_k^2||^2
    cost_track  = w_track_pos * sum_k ||fk_pos(q[k]) - target_pos[k]||^2
                + w_track_quat * sum_k (1 - <fk_quat(q[k]), target_quat[k]>^2)
    cost_col    = w_col * count(self_collides(q[k]))

Hard constraints:

    box        : q_lo <= q[k] <= q_hi  (per-DOF; intersected with trust region)
    trust      : |q[k] - q_warm[k]|_inf <= trust_radius
    endpoints  : q[0] == q_warm[0], q[N-1] == q_warm[N-1]  (when pin_endpoints)
    dt box     : dt_min <= dt[k] <= dt_max
    vel        : |q[k+1] - q[k]| <= v_max * dt[k]   (per-DOF)
    cart box   : |fk_pos.x(q[k]) - target_pos.x[k]| <= max_xy_deviation,
                 |fk_pos.y(q[k]) - target_pos.y[k]| <= max_xy_deviation,
                 |fk_pos.z(q[k]) - target_pos.z[k]| <= max_z_deviation
                 (per-axis box, linear in fk_pos; analytic Jacobian = J_pos row)

All whyevaluations against AMPL (FK, self-collision) are routed through a
`RobotModel`. Finite-diff jacobians are computed by trust-constr; cached FK
across consecutive evaluator calls would help, but the simple uncached path
is what scipy already does internally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    minimize,
)
from scipy.sparse import csr_matrix

from .robot_model import RobotModel


@dataclass
class OptWeights:
    time: float = 1.0
    jerk: float = 10.0
    track_pos: float = 100.0
    track_quat: float = 0.1
    # Soft-cost collision weight is retained for back-compat with old YAML
    # configs but is *not* on the active cost path — collision is now a
    # `NonlinearConstraint` (`dist_pair >= d_safe`) with an analytic Jacobian
    # via ampl_motion's sphere-BVH SDF. The field is logged in solve_stats
    # so historical comparisons stay readable.
    collision: float = 0.0


@dataclass
class OptBounds:
    trust_radius: float = 0.05      # rad, |q - q_warm|_inf
    dt_min: float = 0.01            # s
    dt_max: float = 1.0             # s
    # Per-axis cartesian-deviation caps (m). Replaces the prior L2-norm cap
    # `max_cartesian_deviation`. Surface-cover plans live on a fixed-z plane,
    # so the z-axis cap is the safety-critical knob — the prior L2 form let
    # the solver silently pool all the allowed error into z. Both caps are
    # auto-relaxed independently if the warm-start exceeds them. xy is
    # loose by default (30 mm — in-plane drift along the cover pattern is
    # an approximate spec); z is strict (5 mm — out-of-plane is the
    # surface-following safety property).
    max_xy_deviation: float = 0.030  # |Δx|, |Δy| caps (in-plane drift)
    max_z_deviation: float = 0.005   # |Δz| cap (out-of-plane / surface drift)
    pin_endpoints: bool = True


@dataclass
class SolverOpts:
    # IPOPT is the default — its restoration phase handles the
    # "warmstart-feasible / cost-pulls-infeasible" pathology that stalls
    # trust-constr on the surface-cover problem. trust-constr remains
    # selectable as a fallback for containers without cyipopt; the
    # `solve()` body also drops to trust-constr automatically when
    # `method='ipopt'` is requested but cyipopt isn't installed.
    method: str = "ipopt"
    max_iters: int = 100
    xtol: float = 1e-6
    gtol: float = 1e-6
    verbose: int = 1


@dataclass
class SolverResult:
    q_arm: np.ndarray             # (N, 7)
    dt: np.ndarray                # (N-1,)
    achieved_pos: np.ndarray      # (N, 3)
    achieved_quat_wxyz: np.ndarray  # (N, 4)
    cart_dev: np.ndarray          # (N,) L2 norm (reporting / back-compat)
    cart_dev_axes: np.ndarray     # (N, 3) signed per-axis deviation (p - t)
    self_coll: np.ndarray         # (N,) bool
    q_full: np.ndarray            # (N, 16)
    solve_stats: dict
    # Optional per-sample targets. The IPOPT / trust-constr path inherits
    # targets from the warmstart (one-to-one with the optimized q_arm
    # rows), so these are left None. The local_dp path re-densifies the
    # sparse skeleton on a fresh grid whose length / cadence can differ
    # from the warmstart's, so it must hand the new targets back to the
    # JSON writer alongside `q_arm` / `dt`.
    target_pos: np.ndarray | None = None        # (N, 3) or None
    target_quat_wxyz: np.ndarray | None = None  # (N, 4) or None


def _pack(q: np.ndarray, dt: np.ndarray) -> np.ndarray:
    return np.concatenate([q.reshape(-1), dt])


def _unpack(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    q = x[: 7 * n].reshape(n, 7)
    dt = x[7 * n :]
    return q, dt


def _max_jerk_proxy(q: np.ndarray, dt: np.ndarray) -> float:
    """L_inf finite-diff joint accel — surrogate for jerk-quality reporting."""
    if q.shape[0] < 3:
        return 0.0
    accel = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt[1:, None] * dt[:-1, None])
    return float(np.max(np.abs(accel)))


def _summarize(
    q: np.ndarray,
    dt: np.ndarray,
    cart_dev: np.ndarray,
    cart_dev_axes: np.ndarray,
    self_coll: np.ndarray,
) -> dict:
    abs_axes = np.abs(cart_dev_axes) if cart_dev_axes.size else cart_dev_axes
    return {
        "max_jerk_proxy": _max_jerk_proxy(q, dt),
        "total_time": float(dt.sum()),
        "max_cart_dev": float(cart_dev.max()) if cart_dev.size else 0.0,
        "max_dx": float(abs_axes[:, 0].max()) if abs_axes.size else 0.0,
        "max_dy": float(abs_axes[:, 1].max()) if abs_axes.size else 0.0,
        "max_dz": float(abs_axes[:, 2].max()) if abs_axes.size else 0.0,
        "n_collisions": int(self_coll.sum()),
    }


def _eval_fk_and_collision(
    model: RobotModel, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run FK + self-collision for every row of q (N, 7).

    Returns (achieved_pos (N,3), achieved_quat (N,4), self_coll (N,), q_full (N,16)).
    """
    n = q.shape[0]
    pos = np.empty((n, 3), dtype=np.float64)
    quat = np.empty((n, 4), dtype=np.float64)
    coll = np.empty(n, dtype=bool)
    q_full = np.empty((n, 16), dtype=np.float64)
    for i in range(n):
        q_full[i] = model.assemble_q_full(q[i])
        p, qw = model.fk_pose(q[i])
        pos[i] = p
        quat[i] = qw
        coll[i] = model.self_collides(q[i])
    return pos, quat, coll, q_full


def _eval_fk_jac_and_collision(
    model: RobotModel, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fused single-pass FK + J_pos + self-collision.

    Returns (pos (N,3), quat (N,4), J_pos (N,3,7), self_coll (N,), q_full (N,16)).
    Each row triggers one `update_kin(..., update_jacobian=True)` instead of
    the 2-3 separate AMPL calls the un-fused path needs.
    """
    n = q.shape[0]
    pos = np.empty((n, 3), dtype=np.float64)
    quat = np.empty((n, 4), dtype=np.float64)
    J_pos = np.empty((n, 3, 7), dtype=np.float64)
    coll = np.empty(n, dtype=bool)
    q_full = np.empty((n, 16), dtype=np.float64)
    for i in range(n):
        p, qw, J, c = model.fk_pos_jac_and_collision(q[i])
        pos[i] = p
        quat[i] = qw
        J_pos[i] = J
        coll[i] = c
        q_full[i] = model.assemble_q_full(q[i])  # cheap: scratch + copy
    return pos, quat, J_pos, coll, q_full


def _quat_align_inplace(quat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Flip rows of `quat` whose dot with the matching row of `ref` is negative.

    Quaternion double-cover means q and -q encode the same rotation; aligning
    sign keeps the (1 - <q, ref>^2) tracking cost smooth across the whole
    trajectory rather than oscillating sign by sign.
    """
    dots = np.einsum("ij,ij->i", quat, ref)
    flip = dots < 0
    quat[flip] *= -1.0
    return quat


def _cost_components(
    model: RobotModel,
    q: np.ndarray,
    dt: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    weights: OptWeights,
) -> tuple[float, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate every smooth component plus the collision penalty.

    Returns (total_cost, components_dict, fk_pos, fk_quat, self_coll). The
    extra returns are caller-side bookkeeping (cart_dev, etc.).
    """
    fk_pos, fk_quat, coll, _ = _eval_fk_and_collision(model, q)
    fk_quat = _quat_align_inplace(fk_quat, target_quat)

    pos_err = fk_pos - target_pos
    track_pos = float(np.einsum("ij,ij->", pos_err, pos_err))

    quat_dot = np.einsum("ij,ij->i", fk_quat, target_quat)
    track_quat = float(np.sum(1.0 - quat_dot * quat_dot))

    if q.shape[0] >= 3:
        accel = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt[1:, None] * dt[:-1, None])
        jerk = float(np.einsum("ij,ij->", accel, accel))
    else:
        jerk = 0.0

    time_cost = float(dt.sum())

    components = {
        "time": weights.time * time_cost,
        "jerk": weights.jerk * jerk,
        "track_pos": weights.track_pos * track_pos,
        "track_quat": weights.track_quat * track_quat,
        # Collision is enforced as a hard constraint now (see
        # `_collision_constraint`). The "collision" component is kept in the
        # dict at 0.0 so the cost-component PNG keeps its legend stable.
        "collision": 0.0,
    }
    total = sum(components.values())
    return total, components, fk_pos, fk_quat, coll


def _build_box_bounds(
    q_warm: np.ndarray,
    dt_warm: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    trust_radius: float,
    dt_min: float,
    dt_max: float,
    pin_endpoints: bool,
) -> Bounds:
    n, _ = q_warm.shape
    lo = np.maximum(np.broadcast_to(q_lo, (n, 7)), q_warm - trust_radius).copy()
    hi = np.minimum(np.broadcast_to(q_hi, (n, 7)), q_warm + trust_radius).copy()
    if pin_endpoints:
        # Tight box at the warm-start endpoints; trust-constr treats lo == hi
        # as an equality.
        lo[0] = q_warm[0]
        hi[0] = q_warm[0]
        lo[-1] = q_warm[-1]
        hi[-1] = q_warm[-1]
    # dt lower bound = dt_min (free). The vel LinearConstraint
    # `|Δq[k]| ≤ v_max·dt[k]` already enforces the per-segment velocity floor
    # at every (k, j); leaving the warmstart's dt as the lower bound would
    # artificially block the optimizer from reclaiming time on segments where
    # the planner's cartesian-level dt was looser than the joint-space vel
    # floor. The retimed warm-start is still used as the *initial iterate*
    # (`x0` in `solve()`), so iter-0 is feasible (vel ratio ≈ 1 − headroom).
    dt_lo = np.full(n - 1, dt_min)
    # If the retimed warmstart already needs dt > dt_max for vel feasibility,
    # auto-bump dt_max so iter-0 stays inside the box. Same idea as the
    # cart-cap auto-relax: the warm is the binding spec when it disagrees
    # with the configured cap.
    dt_hi_val = dt_max
    if dt_warm.size and float(dt_warm.max()) > dt_max:
        dt_hi_val = float(dt_warm.max()) * 1.05
        print(
            f"[optimize] WARNING: retimed warmstart needs dt up to "
            f"{float(dt_warm.max()):.3f}s > dt_max={dt_max:.3f}s; "
            f"auto-bumping dt_max -> {dt_hi_val:.3f}s for box feasibility"
        )
    dt_hi = np.full(n - 1, dt_hi_val)
    lo_flat = np.concatenate([lo.reshape(-1), dt_lo])
    hi_flat = np.concatenate([hi.reshape(-1), dt_hi])
    return Bounds(lo_flat, hi_flat)


def _vel_linear_constraint(model: RobotModel, n: int) -> LinearConstraint:
    """Joint-velocity cap as a sparse LinearConstraint.

    Each (k, j) pair contributes two rows to A·x ≤ 0:
        upper:  +q[k+1,j] − q[k,j] − v_max[j]·dt[k] ≤ 0
        lower:  −q[k+1,j] + q[k,j] − v_max[j]·dt[k] ≤ 0

    With x = [q.flatten() (7N), dt (N−1)], each row has 3 nonzeros — two on
    the q variables and one on dt. trust-constr handles linear constraints
    analytically, so this avoids ~14·(N−1) finite-diff probes per outer iter
    of cost-fn evaluations on a constraint that's purely linear in x.
    """
    v_max = model.v_max  # (7,)
    n_x = 7 * n + (n - 1)
    n_pairs = n - 1

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def q_idx(k: int, j: int) -> int:
        return 7 * k + j

    def dt_idx(k: int) -> int:
        return 7 * n + k

    row = 0
    # Upper rows (q[k+1] − q[k] − v·dt ≤ 0).
    for k in range(n_pairs):
        for j in range(7):
            rows.append(row); cols.append(q_idx(k + 1, j)); vals.append(+1.0)
            rows.append(row); cols.append(q_idx(k,     j)); vals.append(-1.0)
            rows.append(row); cols.append(dt_idx(k));       vals.append(-float(v_max[j]))
            row += 1
    # Lower rows (−q[k+1] + q[k] − v·dt ≤ 0).
    for k in range(n_pairs):
        for j in range(7):
            rows.append(row); cols.append(q_idx(k + 1, j)); vals.append(-1.0)
            rows.append(row); cols.append(q_idx(k,     j)); vals.append(+1.0)
            rows.append(row); cols.append(dt_idx(k));       vals.append(-float(v_max[j]))
            row += 1
    A = csr_matrix((vals, (rows, cols)), shape=(2 * 7 * n_pairs, n_x))
    ub = np.zeros(2 * 7 * n_pairs)
    lb = np.full_like(ub, -np.inf)
    return LinearConstraint(A, lb, ub)


def _axis_cart_constraint(
    model: RobotModel,
    n: int,
    target_pos: np.ndarray,
    cap_x: float,
    cap_y: float,
    cap_z: float,
) -> NonlinearConstraint:
    """Per-step per-axis box on cartesian deviation.

    Emits `3N` rows of signed deviation `(fk_pos(q[k]) - target_pos[k])[a]`
    for `a in {x, y, z}`, bounded `[-cap_a, +cap_a]`. The signed deviation
    is *linear in fk_pos*, so its analytic Jacobian is simply the `a`-th
    row of `J_pos[k]` — no chain rule, no `1/||p−t||` singularity at the
    on-target locus, which was the failure mode of the prior L2 cap.

    The Jacobian is sparse: each row depends on only its own `q[k]` block,
    so we emit a CSR with `3N · 7` nonzeros total.
    """
    n_x = 7 * n + (n - 1)
    caps = np.array([cap_x, cap_y, cap_z], dtype=np.float64)
    lb = np.tile(-caps, n)
    ub = np.tile(+caps, n)

    def f(x: np.ndarray) -> np.ndarray:
        q, _ = _unpack(x, n)
        out = np.empty(3 * n, dtype=np.float64)
        for i in range(n):
            p = model.fk_position(q[i])
            out[3 * i : 3 * i + 3] = p - target_pos[i]
        return out

    # Row indices: for each step i and axis a in {0,1,2}, the constraint row
    # is `3*i + a` and depends on q[i] (cols 7*i .. 7*i+7).
    rows = np.repeat(np.arange(3 * n), 7)
    # cols: for each of the 3N rows, the 7 cols are 7*i .. 7*i+6 where i = row//3.
    cols = (np.repeat(np.arange(n) * 7, 3)[:, None] + np.arange(7)[None, :]).reshape(-1)

    def jac(x: np.ndarray):
        q, _ = _unpack(x, n)
        vals = np.empty(3 * n * 7, dtype=np.float64)
        for i in range(n):
            _, _, J_pos = model.fk_pos_and_jac(q[i])  # (3, 7)
            # rows for this step are 3*i, 3*i+1, 3*i+2 — write J_pos rows.
            vals[7 * (3 * i) : 7 * (3 * i + 3)] = J_pos.reshape(-1)
        return csr_matrix((vals, (rows, cols)), shape=(3 * n, n_x))

    return NonlinearConstraint(f, lb, ub, jac=jac)


def _collision_constraint(
    model: RobotModel,
    n: int,
    d_safe: float,
) -> NonlinearConstraint:
    """Per-pair signed-distance hard constraint `dist_pair >= d_safe`.

    For each waypoint `k`, `model.self_collision_distances(q[k])` returns
    `M` rows (`M = top_k * num_macro_regions`, padded with `+inf` for the
    no-contact case). The constraint vector is `[dists_0, …, dists_{N-1}]`,
    length `N*M`. Bound `lb = d_safe`, `ub = +inf`.

    The Jacobian is **block-diagonal** in waypoint index: row `k*M + m` only
    depends on `q[k]`. Sparse CSR with `N*M*7` nonzeros.

    Caching: scipy evaluates the constraint function once per linesearch
    step but the Jacobian only once per outer iter. Both internally call
    `update_kin(q[k], update_jacobian=True)` per waypoint, so we share work
    via an `x`-keyed cache to halve FK calls on adjacent f/jac pairs.
    """
    M = model.max_contact_rows
    n_x = 7 * n + (n - 1)
    n_rows = n * M

    # Sparsity pattern — same every iteration since rows always have 7
    # nonzeros at the same `q[k]` cols. Row r ∈ [k*M, (k+1)*M) depends on
    # q[k, 0..7).
    rows_idx = np.repeat(np.arange(n_rows), 7)
    # cols_idx[r] = [7k, 7k+1, …, 7k+6] where k = r // M.
    cols_idx = (
        np.repeat(np.arange(n) * 7, M)[:, None] + np.arange(7)[None, :]
    )  # (n*M, 7)
    cols_flat = cols_idx.reshape(-1)

    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def _eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = x.tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        q, _ = _unpack(x, n)
        dists_all = np.empty(n_rows, dtype=np.float64)
        J_all = np.empty((n_rows, 7), dtype=np.float64)
        for i in range(n):
            d_i, J_i = model.self_collision_distances(q[i])
            dists_all[i * M : (i + 1) * M] = d_i
            J_all[i * M : (i + 1) * M, :] = J_i
        # Trim the cache so it can't grow unbounded under aggressive
        # linesearch.
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
        cache[key] = (dists_all, J_all)
        return dists_all, J_all

    def f(x: np.ndarray) -> np.ndarray:
        dists, _ = _eval(x)
        # Replace +inf padding with a large finite number — scipy.trust-constr
        # tolerates inf in constraint values, but IPOPT can be fussy. Anything
        # well above d_safe + slack is fine.
        out = dists.copy()
        out[np.isinf(out)] = 1.0e6
        return out

    def jac(x: np.ndarray):
        _, J_arm = _eval(x)
        return csr_matrix(
            (J_arm.reshape(-1), (rows_idx, cols_flat)),
            shape=(n_rows, n_x),
        )

    lb = np.full(n_rows, d_safe, dtype=np.float64)
    ub = np.full(n_rows, np.inf, dtype=np.float64)
    return NonlinearConstraint(f, lb, ub, jac=jac)


def solve(
    *,
    q_warm: np.ndarray,            # (N, 7)
    dt_warm: np.ndarray,            # (N-1,)
    target_pos: np.ndarray,        # (N, 3)
    target_quat_wxyz: np.ndarray,  # (N, 4)
    model: RobotModel,
    weights: OptWeights = OptWeights(),
    bounds: OptBounds = OptBounds(),
    solver_opts: SolverOpts = SolverOpts(),
) -> SolverResult:
    n = q_warm.shape[0]
    if n < 3:
        raise ValueError(f"trajopt needs N >= 3 q-trajectory points; got {n}")
    if dt_warm.shape[0] != n - 1:
        raise ValueError(
            f"dt_warm must have length N-1 = {n - 1}; got {dt_warm.shape[0]}"
        )

    # Auto-relax the per-axis caps if the warm-start already violates them
    # (better to let the solver clean it up than to refuse to start). Each
    # axis is bumped independently so a warm-start with, say, 1 cm of z
    # drift doesn't silently loosen the xy caps too.
    pos_warm, quat_warm, coll_warm, _ = _eval_fk_and_collision(model, q_warm)
    quat_warm = _quat_align_inplace(quat_warm, target_quat_wxyz)
    cart_axes_warm = pos_warm - target_pos                       # (N, 3) signed
    cart_warm = np.linalg.norm(cart_axes_warm, axis=1)            # (N,) L2 (reporting)
    cap_x = bounds.max_xy_deviation
    cap_y = bounds.max_xy_deviation
    cap_z = bounds.max_z_deviation
    abs_axes_warm = np.abs(cart_axes_warm)
    for label, idx, cap_ref in (("x", 0, "max_xy_deviation"),
                                ("y", 1, "max_xy_deviation"),
                                ("z", 2, "max_z_deviation")):
        warm_max = float(abs_axes_warm[:, idx].max()) if abs_axes_warm.size else 0.0
        cur_cap = (cap_x, cap_y, cap_z)[idx]
        if warm_max > cur_cap:
            relaxed = warm_max * 1.05
            print(
                f"[trajopt] warm-start max |Δ{label}| = {warm_max:.4f} m > "
                f"cap {cur_cap:.4f} m ({cap_ref}); relaxing {label}-cap to "
                f"{relaxed:.4f} m for the solve"
            )
            if idx == 0: cap_x = relaxed
            elif idx == 1: cap_y = relaxed
            else: cap_z = relaxed

    # Run with the configured `d_safe` as-is, even when the warm-start's
    # sphere-SDF already undercuts it. Collision is a hard constraint; the
    # right response to an infeasible warm is to let the solver drive `q`
    # *away* from the colliding links, not to silently redefine "safe" as
    # "wherever the warm happens to be". IPOPT's restoration phase handles
    # warm-start-infeasible starts cleanly (the documented motivation for
    # the IPOPT-vs-trust-constr default). The post-solve gate
    # `min_pair_distance_after ≥ d_safe_used` is the regression alarm.
    d_safe_cfg = model.collision_d_safe
    d_safe_used = d_safe_cfg
    min_d_warm = float("inf")
    for q_k in q_warm:
        md_k = model.min_pair_distance(q_k)
        if md_k < min_d_warm:
            min_d_warm = md_k
    if not np.isinf(min_d_warm) and min_d_warm < d_safe_cfg:
        print(
            f"[trajopt] warm-start min sphere-SDF = {min_d_warm:.4f} m < "
            f"d_safe = {d_safe_cfg:.4f} m (binary mesh check at the warm "
            f"reports n_collisions={int(coll_warm.sum())}); keeping d_safe "
            f"as-is and asking the solver to push q away from collision. "
            f"If the solver stalls, tighten Phase 1/2/3 collision avoidance "
            f"or lower `collision.d_safe`."
        )

    warmstart_summary = _summarize(q_warm, dt_warm, cart_warm, cart_axes_warm, coll_warm)

    iter_history: list[dict] = []
    eval_count = {"n": 0}

    def cost_fn(x: np.ndarray) -> float:
        q, dt = _unpack(x, n)
        total, _, _, _, _ = _cost_components(
            model, q, dt, target_pos, target_quat_wxyz, weights
        )
        eval_count["n"] += 1
        return total

    def cost_grad_fn(x: np.ndarray) -> np.ndarray:
        """Analytic gradient for time + jerk + track_pos; track_quat via
        q-only central diff; collision contributes 0 (non-smooth).

        Per outer iter this drops cost-gradient FK calls from O(N²) (scipy's
        2-point over the whole decision vector) to ~8N (one fused FK+J pass
        over each row, plus 2·7·N for the track_quat finite-diff).
        """
        q, dt = _unpack(x, n)
        fk_pos, fk_quat, J_pos, _, _ = _eval_fk_jac_and_collision(model, q)
        fk_quat = _quat_align_inplace(fk_quat, target_quat_wxyz)

        n_x = 7 * n + (n - 1)
        grad = np.zeros(n_x, dtype=np.float64)

        # ---- time: d(w_time · sum(dt)) / d(dt[k]) = w_time ----
        grad[7 * n :] += weights.time

        # ---- track_pos: 2 · w · (p - t)^T · J_pos[k] for each k ----
        if weights.track_pos != 0:
            err = fk_pos - target_pos                       # (N, 3)
            # contribution to grad over q[k, j] = 2 · w · err_k · J_pos[k][:,j]
            tp_grad_q = 2.0 * weights.track_pos * np.einsum("ki,kij->kj", err, J_pos)
            grad[: 7 * n] += tp_grad_q.reshape(-1)

        # ---- jerk: d/dq[m, j] sums contributions from k = m-1, m, m+1 ----
        # accel[k] = (q[k+1] - 2 q[k] + q[k-1]) / (dt[k] · dt[k-1]); k=1..N-2.
        # cost_jerk = w_jerk · sum_{k,j} accel[k,j]^2.
        if weights.jerk != 0 and n >= 3:
            denom = dt[1:] * dt[:-1]                         # (N-2,)
            num = q[2:] - 2.0 * q[1:-1] + q[:-2]             # (N-2, 7)
            accel = num / denom[:, None]                     # (N-2, 7)
            two_w_a = 2.0 * weights.jerk * accel             # (N-2, 7)
            # ∂(accel[k]^2)/∂q[m, j]: pattern in num:
            #   k = m-1 → +1, k = m → -2, k = m+1 → +1
            inv_denom = (1.0 / denom)[:, None]               # (N-2, 1)
            # term per k = two_w_a[k] * (1 / denom[k]) * sign — accumulate.
            jerk_grad_q = np.zeros((n, 7), dtype=np.float64)
            jerk_grad_q[:-2] += two_w_a * inv_denom            # k = m-1, sign +1
            jerk_grad_q[1:-1] += two_w_a * inv_denom * (-2.0)  # k = m,   sign -2
            jerk_grad_q[2:]   += two_w_a * inv_denom            # k = m+1, sign +1
            grad[: 7 * n] += jerk_grad_q.reshape(-1)
            # ∂/∂dt[m]: accel[k] depends on dt via denom[k] = dt[k]·dt[k-1].
            # accel[k] = num[k] / (dt[k] dt[k-1]) → ∂/∂dt[k] = -accel/dt[k];
            # similarly ∂/∂dt[k-1] = -accel/dt[k-1]. Squared: 2·accel·∂accel.
            grad_dt = np.zeros(n - 1, dtype=np.float64)
            # contribution from k for dt[k] (right time):  -2·w·accel² / dt[k] summed over j
            # But we already have two_w_a = 2·w·accel; we need 2·w·accel² = accel · two_w_a.
            squared = float  # type: ignore  # placeholder for clarity
            del squared
            for jrk_k in range(num.shape[0]):
                # k_global for dt is jrk_k (left) and jrk_k+1 (right) of "dt indices"
                # since accel[k=1..N-2] uses dt[k] and dt[k-1] in our 0-indexed code:
                #   accel index `jrk_k` (0-indexed in `num`) corresponds to global k = jrk_k + 1
                #   which uses dt[jrk_k + 1] and dt[jrk_k].
                contrib = float(np.dot(accel[jrk_k], two_w_a[jrk_k]))  # 2·w·sum_j accel²
                grad_dt[jrk_k]     -= contrib / dt[jrk_k]      # ∂/∂dt[k-1]
                grad_dt[jrk_k + 1] -= contrib / dt[jrk_k + 1]  # ∂/∂dt[k]
            grad[7 * n :] += grad_dt

        # ---- track_quat: q-only central finite-diff ----
        if weights.track_quat != 0:
            eps = 1e-6
            quat_dot = np.einsum("ki,ki->k", fk_quat, target_quat_wxyz)
            # base cost per k = (1 - dot²); we'll perturb each q[k, j] and read FK quat.
            for k in range(n):
                for j in range(7):
                    q_plus = q[k].copy(); q_plus[j] += eps
                    q_minus = q[k].copy(); q_minus[j] -= eps
                    qw_plus = model.fk_quat_wxyz(q_plus)
                    qw_minus = model.fk_quat_wxyz(q_minus)
                    if np.dot(qw_plus, target_quat_wxyz[k]) < 0:
                        qw_plus = -qw_plus
                    if np.dot(qw_minus, target_quat_wxyz[k]) < 0:
                        qw_minus = -qw_minus
                    dot_plus = float(np.dot(qw_plus, target_quat_wxyz[k]))
                    dot_minus = float(np.dot(qw_minus, target_quat_wxyz[k]))
                    d_cost = ((1.0 - dot_plus * dot_plus) - (1.0 - dot_minus * dot_minus)) / (2 * eps)
                    grad[7 * k + j] += weights.track_quat * d_cost
            del quat_dot  # silence flake; the central diff replaced the closed-form path

        # ---- collision: gradient = 0 (non-smooth count) ----
        return grad

    # Live SCO-level progress table. trust-constr's verbose=2 already prints
    # an internal CG / trust-radius column; this complementary line shows
    # what the *trajopt* cost is doing per outer iter so the user can follow
    # convergence in real time without parsing the JSON afterwards.
    # The live table shows `cart_z` (the safety-critical out-of-plane axis)
    # rather than the L2 norm — the per-axis box makes the z stat the actual
    # binding constraint. The full per-axis maxes are kept in iter_history
    # for offline review.
    header_line = (
        f"{'iter':>4} | {'cost':>11} | "
        f"{'time':>9} | {'jerk':>10} | {'trk_pos':>10} | {'trk_quat':>9} | "
        f"{'min_d':>7} | {'cart_z':>9} | {'cart_xy':>9} | {'vel_ratio':>9} | {'n_col':>5}"
    )

    def callback(xk: np.ndarray, state) -> bool:  # noqa: ARG001 (state from trust-constr)
        q, dt = _unpack(xk, n)
        total, components, fk_pos, _, coll = _cost_components(
            model, q, dt, target_pos, target_quat_wxyz, weights
        )
        cart_axes = fk_pos - target_pos                          # (N, 3) signed
        cart_dev = np.linalg.norm(cart_axes, axis=1)              # (N,) L2
        abs_axes = np.abs(cart_axes)
        max_dx = float(abs_axes[:, 0].max()) if abs_axes.size else 0.0
        max_dy = float(abs_axes[:, 1].max()) if abs_axes.size else 0.0
        max_dz = float(abs_axes[:, 2].max()) if abs_axes.size else 0.0
        max_dxy = max(max_dx, max_dy)
        diff = q[1:] - q[:-1]
        vel_ratio = float(
            np.max(np.abs(diff) / (dt[:, None] * model.v_max[None, :] + 1e-12))
        )
        # Min pair distance over the whole trajectory — the new hard
        # constraint's headroom. `+inf` means no pair was within d_warn.
        min_d = np.inf
        for i_k in range(n):
            md_k = model.min_pair_distance(q[i_k])
            if md_k < min_d:
                min_d = md_k
        min_d_disp = "  inf  " if np.isinf(min_d) else f"{min_d:>7.4f}"
        i = len(iter_history)
        if i == 0:
            print(f"[trajopt] {header_line}")
        print(
            f"[trajopt] {i:>4d} | {total:>11.4e} | "
            f"{components['time']:>9.3e} | {components['jerk']:>10.3e} | "
            f"{components['track_pos']:>10.3e} | {components['track_quat']:>9.3e} | "
            f"{min_d_disp} | {max_dz:>9.4f} | "
            f"{max_dxy:>9.4f} | {vel_ratio:>9.4f} | {int(coll.sum()):>5d}",
            flush=True,
        )
        iter_history.append({
            "iter": i,
            "cost": float(total),
            "components": components,
            "max_cart_dev": float(cart_dev.max()),
            "max_dx": max_dx,
            "max_dy": max_dy,
            "max_dz": max_dz,
            "max_vel_ratio": vel_ratio,
            "n_colliding": int(coll.sum()),
            "min_pair_distance": (None if np.isinf(min_d) else float(min_d)),
        })
        return False

    box = _build_box_bounds(
        q_warm,
        dt_warm,
        model.q_lo,
        model.q_hi,
        bounds.trust_radius,
        bounds.dt_min,
        bounds.dt_max,
        bounds.pin_endpoints,
    )

    constraints: list = [
        _vel_linear_constraint(model, n),
        _axis_cart_constraint(model, n, target_pos, cap_x, cap_y, cap_z),
        _collision_constraint(model, n, d_safe_used),
    ]

    x0 = _pack(q_warm.copy(), dt_warm.copy())

    print(
        f"[trajopt] N={n} steps, decision vars={x0.size}, "
        f"weights time={weights.time} jerk={weights.jerk} "
        f"track_pos={weights.track_pos} track_quat={weights.track_quat} "
        f"col={weights.collision}"
    )
    print(
        f"[trajopt] bounds trust={bounds.trust_radius} dt=[{bounds.dt_min},"
        f"{bounds.dt_max}] cart_box=(x={cap_x:.4f}, y={cap_y:.4f}, z={cap_z:.4f}) "
        f"pin={bounds.pin_endpoints}"
    )

    t0 = time.perf_counter()
    method = solver_opts.method.lower()
    if method == "ipopt":
        # IPOPT's interior-point method with restoration phase handles the
        # "warmstart-feasible / cost-pulls-infeasible" pathology that stalls
        # trust-constr on this problem. Same bounds / constraints / analytic
        # jac all reusable — cyipopt's `minimize_ipopt` is scipy-compatible.
        # Hessian is L-BFGS-approximated (the analytic cost-Hessian would
        # touch FK second derivatives we don't have).
        try:
            from cyipopt import minimize_ipopt
        except ImportError:
            # Soft-fall back to trust-constr when cyipopt isn't installed.
            # Flipping the default to ipopt should not break containers that
            # haven't run the optional `pip install cyipopt` step.
            print(
                "[solve] cyipopt not installed; falling back to "
                "scipy.optimize.minimize(method='trust-constr'). Install with "
                "`apt-get install -y coinor-libipopt-dev libblas-dev "
                "liblapack-dev && pip install cyipopt` to enable IPOPT."
            )
            method = "trust-constr"

    if method == "ipopt":
        # cyipopt.minimize_ipopt does not support a per-iter callback yet
        # (it raises NotImplementedError). IPOPT's own `print_level=5` table
        # covers the live progress role; the SCO-level iter_history is
        # therefore empty under IPOPT, and the `_iter.png` plot will be too.
        # Final cost decomposition is still recorded in solve_stats.
        res = minimize_ipopt(
            cost_fn,
            x0,
            jac=cost_grad_fn,
            bounds=box,
            constraints=constraints,
            options={
                "max_iter": solver_opts.max_iters,
                "tol": solver_opts.gtol,
                # Constraint feasibility tolerance — what makes "hard" hard
                # here. Default 1e-4 leaves ~0.1 mm slop on the 5 mm cart cap,
                # which is fine.
                "constr_viol_tol": 1e-6,
                "acceptable_constr_viol_tol": 1e-5,
                # L-BFGS Hessian approximation: we don't have FK second
                # derivatives, and BFGS works well on the smoothness cost.
                "hessian_approximation": "limited-memory",
                "limited_memory_max_history": 10,
                # IPOPT print_level 0..12; map our verbose 0/1/2 → 0/3/5.
                "print_level": [0, 3, 5][min(int(solver_opts.verbose), 2)],
                # MUMPS is what ships with coinor-libipopt-dev; ma27/ma57
                # would be faster but require HSL.
                "linear_solver": "mumps",
            },
        )
    else:
        res = minimize(
            cost_fn,
            x0,
            method=method,
            jac=cost_grad_fn,
            bounds=box,
            constraints=constraints,
            callback=callback,
            options={
                "maxiter": solver_opts.max_iters,
                "xtol": solver_opts.xtol,
                "gtol": solver_opts.gtol,
                "verbose": solver_opts.verbose,
            },
        )
    wall = time.perf_counter() - t0

    q_final, dt_final = _unpack(res.x, n)
    pos_final, quat_final, coll_final, q_full_final = _eval_fk_and_collision(
        model, q_final
    )
    quat_final = _quat_align_inplace(quat_final, target_quat_wxyz)
    cart_axes_final = pos_final - target_pos                      # (N, 3) signed
    cart_final = np.linalg.norm(cart_axes_final, axis=1)           # (N,) L2

    final_summary = _summarize(
        q_final, dt_final, cart_final, cart_axes_final, coll_final
    )
    total, components, _, _, _ = _cost_components(
        model, q_final, dt_final, target_pos, target_quat_wxyz, weights
    )

    diff = q_final[1:] - q_final[:-1]
    max_vel_ratio = float(
        np.max(np.abs(diff) / (dt_final[:, None] * model.v_max[None, :] + 1e-12))
    )

    # Post-solve SDF audit. `min_pair_distance(q[k])` reads the SAME sphere-BVH
    # the constraint used, so this is the apples-to-apples way to check the
    # constraint actually held. `+inf` means every waypoint cleared `d_warn`.
    min_d_after = np.inf
    for q_k in q_final:
        md_k = model.min_pair_distance(q_k)
        if md_k < min_d_after:
            min_d_after = md_k

    abs_axes_final = np.abs(cart_axes_final)
    solve_stats = {
        "ok": bool(res.success) and final_summary["n_collisions"] == 0,
        "status": str(res.message),
        "scipy_status": int(res.status) if hasattr(res, "status") else None,
        "n_iters": int(res.nit) if hasattr(res, "nit") else None,
        "n_fun_evals": int(eval_count["n"]),
        "wall_seconds": float(wall),
        "final_cost": float(total),
        "cost_components": {k: float(v) for k, v in components.items()},
        "max_cartesian_deviation": float(cart_final.max()),  # L2, reporting
        "max_dx": float(abs_axes_final[:, 0].max()) if abs_axes_final.size else 0.0,
        "max_dy": float(abs_axes_final[:, 1].max()) if abs_axes_final.size else 0.0,
        "max_dz": float(abs_axes_final[:, 2].max()) if abs_axes_final.size else 0.0,
        "max_joint_vel_ratio": max_vel_ratio,
        "n_collisions_after": int(coll_final.sum()),
        # New contract: per-pair sphere-SDF lower bound across the whole
        # post-solve trajectory. Must be ≥ d_safe_used for a clean solve.
        "min_pair_distance_after": (
            None if np.isinf(min_d_after) else float(min_d_after)
        ),
        # `d_safe_cfg` is the YAML value; `d_safe_used` is what the solver
        # actually saw. The optimizer no longer auto-relaxes, so these are
        # always equal — kept for downstream JSON-schema compatibility.
        "d_safe_cfg": float(model.collision_d_safe),
        "d_safe_used": float(d_safe_used),
        "d_warn_used": float(model.collision_d_warn),
        "min_pair_distance_warm": (
            None if np.isinf(min_d_warm) else float(min_d_warm)
        ),
        "warmstart_summary": warmstart_summary,
        "final_summary": final_summary,
        "iter_history": iter_history,
        "cart_cap_used": {"x": float(cap_x), "y": float(cap_y), "z": float(cap_z)},
        # Total trajectory time before/after the SCO. `topp_floor_total_time`
        # is the sum of the retimed warmstart dt — what would have been the
        # minimum back when dt_lo was clamped to dt_warm. Now that dt_lo is
        # freed to dt_min, `total_time_after` can drop below this floor on
        # any segment where the planner's dt was looser than the vel floor.
        "topp_floor_total_time": float(dt_warm.sum()),
        "total_time_after": float(dt_final.sum()),
    }

    return SolverResult(
        q_arm=q_final,
        dt=dt_final,
        achieved_pos=pos_final,
        achieved_quat_wxyz=quat_final,
        cart_dev=cart_final,
        cart_dev_axes=cart_axes_final,
        self_coll=coll_final,
        q_full=q_full_final,
        solve_stats=solve_stats,
    )
