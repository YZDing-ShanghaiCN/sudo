"""Local-DP refinement (Phase 5b — alternative to IPOPT/trust-constr).

Selected via `optimize.solver.method: local_dp` in YAML. Operates on the
TOPP-pre-retimed warm-start the same way the smooth optimizer does, but
instead of nudging every joint via a sequential convex solve it:

  1. Sparse-samples ~1/N of the warm trajectory (stride configurable;
     default 4).
  2. Per sparse layer, sweeps a 3D `(du, dv, yaw)` grid around the warm
     target, runs analytical IK (with `nb_redundant_search` redundancy)
     at each grid pose, dedups + keeps the top-K closest to the warm
     `q_arm` (default 100).
  3. Runs the same iterative-mask DP Phase 1 uses (`_kshortest_dp_phase1`
     pattern: weighted-L2 edge cost + optional acceleration term + L_inf
     jump-prune + endpoint-collision mask loop) over the sparse layers'
     pools.
  4. Time-parametrizes the DP-selected sparse skeleton via TOPP-RA
     (`toppra.algorithm.TOPPRA`) under joint velocity + acceleration
     constraints, then samples the resulting parametrization at uniform
     `1/hz` time points. Path geometry is preserved; only timing
     changes. Output is uniform 1/20 Hz with v_max + a_max respected by
     construction.
  5. Packs the resulting `q_trajectory` into a `SolverResult` so the
     existing `write_optimizer_json` path consumes it unchanged.

IK + collision queries go through the shared `robot_check` primitives so
the planner / refinement / optimizer audit agree on collision verdicts.
The module-level `_collision_checker` singleton is (re-)initialized at
the top of `solve_local_dp`, mirroring `robot_check.check_trajectory`.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# robot_check lives at the top level of `tools/surface_cover_planner/`;
# `plan.py` already imports it from the same path so this import resolves
# when local_dp is invoked from `plan._run_optimizer`.
import robot_check as rc
from topp import slerp_wxyz

from .io import WarmStart
from .robot_model import HBMP_COMPONENT_DOF_INDICES, RobotModel
from .solver import OptBounds, SolverResult

try:
    import toppra as _toppra
    import toppra.algorithm as _toppra_algo
    import toppra.constraint as _toppra_cstr
except ImportError as _exc:  # pragma: no cover — docker-only dependency
    _toppra = None  # type: ignore[assignment]
    _toppra_algo = None  # type: ignore[assignment]
    _toppra_cstr = None  # type: ignore[assignment]
    _TOPPRA_IMPORT_ERROR = _exc
else:
    _TOPPRA_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Per-layer pool building
# ---------------------------------------------------------------------------

def _build_local_pool(
    agent: Any,
    initial_q: np.ndarray,
    frame: Any,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    plane_u_axis: np.ndarray,
    plane_v_axis: np.ndarray,
    tool_offset_inv_4x4: np.ndarray | None,
    xy_offsets_m: list[float],
    yaw_offsets_deg: list[float],
    nb_redundant_search: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """Sweep the (du, dv, yaw) grid; collect deduped status-ok IK pool.

    Pool entries are `(q_arm, pos_used, quat_used)`. Dedup at 1e-3 rad
    (matches Phase 1 / 3). Collision is NOT filtered here — the DP
    endpoint-mask loop handles that. Returns (pool, n_ik_calls).
    """
    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    n_ik_calls = 0

    for du in xy_offsets_m:
        for dv in xy_offsets_m:
            pos_used = (
                target_pos
                + float(du) * plane_u_axis
                + float(dv) * plane_v_axis
            ).astype(np.float64)
            for yaw_deg in yaw_offsets_deg:
                yaw_f = float(yaw_deg)
                if abs(yaw_f) <= 1e-9:
                    quat_used = target_quat.copy()
                else:
                    quat_used = rc._quat_rpy_perturb(target_quat, (0.0, 0.0, yaw_f))
                tf_target = rc._tool_tf_to_gripper_tf(
                    rc._build_target_tf(pos_used, quat_used), tool_offset_inv_4x4
                )
                agent.update_kin(initial_q)
                ik_result = agent.get_ik(
                    tf_target, frame, nb_redundant_search=nb_redundant_search
                )
                n_ik_calls += 1
                for _which_ik, (status_batch, q_batch) in ik_result.items():
                    for i in range(len(status_batch)):
                        if not status_batch[i]:
                            continue
                        q_arm = np.asarray(q_batch[i], dtype=np.float64).copy()
                        qkey = tuple(int(round(float(v_) * 1000.0)) for v_ in q_arm)
                        if qkey in seen:
                            continue
                        seen.add(qkey)
                        pool.append((q_arm, pos_used.copy(), quat_used.copy()))
    return pool, n_ik_calls


def _topk_closest(
    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    anchor_q_arm: np.ndarray,
    k: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sort the pool by weighted L2 distance to `anchor_q_arm`; keep first k."""
    if k <= 0 or not pool:
        return list(pool)
    scored = sorted(
        ((rc._weighted_arm_dist(entry[0], anchor_q_arm), idx)
         for idx, entry in enumerate(pool)),
        key=lambda t: t[0],
    )
    return [pool[idx] for _dist, idx in scored[:k]]


# ---------------------------------------------------------------------------
# Iterative-mask DP across the sparse layers
# ---------------------------------------------------------------------------

def _run_kshortest_dp(
    pools: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    *,
    base_q: np.ndarray,
    arm_indices: list[int],
    jump_threshold: float,
    accel_weight: float,
    max_iterations: int,
    cost_mode: str = "fastest",
    v_max: np.ndarray | None = None,
    a_max: np.ndarray | None = None,
) -> tuple[list[int] | None, dict]:
    """Iterative-mask DP — direct port of `_kshortest_dp_phase1`'s DP loop.

    `cost_mode` picks the per-edge cost:
      - "fastest" (default): envelope time
            t_vel   = max_j |Δq[j]| / v_max[j]
            t_accel = sqrt(max_j |Δq[j]| / a_max[j])
            edge    = max(t_vel, t_accel)
        Approximates the per-segment dt Phase-2 TOPP will produce, so the
        DP minimizes realized trajectory time. `accel_weight` is ignored
        (t_accel already accounts for accel per edge).
      - "shortest": legacy weighted-L2 joint distance with JOINT_DIST_WEIGHTS,
        plus the optional `accel_weight` grandparent corner-accel surcharge.

    Returns `(selected_indices, stats)` where `selected_indices[L]` is
    the chosen candidate index inside `pools[L]`, or `None` on
    infeasibility / iteration-cap exceeded. `stats` records iteration
    count, DP / collision-check wall times, and total endpoint masks.
    """
    if cost_mode not in ("fastest", "shortest"):
        raise ValueError(
            f"cost_mode must be 'fastest' or 'shortest', got {cost_mode!r}"
        )
    if cost_mode == "fastest":
        if v_max is None or a_max is None:
            raise ValueError(
                "cost_mode='fastest' requires both v_max and a_max"
            )
        v_max_arr = np.asarray(v_max, dtype=np.float64).reshape(-1)
        a_max_arr = np.asarray(a_max, dtype=np.float64).reshape(-1)
        if v_max_arr.shape != (7,) or a_max_arr.shape != (7,):
            raise ValueError(
                f"v_max / a_max must be length 7, got "
                f"{v_max_arr.shape}, {a_max_arr.shape}"
            )

    N = len(pools)
    weights_arr = np.asarray(rc.JOINT_DIST_WEIGHTS, dtype=np.float64)

    # Pre-cache pool joint arrays and edge-cost matrices.
    pools_arr: list[np.ndarray] = [
        np.stack([entry[0] for entry in layer], axis=0).astype(np.float64, copy=False)
        for layer in pools
    ]
    edge_cost_mats: list[np.ndarray] = []
    for L in range(1, N):
        prev_qs = pools_arr[L - 1]
        next_qs = pools_arr[L]
        diffs = next_qs[:, None, :] - prev_qs[None, :, :]
        abs_diffs = np.abs(diffs)
        if cost_mode == "fastest":
            # Per-DOF time bounds, take L∞ over DOFs (worst-binding joint).
            t_vel = (abs_diffs / v_max_arr).max(axis=-1)
            t_acc = np.sqrt((abs_diffs / a_max_arr).max(axis=-1))
            e = np.maximum(t_vel, t_acc)
        else:
            e = np.sqrt((abs_diffs * abs_diffs) @ weights_arr)
        e = np.where(abs_diffs.max(axis=-1) <= jump_threshold, e, np.inf)
        edge_cost_mats.append(e)

    pool_sizes = [arr.shape[0] for arr in pools_arr]
    mask: list[np.ndarray] = [np.zeros(P_L, dtype=bool) for P_L in pool_sizes]
    endpoint_cache: dict[tuple[int, int], bool] = {}

    total_endpoint_masks = 0
    total_dp_time = 0.0
    total_check_time = 0.0
    total_new_endpoint_checks = 0
    last_endpoint_failures = 0

    for it in range(max_iterations):
        t_dp_start = time.perf_counter()
        cost: list[np.ndarray] = [np.full(P_L, np.inf, dtype=np.float64) for P_L in pool_sizes]
        parent: list[np.ndarray] = [np.full(P_L, -1, dtype=np.int32) for P_L in pool_sizes]
        cost[0][:] = 0.0
        cost[0][mask[0]] = np.inf
        for L in range(1, N):
            cands = cost[L - 1][None, :] + edge_cost_mats[L - 1]
            # The grandparent corner-accel surcharge only applies to the
            # weighted-L2 "shortest" cost; in "fastest" mode the per-edge
            # envelope already accounts for accel cost (t_accel term).
            if cost_mode == "shortest" and accel_weight > 0.0 and L >= 2:
                gp_idx = parent[L - 1]
                safe_gp = np.where(gp_idx >= 0, gp_idx, 0)
                q_gp = pools_arr[L - 2][safe_gp]
                q_par = pools_arr[L - 1]
                q_cur = pools_arr[L]
                accel = (
                    q_cur[:, None, :]
                    - 2.0 * q_par[None, :, :]
                    + q_gp[None, :, :]
                )
                accel_cost = np.sqrt((accel * accel) @ weights_arr)
                no_gp = gp_idx < 0
                if no_gp.any():
                    accel_cost[:, no_gp] = 0.0
                cands = cands + accel_weight * accel_cost
            parent[L] = np.argmin(cands, axis=1).astype(np.int32)
            cost[L] = np.take_along_axis(cands, parent[L][:, None], axis=1).reshape(-1)
            cost[L][mask[L]] = np.inf
        total_dp_time += time.perf_counter() - t_dp_start

        if not np.isfinite(cost[N - 1]).any():
            deepest = -1
            for L in range(N - 1, -1, -1):
                if np.isfinite(cost[L]).any():
                    deepest = L
                    break
            return None, {
                "infeasible": True,
                "deepest_layer": int(deepest),
                "iterations_run": int(it + 1),
                "total_endpoint_masks": int(total_endpoint_masks),
                "total_dp_time": float(total_dp_time),
                "total_check_time": float(total_check_time),
                "total_new_endpoint_checks": int(total_new_endpoint_checks),
                "last_endpoint_failures": int(last_endpoint_failures),
                "cost_mode": cost_mode,
            }

        end_j = int(np.argmin(cost[N - 1]))
        path_cost = float(cost[N - 1][end_j])
        chain: list[tuple[int, int]] = [(N - 1, end_j)]
        cur_L, cur_j = N - 1, end_j
        while cur_L > 0:
            par = int(parent[cur_L][cur_j])
            chain.append((cur_L - 1, par))
            cur_L, cur_j = cur_L - 1, par
        chain.reverse()

        t_check_start = time.perf_counter()
        new_endpoint_failures: list[tuple[int, int]] = []
        new_endpoint_checks = 0
        for L_idx, j_idx in chain:
            ep_key = (L_idx, j_idx)
            ep_hit = endpoint_cache.get(ep_key)
            if ep_hit is None:
                ep_hit = rc._endpoint_collides(
                    None, base_q, arm_indices, pools[L_idx][j_idx][0]
                )
                endpoint_cache[ep_key] = ep_hit
                new_endpoint_checks += 1
            if ep_hit:
                new_endpoint_failures.append((L_idx, j_idx))
        total_check_time += time.perf_counter() - t_check_start
        total_new_endpoint_checks += new_endpoint_checks
        last_endpoint_failures = len(new_endpoint_failures)

        cost_unit = "s" if cost_mode == "fastest" else "rad"
        rc._print_progress(
            "Phase 5b: local DP ", it + 1, max_iterations,
            suffix=(
                f"cost={path_cost:.3f}{cost_unit} "
                f"bad_ep={last_endpoint_failures} "
                f"masked={total_endpoint_masks} "
                f"new_checks(ep)={new_endpoint_checks} "
                f"total_new(ep)={total_new_endpoint_checks} "
                f"total(dp/check)={total_dp_time:.3f}/{total_check_time:.3f}s"
            ),
        )

        if not new_endpoint_failures:
            sys.stderr.write("\n")
            sys.stderr.flush()
            selected = [j for _L, j in chain]
            return selected, {
                "infeasible": False,
                "iterations_run": int(it + 1),
                "total_endpoint_masks": int(total_endpoint_masks),
                "total_dp_time": float(total_dp_time),
                "total_check_time": float(total_check_time),
                "total_new_endpoint_checks": int(total_new_endpoint_checks),
                "final_cost": float(path_cost),
                "cost_mode": cost_mode,
                "final_cost_units": "s" if cost_mode == "fastest" else "rad",
            }

        for L_idx, j_idx in new_endpoint_failures:
            if not mask[L_idx][j_idx]:
                mask[L_idx][j_idx] = True
                total_endpoint_masks += 1

    sys.stderr.write("\n")
    sys.stderr.flush()
    return None, {
        "infeasible": False,
        "iteration_cap_exceeded": True,
        "iterations_run": int(max_iterations),
        "total_endpoint_masks": int(total_endpoint_masks),
        "total_dp_time": float(total_dp_time),
        "total_check_time": float(total_check_time),
        "total_new_endpoint_checks": int(total_new_endpoint_checks),
        "last_endpoint_failures": int(last_endpoint_failures),
        "cost_mode": cost_mode,
    }


# ---------------------------------------------------------------------------
# Re-densification via TOPP-RA (Time-Optimal Path Parameterization)
# ---------------------------------------------------------------------------

def _toppra_densify(
    *,
    agent: Any,
    arm_indices: list[int],
    frame: Any,
    tool_offset: np.ndarray | None,
    selected_q_arms: list[np.ndarray],
    selected_pos: list[np.ndarray],
    selected_quats: list[np.ndarray],
    base_q_full: np.ndarray,
    hz: float,
    v_max: np.ndarray,
    a_max: np.ndarray,
    max_cartesian_deviation: float,
    q_trajectory: list,
    failures: list,
) -> dict:
    """TOPP-RA time-parametrization of the DP-selected sparse skeleton.

    Path geometry is preserved; only timing changes. The output is
    sampled at uniform ``1 / hz`` and respects ``v_max`` / ``a_max``
    exactly under toppra's ``ParametrizeConstAccel`` post-processor.

    Step A — build a cubic-spline path through ``selected_q_arms``
    (``toppra.SplineInterpolator`` over ``ss ∈ [0, 1]``).

    Step B — set ``JointVelocityConstraint(v_max)`` and
    ``JointAccelerationConstraint(a_max)`` (each as a (7, 2) lo/hi
    array).

    Step C — run ``toppra.algorithm.TOPPRA`` with
    ``parametrizer="ParametrizeConstAccel"`` and get the parametrized
    trajectory. Infeasibility (``compute_trajectory() is None``)
    surfaces as a `RuntimeError` so the caller can adjust caps.

    Step D — sample uniformly at ``1 / hz``: ``t_sample = linspace(0,
    duration, n_dense)``; ``q_arm_dense = jnt_traj(t_sample)``;
    velocities / accelerations from ``evald`` / ``evaldd`` for the audit.

    Step E — recover ``s(t)`` from toppra's grid (``gridpoints`` and
    ``sd_vec``: integrate ``ds / sd_avg`` over the grid to get
    ``t_grid``, then ``s_at_t = np.interp(t_sample, t_grid,
    gridpoints)``). Use ``s_at_t`` as a fractional sparse index to
    linearly interpolate the sparse ``pos`` skeleton and slerp the
    sparse ``quat`` skeleton — these become the per-sample
    ``target_position`` / ``target_quat_wxyz`` reported on each
    emitted ``QPoint``.

    Step F — emit ``QPoint``s with FK + self-collision gates: deviation
    > ``max_cartesian_deviation`` → ``kind="densify_deviation"`` in
    ``failures``; self-collision → ``kind="densify_collision"``. Both
    failure dicts use the same shape ``_phase2_topp_smooth_densify``
    emits, so downstream consumers don't need to change.

    Returns the stats dict that lands in
    ``solve_stats.local_dp.densify``.
    """
    if _toppra is None:
        raise ImportError(
            "toppra is required for _toppra_densify. It ships pre-installed in "
            "the SudoDeploy `sudodeploy_full_dev` docker image (path: "
            "`/usr/local/lib/python3.10/dist-packages/toppra/`). Install via "
            "`pip install toppra` if running outside the container."
        ) from _TOPPRA_IMPORT_ERROR

    n_sparse = len(selected_q_arms)
    if n_sparse < 2:
        raise ValueError(
            f"_toppra_densify needs ≥ 2 sparse waypoints, got {n_sparse}"
        )

    # Step A — stack sparse inputs + build the spline path.
    q_arm_in = np.asarray(selected_q_arms, dtype=np.float64)              # (N, 7)
    pos_in = np.asarray(selected_pos, dtype=np.float64)                    # (N, 3)
    quat_in = np.asarray(selected_quats, dtype=np.float64)                 # (N, 4)
    ss = np.linspace(0.0, 1.0, n_sparse)
    # SplineInterpolator wants ≥ 3 points; pad with the segment midpoint
    # when the DP picked only 2 waypoints (rare in practice — the warm
    # trajectory's stride bound usually gives many more).
    if n_sparse == 2:
        q_path = np.vstack([q_arm_in[0], 0.5 * (q_arm_in[0] + q_arm_in[1]), q_arm_in[1]])
        ss_path = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    else:
        q_path = q_arm_in
        ss_path = ss
    path = _toppra.SplineInterpolator(ss_path, q_path)

    # Step B — joint velocity + acceleration constraints (lo, hi columns).
    pc_vel = _toppra_cstr.JointVelocityConstraint(
        np.vstack([-v_max, v_max]).T
    )
    pc_acc = _toppra_cstr.JointAccelerationConstraint(
        np.vstack([-a_max, a_max]).T
    )

    # Step C — run TOPP-RA. None ↔ infeasible problem.
    instance = _toppra_algo.TOPPRA(
        [pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel"
    )
    jnt_traj = instance.compute_trajectory()
    if jnt_traj is None:
        raise RuntimeError(
            f"[local_dp] TOPP-RA infeasible: n_sparse={n_sparse}, "
            f"v_max={v_max.tolist()}, a_max={a_max.tolist()}. Check that the "
            f"sparse skeleton is C¹ — an IK branch flip with discontinuous "
            f"tangent will make the path inadmissible."
        )

    duration = float(jnt_traj.duration)
    dt_out = 1.0 / float(hz)
    # Exact 1/hz spacing with the final sample pinned at `duration` (matches
    # the planner-side `topp.resample_to_uniform_grid` contract). The last
    # interval may be < dt_out when duration is not an integer multiple of
    # dt_out — accept that to keep endpoint pinning + cadence elsewhere.
    n_full = int(np.floor(duration / dt_out))
    t_sample = np.arange(n_full + 1, dtype=np.float64) * dt_out
    if t_sample[-1] < duration - 1e-9:
        t_sample = np.append(t_sample, duration)
    n_dense = int(t_sample.size)

    # Step D — sample q / qd / qdd at uniform 1/hz.
    q_arm_dense = jnt_traj.eval(t_sample)       # (n_dense, 7)
    qd_dense = jnt_traj.evald(t_sample)         # (n_dense, 7)
    qdd_dense = jnt_traj.evaldd(t_sample)       # (n_dense, 7)

    # Step E — recover s(t) from toppra's grid to interpolate target pos/quat.
    pd = instance.problem_data
    s_grid = np.asarray(pd.gridpoints, dtype=np.float64)
    sd_grid = np.asarray(pd.sd_vec, dtype=np.float64)
    ds = np.diff(s_grid)
    sd_avg = 0.5 * (sd_grid[:-1] + sd_grid[1:])
    seg_dt = ds / np.maximum(sd_avg, 1e-12)
    t_grid = np.concatenate([[0.0], np.cumsum(seg_dt)])
    s_at_t = np.interp(t_sample, t_grid, s_grid)
    # Map path parameter s ∈ [0, 1] back to the *sparse* index space
    # (pos_in / quat_in are indexed by the DP-selected layer, regardless
    # of the spline-padding tweak in Step A).
    f_at_t = s_at_t * (n_sparse - 1)
    i_lo = np.floor(f_at_t).astype(int)
    i_lo = np.clip(i_lo, 0, n_sparse - 2)
    alpha = f_at_t - i_lo
    target_pos_dense = (
        (1.0 - alpha)[:, None] * pos_in[i_lo] + alpha[:, None] * pos_in[i_lo + 1]
    )

    # Step F — emit QPoints with FK + collision gates.
    n_collision_failures = 0
    n_dev_failures = 0

    for k in range(n_dense):
        q_full_k = base_q_full.copy()
        q_full_k[arm_indices] = q_arm_dense[k]
        target_pos_k = target_pos_dense[k]
        target_quat_k = slerp_wxyz(quat_in[i_lo[k]], quat_in[i_lo[k] + 1], float(alpha[k]))

        agent.update_kin(q_full_k, None, False)
        fk_matrix = agent.get_fk(frame).matrix
        achieved = rc._gripper_fk_to_tool_position(fk_matrix, tool_offset)
        achieved_quat = rc._gripper_fk_to_tool_quat_wxyz(fk_matrix, tool_offset)
        dev = float(np.linalg.norm(achieved - target_pos_k))

        cause: str | None = None
        if dev > max_cartesian_deviation:
            cause = (
                f"Cartesian deviation {dev:.4f}m exceeds "
                f"{max_cartesian_deviation:.4f}m"
            )
            failures.append({
                "user_wp_index": k,
                "user_wp_name": f"local_dp_{k}",
                "sub_index": 0,
                "kind": "densify_deviation",
                "cause": cause,
                "deviation": dev,
                "target_position": target_pos_k.tolist(),
                "target_quat_wxyz": target_quat_k.tolist(),
                "achieved_position": achieved.tolist(),
            })
            n_dev_failures += 1
        if rc._is_self_colliding(q_full_k):
            col_cause = "self-collision on toppra densify sample"
            failures.append({
                "user_wp_index": k,
                "user_wp_name": f"local_dp_{k}",
                "sub_index": 0,
                "kind": "densify_collision",
                "cause": col_cause,
                "target_position": target_pos_k.tolist(),
                "target_quat_wxyz": target_quat_k.tolist(),
            })
            if cause is None:
                cause = col_cause
            n_collision_failures += 1

        q_trajectory.append(rc._make_qpoint(
            t_global=float(t_sample[k]),
            user_wp_index=k,
            sub_index=0,
            target_position=target_pos_k,
            target_quat_wxyz=target_quat_k,
            achieved_position=achieved,
            achieved_quat_wxyz=achieved_quat,
            cartesian_deviation=dev,
            q_full=q_full_k,
            ok=cause is None,
            failure_cause=cause,
        ))

    # Vel / accel ratios straight from toppra's qd / qdd — these match the
    # constraints toppra enforced. By construction both should be ≤ 1
    # (modulo small numerical drift from the const-accel parametrization).
    max_vel_ratio = float((np.abs(qd_dense) / v_max[None, :]).max())
    max_accel_ratio = float((np.abs(qdd_dense) / a_max[None, :]).max())

    return {
        "method": "toppra",
        "failed": False,
        "n_sparse_in": int(n_sparse),
        "n_dense": int(n_dense),
        "total_time": duration,
        "max_vel_ratio": max_vel_ratio,
        "max_accel_ratio": max_accel_ratio,
        "n_collision_failures": int(n_collision_failures),
        "n_dev_failures": int(n_dev_failures),
        "joint_vel_limits": v_max.tolist(),
        "joint_accel_limits": a_max.tolist(),
        "hz": float(hz),
        "dt_out": dt_out,
        "toppra_n_gridpoints": int(s_grid.size),
    }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def solve_local_dp(
    *,
    warm: WarmStart,
    model: RobotModel,
    bounds: OptBounds,
    lp_cfg: dict,
    hz: float = 30.0,
    ee_speed: float = 0.05,
    max_cartesian_deviation: float = 0.005,
) -> SolverResult:
    """Run the local-DP refinement and return a `SolverResult`.

    Required inputs from `warm`: `q_arm`, `dt`, `tcp_target_pos`,
    `tcp_target_quat_wxyz`, `arm`, `initial_q`, `tool_offset_4x4`,
    `joint_vel_limits`, `plane_u_axis`, `plane_v_axis`.

    Required inputs from `lp_cfg` (YAML `optimize.local_dp`): see
    config_autorun.yaml's local_dp block for the canonical schema.
    """
    if warm.plane_u_axis is None or warm.plane_v_axis is None:
        raise ValueError(
            "solve_local_dp requires warm.plane_u_axis / plane_v_axis. "
            "Re-generate the warmstart JSON with the planner so it carries "
            "metadata.plane_u_axis / plane_v_axis, or run end-to-end with "
            "robot.enabled: true (the planner threads the plane axes through "
            "from PlanResult.plane)."
        )
    plane_u_axis = np.asarray(warm.plane_u_axis, dtype=np.float64)
    plane_v_axis = np.asarray(warm.plane_v_axis, dtype=np.float64)

    arm = warm.arm
    arm_indices = HBMP_COMPONENT_DOF_INDICES[arm]
    initial_q = np.asarray(warm.initial_q, dtype=np.float64)
    if initial_q.shape != (16,):
        raise ValueError(f"warm.initial_q must be length 16, got {initial_q.shape}")
    tool_offset_4x4 = (
        None
        if warm.tool_offset_4x4 is None or np.allclose(warm.tool_offset_4x4, np.eye(4))
        else np.asarray(warm.tool_offset_4x4, dtype=np.float64)
    )
    tool_offset_inv = (
        None if tool_offset_4x4 is None else np.linalg.inv(tool_offset_4x4)
    )

    if warm.joint_vel_limits is None:
        v_max = np.full(7, 1.0, dtype=np.float64)
    elif isinstance(warm.joint_vel_limits, (int, float)):
        v_max = np.full(7, float(warm.joint_vel_limits), dtype=np.float64)
    else:
        v_max = np.asarray(warm.joint_vel_limits, dtype=np.float64)
        if v_max.shape != (7,):
            raise ValueError(
                f"warm.joint_vel_limits must broadcast to length 7, got "
                f"{v_max.shape}"
            )

    # YAML knobs (with sensible defaults).
    stride = int(lp_cfg.get("sample_stride", 4))
    xy_offsets_m = list(lp_cfg.get("xy_offsets_m", [-0.02, -0.01, 0.0, 0.01, 0.02]))
    yaw_offsets_deg = list(
        lp_cfg.get("yaw_offsets_deg", list(range(-5, 6)))
    )
    top_k = int(lp_cfg.get("top_k", 100))
    nb_redundant_search = int(lp_cfg.get("nb_redundant_search", 64))
    jump_threshold = float(lp_cfg.get("jump_threshold", 1.5))
    accel_weight = float(lp_cfg.get("accel_weight", 0.3))
    max_iterations = int(lp_cfg.get("max_iterations", 20000))
    cost_mode = str(lp_cfg.get("cost_mode", "fastest"))
    if cost_mode not in ("fastest", "shortest"):
        raise ValueError(
            f"optimize.local_dp.cost_mode must be 'fastest' or 'shortest', "
            f"got {cost_mode!r}"
        )
    densify_cfg = lp_cfg.get("densify") or {}

    # a_max for the fastest-path DP cost — same value the Phase-2 TOPP
    # retime will use, broadcast to length 7.
    a_max_raw = densify_cfg.get("joint_accel_limits", 2.0)
    if isinstance(a_max_raw, (int, float)):
        a_max = np.full(7, float(a_max_raw), dtype=np.float64)
    else:
        a_max = np.asarray(a_max_raw, dtype=np.float64).reshape(-1)
        if a_max.shape != (7,):
            raise ValueError(
                f"optimize.local_dp.densify.joint_accel_limits must broadcast "
                f"to length 7, got shape {a_max.shape}"
            )

    wall = dict(model.cfg.wall or {})

    # (Re-)build the shared collision checker singleton so robot_check's
    # _endpoint_collides / _is_self_colliding work. `_setup_collision_checker`
    # only reads `cfg.wall`; the rest of the RobotCheckConfig fields are
    # irrelevant for the optimizer-only path now that the skip_walk densify
    # below replaces the planner-side `_phase2_topp_smooth_densify`.
    rc._setup_collision_checker(rc.RobotCheckConfig(wall=wall))
    agent = rc._collision_checker["kin"]
    frame = (
        agent.Frames.left_link_tactile_center
        if arm == "left_arm"
        else agent.Frames.right_link_tactile_center
    )

    # ---- Sparse sample ----
    N = int(warm.q_arm.shape[0])
    if N < 2:
        raise ValueError(f"warm trajectory too short for local_dp: N={N}")
    sparse_idx = list(range(0, N, max(stride, 1)))
    if sparse_idx[-1] != N - 1:
        sparse_idx.append(N - 1)
    n_sparse = len(sparse_idx)
    print(
        f"[local_dp] sparse N={n_sparse} from warm N={N} (stride={stride}); "
        f"grid xy={len(xy_offsets_m)}x{len(xy_offsets_m)} yaw={len(yaw_offsets_deg)} "
        f"({len(xy_offsets_m) ** 2 * len(yaw_offsets_deg)} perturbations/layer), "
        f"top_k={top_k}, nb_redundant={nb_redundant_search}, cost_mode={cost_mode}"
    )

    # ---- Per-layer pool build ----
    pools: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    pool_sizes_raw: list[int] = []
    total_ik_calls = 0
    t_pool_start = time.perf_counter()
    for L, s in enumerate(sparse_idx):
        pool_raw, n_calls = _build_local_pool(
            agent=agent,
            initial_q=initial_q,
            frame=frame,
            target_pos=warm.tcp_target_pos[s].astype(np.float64),
            target_quat=warm.tcp_target_quat_wxyz[s].astype(np.float64),
            plane_u_axis=plane_u_axis,
            plane_v_axis=plane_v_axis,
            tool_offset_inv_4x4=tool_offset_inv,
            xy_offsets_m=xy_offsets_m,
            yaw_offsets_deg=yaw_offsets_deg,
            nb_redundant_search=nb_redundant_search,
        )
        total_ik_calls += n_calls
        pool_sizes_raw.append(len(pool_raw))
        if not pool_raw:
            raise RuntimeError(
                f"[local_dp] empty IK pool at sparse layer {L} "
                f"(warm index {s}); widen xy_offsets_m / yaw_offsets_deg or "
                f"raise nb_redundant_search"
            )
        # Anchor on the warm trajectory's q_arm at this index for the
        # top-K closeness ranking — this is the "closest to original
        # trajectory" criterion.
        anchor = warm.q_arm[s].astype(np.float64)
        # Always include the warm q_arm itself as the canonical anchor
        # (so the DP can recover the warm-start exactly if no
        # perturbation pays off). It still passes through the dedup
        # because dedup happens inside `_build_local_pool`; if the warm
        # q_arm matches an existing entry up to 1e-3 rad it's already
        # there.
        qkey = tuple(int(round(float(v_) * 1000.0)) for v_ in anchor)
        in_pool = any(
            tuple(int(round(float(v_) * 1000.0)) for v_ in e[0]) == qkey
            for e in pool_raw
        )
        if not in_pool:
            pool_raw.append((
                anchor.copy(),
                warm.tcp_target_pos[s].astype(np.float64).copy(),
                warm.tcp_target_quat_wxyz[s].astype(np.float64).copy(),
            ))
        pool_topk = _topk_closest(pool_raw, anchor, top_k)
        pools.append(pool_topk)
        rc._print_progress(
            "Phase 5b: pools     ", L + 1, n_sparse,
            suffix=(
                f"raw_avg={sum(pool_sizes_raw) / (L + 1):.0f} "
                f"top_k={top_k} ik_calls={total_ik_calls}"
            ),
        )
    sys.stderr.write("\n")
    sys.stderr.flush()
    t_pool_total = time.perf_counter() - t_pool_start

    # ---- Iterative-mask DP ----
    selected_indices, dp_stats = _run_kshortest_dp(
        pools,
        base_q=initial_q,
        arm_indices=arm_indices,
        jump_threshold=jump_threshold,
        accel_weight=accel_weight,
        max_iterations=max_iterations,
        cost_mode=cost_mode,
        v_max=v_max,
        a_max=a_max,
    )
    if selected_indices is None:
        raise RuntimeError(
            f"[local_dp] DP failed: {dp_stats}. Try raising top_k / "
            f"nb_redundant_search, widening yaw_offsets_deg / xy_offsets_m, "
            f"or raising max_iterations."
        )

    # ---- Sparse skeleton (q_arm, pos, quat) per layer ----
    selected_q_arms = [pools[L][selected_indices[L]][0] for L in range(n_sparse)]
    selected_pos = [pools[L][selected_indices[L]][1] for L in range(n_sparse)]
    selected_quats = [pools[L][selected_indices[L]][2] for L in range(n_sparse)]

    # ---- TOPP-RA re-densification ----
    q_trajectory: list[rc.QPoint] = []
    failures: list[dict] = []
    densify_stats = _toppra_densify(
        agent=agent,
        arm_indices=arm_indices,
        frame=frame,
        tool_offset=tool_offset_4x4,
        selected_q_arms=selected_q_arms,
        selected_pos=selected_pos,
        selected_quats=selected_quats,
        base_q_full=initial_q,
        hz=hz,
        v_max=v_max,
        a_max=a_max,
        max_cartesian_deviation=max_cartesian_deviation,
        q_trajectory=q_trajectory,
        failures=failures,
    )

    if densify_stats.get("failed", False):
        raise RuntimeError(
            f"[local_dp] toppra re-densification failed: {densify_stats}; "
            f"failures={failures[:3]}"
        )

    # ---- Pack into SolverResult ----
    return _pack_solver_result(
        q_trajectory=q_trajectory,
        arm_indices=arm_indices,
        sparse_idx=sparse_idx,
        n_warm=N,
        pool_sizes_raw=pool_sizes_raw,
        top_k=top_k,
        total_ik_calls=total_ik_calls,
        t_pool_total=t_pool_total,
        dp_stats=dp_stats,
        densify_stats=densify_stats,
        bounds=bounds,
        v_max=v_max,
        failures=failures,
        warm_total_time=float(warm.dt.sum()) if warm.dt.size else 0.0,
    )


def _pack_solver_result(
    *,
    q_trajectory: list[rc.QPoint],
    arm_indices: list[int],
    sparse_idx: list[int],
    n_warm: int,
    pool_sizes_raw: list[int],
    top_k: int,
    total_ik_calls: int,
    t_pool_total: float,
    dp_stats: dict,
    densify_stats: dict,
    bounds: OptBounds,
    v_max: np.ndarray,
    failures: list[dict],
    warm_total_time: float,
) -> SolverResult:
    """Flatten q_trajectory into SolverResult dataclass."""
    M = len(q_trajectory)
    if M < 2:
        raise RuntimeError(
            f"[local_dp] re-densified trajectory has only {M} samples; "
            f"check Phase-2 IK failures in `failures`"
        )

    q_arm = np.zeros((M, 7), dtype=np.float64)
    q_full = np.zeros((M, 16), dtype=np.float64)
    target_pos = np.zeros((M, 3), dtype=np.float64)
    target_quat = np.zeros((M, 4), dtype=np.float64)
    achieved_pos = np.zeros((M, 3), dtype=np.float64)
    achieved_quat = np.zeros((M, 4), dtype=np.float64)
    cart_dev = np.zeros(M, dtype=np.float64)
    cart_dev_axes = np.zeros((M, 3), dtype=np.float64)
    self_coll = np.zeros(M, dtype=bool)
    t_global = np.zeros(M, dtype=np.float64)

    for k, qp in enumerate(q_trajectory):
        if qp.q_full is None:
            raise RuntimeError(
                f"[local_dp] q_trajectory[{k}] has q_full=None — Phase-2 "
                f"densify left a sentinel sample; cause={qp.failure_cause}"
            )
        q_full[k] = np.asarray(qp.q_full, dtype=np.float64)
        target_pos[k] = np.asarray(qp.target_position, dtype=np.float64)
        target_quat[k] = np.asarray(qp.target_quat_wxyz, dtype=np.float64)
        achieved_pos[k] = np.asarray(qp.achieved_position, dtype=np.float64)
        if qp.achieved_quat_wxyz is not None:
            achieved_quat[k] = np.asarray(qp.achieved_quat_wxyz, dtype=np.float64)
        cart_dev[k] = float(qp.cartesian_deviation)
        cart_dev_axes[k] = achieved_pos[k] - target_pos[k]
        t_global[k] = float(qp.t_global)
        self_coll[k] = bool(rc._is_self_colliding(q_full[k]))

    # Arm slice is known from warm.arm (passed in via arm_indices). Read
    # the 7-DOF arm block straight out of q_full.
    arm_lo, arm_hi = arm_indices[0], arm_indices[-1] + 1
    q_arm = q_full[:, arm_lo:arm_hi]

    dt = np.diff(t_global)
    if dt.size == 0:
        raise RuntimeError("[local_dp] dt array is empty (M==1)")
    # Phase 2 stamps t_global = k / hz, so dt should be uniform; guard
    # against zero entries (which would crash downstream FK + write).
    if (dt <= 0).any():
        floor = max(float(dt[dt > 0].min()) if (dt > 0).any() else 1e-3, 1e-3)
        dt = np.maximum(dt, floor)

    solve_stats = {
        "method": "local_dp",
        "ok": not dp_stats.get("infeasible", False) and not dp_stats.get("iteration_cap_exceeded", False),
        "status": "ok",
        "n_iters": int(dp_stats.get("iterations_run", 0)),
        "wall_seconds": float(
            t_pool_total + dp_stats.get("total_dp_time", 0.0)
            + dp_stats.get("total_check_time", 0.0)
        ),
        "final_cost": float(dp_stats.get("final_cost", 0.0)),
        "warmstart_summary": {
            "total_time": warm_total_time,
            "n_warm_samples": int(n_warm),
        },
        "final_summary": {
            "total_time": float(dt.sum()),
            "n_dense_samples": int(M),
            "max_cart_dev": float(cart_dev.max()) if cart_dev.size else 0.0,
            "n_collisions": int(self_coll.sum()),
        },
        # Collision audit fields the existing write/print path reads; the
        # local_dp pipeline doesn't reshape sphere-SDF margins so report
        # what we received from bounds verbatim.
        "d_safe_used": 0.0,
        "d_warn_used": 0.0,
        "min_pair_distance_after": None,
        "topp_floor_total_time": warm_total_time,
        "total_time_after": float(dt.sum()),
        # Local-DP-specific telemetry.
        "local_dp": {
            "n_sparse_layers": int(len(sparse_idx)),
            "sparse_indices": [int(i) for i in sparse_idx],
            "raw_pool_avg": (
                float(sum(pool_sizes_raw) / max(len(pool_sizes_raw), 1))
            ),
            "raw_pool_min": int(min(pool_sizes_raw)) if pool_sizes_raw else 0,
            "raw_pool_max": int(max(pool_sizes_raw)) if pool_sizes_raw else 0,
            "top_k": int(top_k),
            "total_ik_calls": int(total_ik_calls),
            "pool_build_seconds": float(t_pool_total),
            "dp": dp_stats,
            "densify": densify_stats,
            "n_phase2_failures": int(len(failures)),
        },
        # Fields the existing format_constraint_report touches; populated
        # with reasonable defaults so the print at _run_optimizer end
        # doesn't crash.
        "cart_cap_used": {
            "x": float(bounds.max_xy_deviation),
            "y": float(bounds.max_xy_deviation),
            "z": float(bounds.max_z_deviation),
        },
        "trust_radius_max_used": 0.0,
    }

    return SolverResult(
        q_arm=q_arm,
        dt=dt,
        achieved_pos=achieved_pos,
        achieved_quat_wxyz=achieved_quat,
        cart_dev=cart_dev,
        cart_dev_axes=cart_dev_axes,
        self_coll=self_coll,
        q_full=q_full,
        solve_stats=solve_stats,
        target_pos=target_pos,
        target_quat_wxyz=target_quat,
    )
