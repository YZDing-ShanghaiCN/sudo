"""I/O for the surface_cover_planner optimizer.

`load_warmstart` parses a `surface_cover_planner` output JSON (the kind written
by `tools/surface_cover_planner/plan.py:write_trajectory_json`) and pulls out
the warm-start arrays the optimizer needs: arm joints, cartesian targets, time
deltas, and the robot config that produced the trajectory.

`write_optimizer_json` writes a fresh optimizer JSON (its own schema, *not*
surface_cover_planner's) that captures solver iteration history, final stats,
and the optimized q-trajectory. The JSON shape is documented at the top of
the file so consumers don't have to re-derive it.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# geometry lives at the top level of `tools/surface_cover_planner/`; local_dp
# resolves it the same way (via the sys.path entry set by plan.py).
from geometry import tool_pose_to_tactile_pose


@dataclass
class WarmStart:
    """Warm-start bundle parsed out of a surface_cover_planner JSON.

    All numpy arrays are float64. Lengths satisfy:
      q_arm.shape == (N, 7)
      tcp_target_pos.shape == (N, 3)
      tcp_target_quat_wxyz.shape == (N, 4)
      dt.shape == (N - 1,)  (np.diff(t_global), guaranteed positive)
    """

    q_arm: np.ndarray
    q_full: np.ndarray  # (N, 16) — kept around for the output JSON
    tcp_target_pos: np.ndarray
    tcp_target_quat_wxyz: np.ndarray
    dt: np.ndarray
    t_global: np.ndarray  # (N,) — original control-rate timestamps
    arm: str  # "left_arm" | "right_arm"
    initial_q: list[float]  # length 16
    tool_offset_4x4: np.ndarray  # (4,4)
    joint_vel_limits: list[float] | None
    source_path: str
    # Optional surface-plane basis (unit vectors in world frame). Threaded
    # through from the planner's PlaneFrame so the local_dp refinement can
    # perturb targets in (u, v) without having to re-derive the plane.
    # None on legacy warm-start JSONs that pre-date the field.
    plane_u_axis: np.ndarray | None = None
    plane_v_axis: np.ndarray | None = None

    @classmethod
    def from_planner_result(
        cls,
        check_result: Any,
        *,
        arm: str,
        tool_offset_4x4: np.ndarray | None,
        joint_vel_limits: list[float] | None,
        source_path: str = "<in-memory planner result>",
        plane_u_axis: np.ndarray | None = None,
        plane_v_axis: np.ndarray | None = None,
    ) -> "WarmStart":
        """Build a WarmStart from a `robot_check.CheckResult` directly.

        Skips the JSON round-trip used by `load_warmstart`. Applies the same
        null-`q_full` drop as the JSON loader so the optimizer sees a clean
        contiguous sequence.
        """
        qt = [
            qp for qp in check_result.q_trajectory
            if qp.q_full is not None
        ]
        if not qt:
            raise ValueError(
                "planner CheckResult has no usable q_trajectory entries "
                "(every row has q_full=None)"
            )
        arm_attr = "q_left_arm" if arm == "left_arm" else "q_right_arm"
        q_arm_rows = [list(getattr(qp, arm_attr)) for qp in qt]
        q_full_rows = [list(qp.q_full) for qp in qt]
        pos_rows = [list(qp.target_position) for qp in qt]
        quat_rows = [list(qp.target_quat_wxyz) for qp in qt]
        t_rows = [float(qp.t_global) for qp in qt]

        q_arm = np.asarray(q_arm_rows, dtype=np.float64)
        q_full = np.asarray(q_full_rows, dtype=np.float64)
        pos = np.asarray(pos_rows, dtype=np.float64)
        quat = np.asarray(quat_rows, dtype=np.float64)
        t = np.asarray(t_rows, dtype=np.float64)

        if q_arm.shape[1] != 7 or q_full.shape[1] != 16:
            raise ValueError(
                f"planner result: expected (N,7) arm and (N,16) full DOFs, "
                f"got {q_arm.shape} and {q_full.shape}"
            )

        dt = np.diff(t)
        if dt.size > 0 and (dt <= 0).any():
            floor = max(float(dt[dt > 0].min()) if (dt > 0).any() else 1e-3, 1e-3)
            dt = np.maximum(dt, floor)

        return cls(
            q_arm=q_arm,
            q_full=q_full,
            tcp_target_pos=pos,
            tcp_target_quat_wxyz=quat,
            dt=dt,
            t_global=t,
            arm=arm,
            initial_q=[float(x) for x in q_full_rows[0]],
            tool_offset_4x4=(
                tool_offset_4x4 if tool_offset_4x4 is not None
                else np.eye(4, dtype=np.float64)
            ),
            joint_vel_limits=joint_vel_limits,
            source_path=source_path,
            plane_u_axis=(
                None if plane_u_axis is None
                else np.asarray(plane_u_axis, dtype=np.float64)
            ),
            plane_v_axis=(
                None if plane_v_axis is None
                else np.asarray(plane_v_axis, dtype=np.float64)
            ),
        )


def load_warmstart(path: str, decimate_stride: int = 1) -> WarmStart:
    """Parse a surface_cover_planner output JSON.

    `decimate_stride` keeps every Nth q_trajectory entry (always preserving
    the last entry so endpoint pinning still lines up with the real terminal
    state). Stride 1 = no subsampling. Useful because finite-diff trust-constr
    runtime scales with N², so the typical 4576-point Phase 4 output is
    intractable without decimation.
    """
    p = Path(path).resolve()
    with open(p) as f:
        data = json.load(f)

    if "q_trajectory" not in data:
        raise ValueError(
            f"warmstart {p} has no 'q_trajectory' field; was robot.enabled set "
            "when surface_cover_planner generated this trajectory?"
        )
    qt_raw = data["q_trajectory"]
    if not qt_raw:
        raise ValueError(f"warmstart {p} has empty q_trajectory")
    if decimate_stride < 1:
        raise ValueError(f"decimate_stride must be >= 1, got {decimate_stride}")

    md = data.get("metadata", {}) or {}
    rc_md = md.get("robot_check", {}) or {}

    # arm: not stored explicitly in the warm-start JSON, so infer from which
    # of q_left_arm / q_right_arm is populated. (surface_cover_planner only
    # plans one arm per run; the other stays at initial_q.)
    arm = _infer_arm(qt_raw)
    arm_field = "q_left_arm" if arm == "left_arm" else "q_right_arm"

    # Drop rows where q_full / q_arm are null (Phase-2 IK failures). The
    # earlier carry-forward behavior fabricated cartesian-deviation gaps the
    # optimizer couldn't satisfy, because it kept the previous q_full but
    # advanced target_position. Skipping cleanly is what the optimizer wants.
    qt: list[dict] = []
    n_dropped = 0
    for qp in qt_raw:
        if qp.get("q_full") is None or qp.get(arm_field) is None:
            n_dropped += 1
            continue
        qt.append(qp)
    if n_dropped:
        print(f"[load_warmstart] dropped {n_dropped} of {len(qt_raw)} rows "
              f"with null q_full / {arm_field}")
    if not qt:
        raise ValueError(
            f"warmstart {p}: every q_trajectory row has null q_full; "
            "warm-start is unusable"
        )

    if decimate_stride > 1:
        # Keep every stride-th surviving entry, force-include the last so
        # endpoint pinning still hits the real terminal pose.
        kept = list(range(0, len(qt), decimate_stride))
        if kept[-1] != len(qt) - 1:
            kept.append(len(qt) - 1)
        qt = [qt[i] for i in kept]
        print(f"[load_warmstart] decimated by stride={decimate_stride}: "
              f"{len(kept)} of {len(qt_raw) - n_dropped} cleaned steps kept")

    q_arm_rows: list[list[float]] = []
    q_full_rows: list[list[float]] = []
    pos_rows: list[list[float]] = []
    quat_rows: list[list[float]] = []
    t_rows: list[float] = []
    for qp in qt:
        q_full_rows.append([float(x) for x in qp["q_full"]])
        q_arm_rows.append([float(x) for x in qp[arm_field]])
        pos_rows.append([float(x) for x in qp["target_position"]])
        quat_rows.append([float(x) for x in qp["target_quat_wxyz"]])
        t_rows.append(float(qp["t_global"]))

    q_arm = np.asarray(q_arm_rows, dtype=np.float64)
    q_full = np.asarray(q_full_rows, dtype=np.float64)
    pos = np.asarray(pos_rows, dtype=np.float64)
    quat = np.asarray(quat_rows, dtype=np.float64)
    t = np.asarray(t_rows, dtype=np.float64)

    if q_arm.shape[1] != 7 or q_full.shape[1] != 16:
        raise ValueError(
            f"warmstart {p}: expected (N,7) arm and (N,16) full DOFs, got "
            f"{q_arm.shape} and {q_full.shape}"
        )

    dt = np.diff(t)
    if dt.size > 0 and (dt <= 0).any():
        # Recover by enforcing a small floor; surface_cover_planner Phase 4
        # re-stamps t to i*dt so this should never trigger, but be defensive.
        floor = max(float(dt[dt > 0].min()) if (dt > 0).any() else 1e-3, 1e-3)
        dt = np.maximum(dt, floor)

    tool_offset = np.asarray(md.get("tool_offset_4x4", np.eye(4).tolist()), dtype=np.float64)

    plane_u_raw = md.get("plane_u_axis")
    plane_v_raw = md.get("plane_v_axis")
    plane_u = (
        None if plane_u_raw is None
        else np.asarray(plane_u_raw, dtype=np.float64)
    )
    plane_v = (
        None if plane_v_raw is None
        else np.asarray(plane_v_raw, dtype=np.float64)
    )

    return WarmStart(
        q_arm=q_arm,
        q_full=q_full,
        tcp_target_pos=pos,
        tcp_target_quat_wxyz=quat,
        dt=dt,
        t_global=t,
        arm=arm,
        initial_q=[float(x) for x in q_full_rows[0]],
        tool_offset_4x4=tool_offset,
        joint_vel_limits=rc_md.get("joint_vel_limits"),
        source_path=str(p),
        plane_u_axis=plane_u,
        plane_v_axis=plane_v,
    )


def _infer_arm(qt: list[dict]) -> str:
    """Pick the arm whose q delta from the first sample is non-trivial."""
    for entry in qt:
        if entry.get("q_left_arm") is None or entry.get("q_right_arm") is None:
            continue
        l_arm = np.asarray(entry["q_left_arm"], dtype=np.float64)
        r_arm = np.asarray(entry["q_right_arm"], dtype=np.float64)
        l0 = np.asarray(qt[0]["q_left_arm"], dtype=np.float64)
        r0 = np.asarray(qt[0]["q_right_arm"], dtype=np.float64)
        l_delta = float(np.max(np.abs(l_arm - l0)))
        r_delta = float(np.max(np.abs(r_arm - r0)))
        if l_delta > 1e-4 and r_delta < 1e-4:
            return "left_arm"
        if r_delta > 1e-4 and l_delta < 1e-4:
            return "right_arm"
    # Fall back: surface_cover_planner's example uses left_arm.
    return "left_arm"


def _qpoint_dict(
    t_global: float,
    q_arm: np.ndarray,
    q_full: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    achieved_pos: np.ndarray,
    achieved_quat: np.ndarray,
    cartesian_deviation: float,
    cartesian_deviation_axes: np.ndarray,
    self_collision: bool,
    tool_offset_4x4: np.ndarray | None,
) -> dict:
    target_tactile_pos, target_tactile_quat = tool_pose_to_tactile_pose(
        target_pos, target_quat, tool_offset_4x4
    )
    achieved_tactile_pos, achieved_tactile_quat = tool_pose_to_tactile_pose(
        achieved_pos, achieved_quat, tool_offset_4x4
    )
    return {
        "t_global": float(t_global),
        "q_arm": [float(x) for x in q_arm],
        "q_full": [float(x) for x in q_full],
        "target_position": [float(x) for x in target_pos],
        "target_quat_wxyz": [float(x) for x in target_quat],
        "target_tactile_position": [float(x) for x in target_tactile_pos],
        "target_tactile_quat_wxyz": (
            None
            if target_tactile_quat is None
            else [float(x) for x in target_tactile_quat]
        ),
        "achieved_position": [float(x) for x in achieved_pos],
        "achieved_quat_wxyz": [float(x) for x in achieved_quat],
        "achieved_tactile_position": [float(x) for x in achieved_tactile_pos],
        "achieved_tactile_quat_wxyz": (
            None
            if achieved_tactile_quat is None
            else [float(x) for x in achieved_tactile_quat]
        ),
        "cartesian_deviation": float(cartesian_deviation),
        # Signed per-axis deviation (achieved - target), in the same frame as
        # target_position. Surface-cover plans pin z to the surface, so
        # `cartesian_deviation_axes[2]` is the out-of-plane drift.
        "cartesian_deviation_axes": [float(x) for x in cartesian_deviation_axes],
        "self_collision": bool(self_collision),
    }


def write_optimizer_json(
    out_path: str,
    *,
    warmstart: WarmStart,
    config_path: str,
    weights: dict,
    bounds: dict,
    solve_stats: dict,
    q_arm_final: np.ndarray,
    dt_final: np.ndarray,
    achieved_pos: np.ndarray,
    achieved_quat: np.ndarray,
    cart_dev: np.ndarray,
    cart_dev_axes: np.ndarray,
    self_coll: np.ndarray,
    q_full_final: np.ndarray,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = q_arm_final.shape[0]
    t = np.zeros(n, dtype=np.float64)
    t[1:] = np.cumsum(dt_final)

    payload: dict[str, Any] = {
        "metadata": {
            "warmstart_path": warmstart.source_path,
            "config_path": str(Path(config_path).resolve()),
            "generated_at_iso": _dt.datetime.now().isoformat(timespec="seconds"),
            "n_steps": int(n),
            "arm": warmstart.arm,
            "tool_offset_4x4": warmstart.tool_offset_4x4.tolist(),
            "weights": weights,
            "bounds": bounds,
        },
        "solve_stats": solve_stats,
        "q_trajectory": [
            _qpoint_dict(
                t_global=t[i],
                q_arm=q_arm_final[i],
                q_full=q_full_final[i],
                target_pos=warmstart.tcp_target_pos[i],
                target_quat=warmstart.tcp_target_quat_wxyz[i],
                achieved_pos=achieved_pos[i],
                achieved_quat=achieved_quat[i],
                cartesian_deviation=cart_dev[i],
                cartesian_deviation_axes=cart_dev_axes[i],
                self_collision=self_coll[i],
                tool_offset_4x4=warmstart.tool_offset_4x4,
            )
            for i in range(n)
        ],
    }

    with open(out, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
    print(f"wrote {n} q-trajectory steps -> {out}")


def format_constraint_report(
    *,
    solve_stats: dict,
    q_final: np.ndarray,
    q_warm: np.ndarray,
    bounds_cfg: dict,
    v_max: np.ndarray,
) -> str:
    """Build a multi-line human-readable final-constraint status block.

    Reports every hard constraint with SATISFIED/VIOLATED + the binding value
    vs cap. Endpoint pinning is structural (lo==hi), so it's mentioned but not
    checked. The `time speedup` row compares the final total time against the
    retimed warmstart total (`topp_floor_total_time`) — positive speedup is
    only possible now that `dt_lo = dt_min` (vs the old `dt_lo = dt_warm`).
    """
    tol = 1e-6

    def _ok(value: float, cap: float) -> str:
        return "SATISFIED" if value <= cap * (1.0 + tol) else "VIOLATED "

    lines: list[str] = []
    lines.append("[optimize] === final constraint status ===")

    vel_ratio = float(solve_stats.get("max_joint_vel_ratio", float("nan")))
    sat_suffix = "  ← SATURATED" if vel_ratio > 0.99 else ""
    lines.append(
        f"  joint velocity:   {_ok(vel_ratio, 1.0)}  max_ratio={vel_ratio:.4f} "
        f"(limit ≤ 1.000){sat_suffix}"
    )

    cart_used = solve_stats.get("cart_cap_used") or {}
    cfg_xy = float(bounds_cfg.get("max_xy_deviation", 0.030))
    cfg_z = float(bounds_cfg.get("max_z_deviation", 0.005))
    for axis, max_key, cfg_cap in (
        ("x", "max_dx", cfg_xy),
        ("y", "max_dy", cfg_xy),
        ("z", "max_dz", cfg_z),
    ):
        max_val = float(solve_stats.get(max_key, 0.0))
        used = float(cart_used.get(axis, cfg_cap))
        relaxed = used > cfg_cap * 1.001
        note = (
            f" (AUTO-RELAXED from {cfg_cap:.4f} m)" if relaxed else ""
        )
        lines.append(
            f"  cartesian |Δ{axis}|:   {_ok(max_val, used)}  "
            f"max={max_val:.4f} m (cap={used:.4f} m){note}"
        )

    min_d = solve_stats.get("min_pair_distance_after")
    d_safe_used = float(solve_stats.get("d_safe_used", 0.0))
    if min_d is None:
        lines.append(
            f"  self-collision:   SATISFIED  min_dist=> d_warn "
            f"(d_safe={d_safe_used:.4f} m)"
        )
    else:
        status = "SATISFIED" if float(min_d) >= d_safe_used - tol else "VIOLATED "
        lines.append(
            f"  self-collision:   {status}  min_dist={float(min_d):.4f} m "
            f"(d_safe={d_safe_used:.4f} m)"
        )

    trust_radius = float(bounds_cfg.get("trust_radius", 0.0))
    if trust_radius > 0.0 and q_final.shape == q_warm.shape:
        dq = np.abs(q_final - q_warm)
        max_inf = float(dq.max())
        per_wp_max = dq.max(axis=1)
        frac_active = float((per_wp_max > 0.95 * trust_radius).mean())
        binding = "  ← BINDING" if max_inf > 0.95 * trust_radius else ""
        lines.append(
            f"  trust radius:     max|Δq|_∞={max_inf:.5f} rad "
            f"(cap={trust_radius:.5f}, {frac_active * 100:.0f}% of wps "
            f"at ≥95% cap){binding}"
        )

    t_after = solve_stats.get("total_time_after")
    t_floor = solve_stats.get("topp_floor_total_time")
    if t_after is not None and t_floor is not None and float(t_floor) > 0.0:
        gap = float(t_floor) - float(t_after)
        gap_pct = gap / float(t_floor)
        gap_note = (
            "  ← warm dt was already near vel floor "
            "(widen trust_radius / raise v_max for further speedup)"
            if gap_pct < 0.005 else ""
        )
        lines.append(
            f"  time speedup:     T_total={float(t_after):.3f}s "
            f"(warmstart {float(t_floor):.3f}s, "
            f"speedup={gap:.3f}s = {gap_pct * 100:.1f}%){gap_note}"
        )

    if bounds_cfg.get("pin_endpoints", True):
        lines.append("  endpoints pinned: lo==hi enforced (structural)")

    lines.append(
        f"  solver:           ok={solve_stats.get('ok')} "
        f"status={solve_stats.get('status')!r} "
        f"iters={solve_stats.get('n_iters')} "
        f"wall={float(solve_stats.get('wall_seconds', 0.0)):.2f}s "
        f"final_cost={float(solve_stats.get('final_cost', 0.0)):.4g}"
    )
    return "\n".join(lines)
