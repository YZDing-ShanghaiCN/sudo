"""Boustrophedon surface-cover planner — entrypoint.

Usage:
    .venv/bin/python plan.py --config config_example.yaml

Reads a YAML config, generates a TCP-pose trajectory along the surface, and
writes the trajectory to a JSON file (configurable via output.json_path).

Routing is config-driven (no extra CLI flags):

    optimize.enabled: false      -> planner only (current default behavior).
    optimize.enabled: true       -> planner runs, then the optimizer subpackage
                                     refines the q-trajectory in-memory
                                     (no temp warm-start JSON written).
    input.warmstart_json: <path> -> skip the planner entirely; load that
                                     warm-start and run the optimizer.
                                     `optimize.enabled` is implied true.

When `robot.enabled: true` (planner phase), runs a feasibility check against
the HBMP robot agent (analytical IK + self/wall collision + joint-speed
limits) and embeds the per-densified-step joint trajectory in the JSON.
The optimizer phase additionally requires `cyipopt` (recommended) or
silently falls back to scipy's `trust-constr` when cyipopt isn't installed.
Both phases need the SudoDeploy docker container (ampl + ampl_motion).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from boustrophedon import PlanResult, plan_boustrophedon
from geometry import tool_pose_to_tactile_pose

try:
    from visualize import render_png
    _HAS_VISUALIZE = True
except Exception:  # noqa: BLE001
    _HAS_VISUALIZE = False


def _resolve_path(p: str, base_dir: Path) -> str:
    """Resolve a config-relative or repo-relative path to an absolute path."""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    cand = (base_dir / pp).resolve()
    if cand.exists():
        return str(cand)
    return str(pp.resolve())


def _parse_resample_limits(raw) -> list[float] | None:
    """Accept None, a scalar (broadcast to length 7), or a length-7 list."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return [float(raw)] * 7
    lst = [float(x) for x in raw]
    if len(lst) != 7:
        raise ValueError(
            f"robot.resample.joint_vel_limits must be a scalar or length-7 list "
            f"(arm DOFs); got length {len(lst)}"
        )
    return lst


def _parse_vel_limits(raw) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return [float(raw)] * 7
    lst = [float(x) for x in raw]
    if len(lst) != 7:
        raise ValueError(
            f"robot.joint_vel_limits must be scalar or length-7 list, got length {len(lst)}"
        )
    return lst


def _parse_accel_limits(raw) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return [float(raw)] * 7
    lst = [float(x) for x in raw]
    if len(lst) != 7:
        raise ValueError(
            f"robot.densify.joint_accel_limits must be a scalar or length-7 "
            f"list (arm DOFs); got length {len(lst)}"
        )
    return lst


def _parse_joint_preference_filters(
    raw,
) -> dict[int, tuple[float | None, float | None]] | None:
    """Validate the `robot.search.joint_preference_filters` YAML block.

    Expected shape: `{j_idx (int 0..6): [lo, hi]}` (either bound may be
    null for one-sided filtering). Returns None when raw is empty / None.
    """
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            "robot.search.joint_preference_filters must be a mapping "
            f"{{j_idx: [lo, hi]}}; got {type(raw).__name__}"
        )
    parsed: dict[int, tuple[float | None, float | None]] = {}
    for k, v in raw.items():
        j = int(k)
        if not (0 <= j <= 6):
            raise ValueError(
                f"robot.search.joint_preference_filters: index {j} out of "
                f"arm-DOF range [0, 6]"
            )
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(
                f"robot.search.joint_preference_filters[{j}]: expected "
                f"[lo, hi], got {v!r}"
            )
        lo = None if v[0] is None else float(v[0])
        hi = None if v[1] is None else float(v[1])
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"robot.search.joint_preference_filters[{j}]: lo ({lo}) > "
                f"hi ({hi})"
            )
        parsed[j] = (lo, hi)
    return parsed


def _parse_tool_offset(raw) -> np.ndarray | None:
    """Convert a tool_offset YAML block to a 4x4 matrix (or None for identity)."""
    if raw is None:
        return None
    from scipy.spatial.transform import Rotation as R

    t = np.asarray(raw.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    mode = raw.get("rotation_mode", "quat")
    if mode == "quat":
        q = np.asarray(raw.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    elif mode == "euler_xyz_deg":
        e = raw.get("euler_xyz_deg", [0.0, 0.0, 0.0])
        rot = R.from_euler("XYZ", e, degrees=True).as_matrix()
    elif mode == "matrix":
        rot = np.asarray(raw.get("matrix", np.eye(3).tolist()), dtype=np.float64)
    else:
        raise ValueError(f"unknown tool_offset.rotation_mode: {mode!r}")
    tof = np.eye(4, dtype=np.float64)
    tof[:3, :3] = rot
    tof[:3, 3] = t
    if np.allclose(tof, np.eye(4)):
        return None
    return tof


def _parse_attached_tool(raw, cfg_dir: Path) -> dict | None:
    """Validate / path-resolve a `robot.attached_tool` YAML block.

    Returns the original dict with `path` resolved against `cfg_dir`, or
    `None` when the block is absent / empty. Schema-checking is light —
    `robot_check._setup_collision_checker` reads the dict directly.
    """
    if not raw:
        return None
    if "path" not in raw:
        raise ValueError("robot.attached_tool requires a `path` field")
    out = dict(raw)
    out["path"] = _resolve_path(out["path"], cfg_dir)
    return out


def _parse_world_obstacles(raw, cfg_dir: Path) -> list[dict]:
    """Validate / path-resolve each entry of `robot.world_obstacles`.

    Each entry must have `name` and `path`; pose fields (`translation`,
    `rotation_mode`, `quat` / `euler_xyz_deg` / `matrix`) are forwarded
    verbatim. Returns `[]` when the block is null / empty.
    """
    if not raw:
        return []
    obstacles: list[dict] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if "name" not in entry or "path" not in entry:
            raise ValueError(
                f"robot.world_obstacles[{i}] requires `name` and `path` fields"
            )
        name = str(entry["name"])
        if name in seen:
            raise ValueError(
                f"robot.world_obstacles: duplicate name {name!r}"
            )
        seen.add(name)
        out = dict(entry)
        out["name"] = name
        out["path"] = _resolve_path(out["path"], cfg_dir)
        obstacles.append(out)
    return obstacles


def _qpoint_to_dict(qp, tool_offset_4x4: np.ndarray | None) -> dict:
    target_tactile_pos, target_tactile_quat = tool_pose_to_tactile_pose(
        qp.target_position, qp.target_quat_wxyz, tool_offset_4x4
    )
    achieved_tactile_pos, achieved_tactile_quat = tool_pose_to_tactile_pose(
        qp.achieved_position, qp.achieved_quat_wxyz, tool_offset_4x4
    )
    return {
        "t_global": qp.t_global,
        "user_wp_index": qp.user_wp_index,
        "sub_index": qp.sub_index,
        "target_position": qp.target_position.tolist(),
        "target_quat_wxyz": qp.target_quat_wxyz.tolist(),
        "target_tactile_position": target_tactile_pos.tolist(),
        "target_tactile_quat_wxyz": (
            None if target_tactile_quat is None else target_tactile_quat.tolist()
        ),
        "achieved_position": qp.achieved_position.tolist(),
        "achieved_quat_wxyz": (
            None if qp.achieved_quat_wxyz is None else qp.achieved_quat_wxyz.tolist()
        ),
        "achieved_tactile_position": achieved_tactile_pos.tolist(),
        "achieved_tactile_quat_wxyz": (
            None if achieved_tactile_quat is None else achieved_tactile_quat.tolist()
        ),
        "cartesian_deviation": qp.cartesian_deviation,
        "q_full": None if qp.q_full is None else qp.q_full.tolist(),
        "q_rail": qp.q_rail,
        "q_waist": qp.q_waist,
        "q_left_arm": qp.q_left_arm,
        "q_right_arm": qp.q_right_arm,
        "ok": qp.ok,
        "failure_cause": qp.failure_cause,
    }


def write_trajectory_json(
    result: PlanResult,
    obj_path: str,
    config_path: str,
    out_path: str,
    check_result=None,
    save_full_pools: bool = False,
) -> None:
    metadata = {
        "obj_path": obj_path,
        "config_path": config_path,
        "n_waypoints": len(result.waypoints),
        "generated_at_iso": _dt.datetime.now().isoformat(timespec="seconds"),
        "mesh_transform_matrix_4x4": result.mesh_transform_4x4.tolist(),
        "tool_offset_4x4": result.tool_offset_4x4.tolist(),
        "plane_origin": result.plane.origin.tolist(),
        "plane_normal": result.plane.normal.tolist(),
        "plane_u_axis": result.plane.u_axis.tolist(),
        "plane_v_axis": result.plane.v_axis.tolist(),
        "polygon_xyz": result.polygon_xyz.tolist(),
    }

    waypoints = []
    for w in result.waypoints:
        tactile_pos, tactile_quat = tool_pose_to_tactile_pose(
            w.position, w.quaternion_wxyz, result.tool_offset_4x4
        )
        waypoints.append({
            "name": w.name,
            "position": w.position.tolist(),
            "quaternion": w.quaternion_wxyz.tolist(),
            "tactile_position": tactile_pos.tolist(),
            "tactile_quat_wxyz": tactile_quat.tolist(),
            "surface_point": w.surface_point.tolist(),
            "surface_normal": w.surface_normal.tolist(),
            "pass_index": w.pass_index,
            "point_in_pass": w.point_in_pass,
            "kind": w.kind,
        })
    payload: dict = {"metadata": metadata, "waypoints": waypoints}

    if check_result is not None:
        metadata["robot_check"] = {
            "ok": check_result.ok,
            "n_q_waypoints": check_result.n_q_waypoints,
            "total_dist": check_result.total_dist,
            "dt": check_result.dt,
            "worst_joint_speed_ratio": check_result.worst_joint_speed_ratio,
            "worst_dof": check_result.worst_dof,
            "joint_speed_ok": check_result.joint_speed_ok,
            "failures": check_result.failures,
        }
        if check_result.refine_stats is not None:
            metadata["robot_check"]["refine"] = check_result.refine_stats
        if check_result.resample_stats is not None:
            metadata["robot_check"]["resample"] = check_result.resample_stats
        if check_result.densify_stats is not None:
            metadata["robot_check"]["densify"] = check_result.densify_stats
        payload["q_trajectory"] = [
            _qpoint_to_dict(qp, result.tool_offset_4x4)
            for qp in check_result.q_trajectory
        ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # The per-waypoint Phase 1a IK landscape lands in a sidecar file
    # <stem>_ik_pools.json so the main JSON stays small. The sidecar is
    # written whenever Phase 1a produced any `ik_pool_statuses` (so the
    # IK-grid PNG always renders, even when Phase 1 fails — the user needs
    # the feasibility landscape most exactly then). The heavy `ik_pools`
    # block (per-waypoint (q_arm, quat_wxyz) tuples) is included only when
    # `save_full_pools` is true (mirrors `robot.save_all_ik_results`), since
    # the pools themselves can be N_waypoints × hundreds of entries.
    # `CheckResult.ik_pools` is always populated in memory now (so the main
    # PNG can always render the colorbar) — this gate controls only the
    # sidecar payload. The main JSON records the sidecar's filename in
    # `metadata.robot_check.ik_pools_path` for traceability;
    # visualize.render_png / render_ik_grid_png load it transparently when
    # present.
    if check_result is not None and check_result.ik_pool_statuses is not None:
        ik_path = out.with_name(out.stem + "_ik_pools.json")
        ik_payload: dict = {
            "ik_pool_statuses": [
                [bool(s) for s in statuses]
                for statuses in check_result.ik_pool_statuses
            ]
        }
        if check_result.ik_pools is not None and save_full_pools:
            ik_payload["ik_pools"] = [
                {
                    "user_wp_index": i,
                    "candidates": [
                        {
                            "q_arm": [float(x) for x in q_arm],
                            "quat_wxyz": [float(x) for x in quat_wxyz],
                        }
                        for q_arm, quat_wxyz in pool
                    ],
                }
                for i, pool in enumerate(check_result.ik_pools)
            ]
        if check_result.ik_selected_flat_indices is not None:
            ik_payload["ik_selected_flat_indices"] = [
                int(x) for x in check_result.ik_selected_flat_indices
            ]
        if check_result.ik_pool_collision_checked is not None:
            ik_payload["ik_pool_collision_checked"] = [
                [int(c) for c in row]
                for row in check_result.ik_pool_collision_checked
            ]
        if check_result.collision_pair_stats:
            ik_payload["collision_pair_stats"] = check_result.collision_pair_stats
        with open(ik_path, "w") as f:
            json.dump(ik_payload, f, indent=2, allow_nan=True)
        metadata["robot_check"]["ik_pools_path"] = ik_path.name
        print(f"wrote ik_pools sidecar -> {ik_path}")

    with open(out, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
    print(f"wrote {len(result.waypoints)} waypoints -> {out}")


def _print_failures(failures: list) -> None:
    if not failures:
        return
    print(f"\n!! robot_check produced {len(failures)} failure(s):")
    for i, fail in enumerate(failures):
        kind = fail.get("kind", "?")
        wp = fail.get("user_wp_index")
        wp_name = fail.get("user_wp_name")
        sub = fail.get("sub_index")
        cause = fail.get("cause", "")
        prefix = f"  [{i}] kind={kind}"
        if wp is not None:
            prefix += f" wp={wp}"
            if wp_name:
                prefix += f" ({wp_name})"
        if sub is not None:
            prefix += f" sub={sub}"
        print(f"{prefix} :: {cause}")
        pos = fail.get("target_position")
        quat = fail.get("target_quat_wxyz")
        if pos is not None:
            pos_str = f"[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
            if quat is not None:
                quat_str = (
                    f"[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"
                )
                print(f"      tcp pose: pos={pos_str} quat(wxyz)={quat_str}")
            else:
                print(f"      tcp pose: pos={pos_str}")
        if "achieved_position" in fail:
            ach = fail["achieved_position"]
            print(f"      achieved: pos=[{ach[0]:.4f}, {ach[1]:.4f}, {ach[2]:.4f}]")
        if "branches" in fail:
            print(f"      branches: {fail['branches']}")


def _run_optimizer(
    warm,
    cfg: dict,
    config_path: Path,
    cfg_dir: Path,
    output_json_stub: Path,
    timestamp: str,
) -> bool:
    """Run the optimizer subpackage against a WarmStart bundle.

    Returns True on success, False otherwise. Output paths are derived from
    `output_json_stub` (which is the planner's output JSON for end-to-end mode,
    or `output.json_path` for optimizer-only mode); the optimizer JSON is
    written as `<stem>_optimized_<ts>.json` alongside.
    """
    from optimize import (
        OptBounds,
        OptWeights,
        RobotConfig,
        RobotModel,
        SolverOpts,
        solve,
        write_optimizer_json,
    )
    from optimize.retime import topp_retime, vel_ratios

    robot_cfg_yaml = cfg.get("robot", {}) or {}
    opt_cfg = cfg.get("optimize") or cfg.get("trajopt") or {}

    initial_q = robot_cfg_yaml.get("initial_q") or warm.initial_q
    if initial_q is None:
        raise ValueError(
            "robot.initial_q is required in the YAML or its warmstart JSON"
        )
    # Catch torso/off-arm drift between YAML and warmstart (caused silent
    # cart_dev inflation in the past). Refuse when they disagree > 1 mm.
    if robot_cfg_yaml.get("initial_q") is not None and warm.initial_q is not None:
        warm_arr = np.asarray(warm.initial_q, dtype=np.float64)
        yaml_arr = np.asarray(robot_cfg_yaml["initial_q"], dtype=np.float64)
        non_arm_mask = np.ones(16, dtype=bool)
        from optimize.robot_model import HBMP_COMPONENT_DOF_INDICES
        for j in HBMP_COMPONENT_DOF_INDICES[robot_cfg_yaml.get("arm", warm.arm)]:
            non_arm_mask[j] = False
        diff = float(np.abs(warm_arr[non_arm_mask] - yaml_arr[non_arm_mask]).max())
        if diff > 1e-3:
            raise ValueError(
                f"YAML robot.initial_q's torso / off-arm differs from "
                f"warmstart q_full[0] by max {diff:.4f}. Set robot.initial_q "
                f"to match q_full[0]:\n"
                f"  warm.initial_q = {list(warm_arr)}\n"
                f"  yaml.initial_q = {list(yaml_arr)}"
            )

    tool_offset_raw = robot_cfg_yaml.get("tool_offset")
    tool_offset = (
        _parse_tool_offset(tool_offset_raw)
        if tool_offset_raw is not None
        else (
            None
            if np.allclose(warm.tool_offset_4x4, np.eye(4))
            else warm.tool_offset_4x4
        )
    )
    vel_limits_raw = robot_cfg_yaml.get("joint_vel_limits", warm.joint_vel_limits)
    vel_limits = _parse_vel_limits(vel_limits_raw)

    coll_cfg = (opt_cfg.get("collision") or {})
    rcfg = RobotConfig(
        arm=robot_cfg_yaml.get("arm", warm.arm),
        initial_q=initial_q,
        hbmp_arm_left=str(robot_cfg_yaml.get("hbmp_arm_left", "hb11_left")),
        hbmp_arm_right=str(robot_cfg_yaml.get("hbmp_arm_right", "hb11_right")),
        hbmp_torso=str(robot_cfg_yaml.get("hbmp_torso", "hb11_torso")),
        wall=robot_cfg_yaml.get("wall"),
        tool_offset_4x4=tool_offset,
        q_lo_arm=robot_cfg_yaml.get("q_lo_arm"),
        q_hi_arm=robot_cfg_yaml.get("q_hi_arm"),
        joint_vel_limits=vel_limits,
        collision_d_safe=float(coll_cfg.get("d_safe", 0.005)),
        collision_d_warn=float(coll_cfg.get("d_warn", 0.050)),
        collision_top_k=int(coll_cfg.get("top_k", 1)),
    )

    print(f"[optimize] building robot model (arm={rcfg.arm})")
    model = RobotModel(rcfg)

    # TOPP-style pre-retime: stretch dt where the warm-start violates v_max
    # so the SCO starts inside the velocity-feasible set. q is unchanged.
    pre_dt = warm.dt
    pre_worst = float(vel_ratios(warm.q_arm, pre_dt, model.v_max).max())
    dt_min_for_retime = float((opt_cfg.get("bounds") or {}).get("dt_min", 0.01))
    retimed_dt = topp_retime(warm.q_arm, pre_dt, model.v_max, dt_min_for_retime)
    post_worst = float(vel_ratios(warm.q_arm, retimed_dt, model.v_max).max())
    print(
        f"[optimize] pre-retime: total_time {pre_dt.sum():.3f}s -> "
        f"{retimed_dt.sum():.3f}s; worst vel ratio {pre_worst:.3f} -> "
        f"{post_worst:.3f}"
    )
    warm = warm.__class__(
        q_arm=warm.q_arm, q_full=warm.q_full,
        tcp_target_pos=warm.tcp_target_pos,
        tcp_target_quat_wxyz=warm.tcp_target_quat_wxyz,
        dt=retimed_dt,
        t_global=np.concatenate([[0.0], np.cumsum(retimed_dt)]),
        arm=warm.arm, initial_q=warm.initial_q,
        tool_offset_4x4=warm.tool_offset_4x4,
        joint_vel_limits=warm.joint_vel_limits,
        source_path=warm.source_path,
        plane_u_axis=warm.plane_u_axis,
        plane_v_axis=warm.plane_v_axis,
    )

    weights_dict = (opt_cfg.get("weights") or {})
    weights = OptWeights(
        time=float(weights_dict.get("time", 1.0)),
        jerk=float(weights_dict.get("jerk", 10.0)),
        track_pos=float(weights_dict.get("track_pos", 100.0)),
        track_quat=float(weights_dict.get("track_quat", 0.1)),
        collision=float(weights_dict.get("collision", 0.0)),
    )
    bounds_dict = opt_cfg.get("bounds") or {}
    if "max_cartesian_deviation" in bounds_dict:
        print(
            "[optimize] WARNING: 'max_cartesian_deviation' is deprecated; "
            "replace with 'max_xy_deviation' and 'max_z_deviation'."
        )
    bounds = OptBounds(
        trust_radius=float(bounds_dict.get("trust_radius", 0.05)),
        dt_min=float(bounds_dict.get("dt_min", 0.01)),
        dt_max=float(bounds_dict.get("dt_max", 1.0)),
        max_xy_deviation=float(bounds_dict.get("max_xy_deviation", 0.030)),
        max_z_deviation=float(bounds_dict.get("max_z_deviation", 0.005)),
        pin_endpoints=bool(bounds_dict.get("pin_endpoints", True)),
    )
    solver_opts = SolverOpts(
        method=str((opt_cfg.get("solver") or {}).get("method", "ipopt")),
        max_iters=int((opt_cfg.get("solver") or {}).get("max_iters", 100)),
        xtol=float((opt_cfg.get("solver") or {}).get("xtol", 1e-6)),
        gtol=float((opt_cfg.get("solver") or {}).get("gtol", 1e-6)),
        verbose=int((opt_cfg.get("solver") or {}).get("verbose", 1)),
    )

    if solver_opts.method == "local_dp":
        from optimize import solve_local_dp

        lp_cfg = opt_cfg.get("local_dp") or {}
        # Phase-2 re-densify needs the planner-equivalent hz / ee_speed
        # to set the slerp step + uniform-grid emission rate. Read from
        # `robot:` block (the planner side that wrote the warmstart),
        # falling back to typical defaults.
        hz = float(robot_cfg_yaml.get("hz", 30.0))
        ee_speed = float(robot_cfg_yaml.get("ee_speed", 0.05))
        max_cd = float(
            robot_cfg_yaml.get("max_cartesian_deviation", bounds.max_xy_deviation)
        )
        result = solve_local_dp(
            warm=warm,
            model=model,
            bounds=bounds,
            lp_cfg=lp_cfg,
            hz=hz,
            ee_speed=ee_speed,
            max_cartesian_deviation=max_cd,
        )
    else:
        result = solve(
            q_warm=warm.q_arm,
            dt_warm=warm.dt,
            target_pos=warm.tcp_target_pos,
            target_quat_wxyz=warm.tcp_target_quat_wxyz,
            model=model,
            weights=weights,
            bounds=bounds,
            solver_opts=solver_opts,
        )

    s = result.solve_stats
    print(
        f"[optimize] ok={s['ok']} status={s['status']!s} iters={s['n_iters']} "
        f"wall={s['wall_seconds']:.2f}s final_cost={s['final_cost']:.4g}"
    )
    print(f"[optimize] warmstart_summary={s['warmstart_summary']}")
    print(f"[optimize] final_summary={s['final_summary']}")
    print(
        f"[optimize] collision: d_safe={s['d_safe_used']:.4f} "
        f"d_warn={s['d_warn_used']:.4f} "
        f"min_pair_distance_after={s['min_pair_distance_after']} "
        f"(None means every pair stayed past d_warn)"
    )

    if solver_opts.method == "local_dp":
        ldp = s.get("local_dp", {})
        print(
            f"[optimize] local_dp: sparse={ldp.get('n_sparse_layers')} "
            f"raw_pool_avg={ldp.get('raw_pool_avg', 0):.0f} "
            f"top_k={ldp.get('top_k')} ik_calls={ldp.get('total_ik_calls')} "
            f"dp_iters={ldp.get('dp', {}).get('iterations_run')} "
            f"phase2_failures={ldp.get('n_phase2_failures')}"
        )
    else:
        from optimize.io import format_constraint_report
        print(
            format_constraint_report(
                solve_stats=s,
                q_final=result.q_arm,
                q_warm=warm.q_arm,
                bounds_cfg={
                    "trust_radius": bounds.trust_radius,
                    "max_xy_deviation": bounds.max_xy_deviation,
                    "max_z_deviation": bounds.max_z_deviation,
                    "pin_endpoints": bounds.pin_endpoints,
                },
                v_max=model.v_max,
            )
        )

    # Output path: <planner_stem>_optimized_<ts>.json (already-timestamped
    # planner JSON keeps its name; the optimizer JSON sits next to it).
    out_path = output_json_stub.with_name(
        f"{output_json_stub.stem}_optimized_{timestamp}.json"
    )
    # `write_optimizer_json` indexes `warmstart.tcp_target_pos[i]` per row, so
    # when local_dp re-densifies onto a different-length grid we need a
    # warmstart bundle whose target arrays match the new trajectory length.
    # The IPOPT / trust-constr paths leave `result.target_pos` as None and
    # fall through to the original warm bundle (one-to-one with q_arm).
    write_warm = warm
    if result.target_pos is not None and result.target_quat_wxyz is not None:
        write_warm = warm.__class__(
            q_arm=result.q_arm,
            q_full=result.q_full,
            tcp_target_pos=result.target_pos,
            tcp_target_quat_wxyz=result.target_quat_wxyz,
            dt=result.dt,
            t_global=np.concatenate([[0.0], np.cumsum(result.dt)]),
            arm=warm.arm,
            initial_q=warm.initial_q,
            tool_offset_4x4=warm.tool_offset_4x4,
            joint_vel_limits=warm.joint_vel_limits,
            source_path=warm.source_path,
            plane_u_axis=warm.plane_u_axis,
            plane_v_axis=warm.plane_v_axis,
        )
    write_optimizer_json(
        out_path=str(out_path),
        warmstart=write_warm,
        config_path=str(config_path),
        weights={
            "time": weights.time, "jerk": weights.jerk,
            "track_pos": weights.track_pos, "track_quat": weights.track_quat,
            "collision": weights.collision,
        },
        bounds={
            "trust_radius": bounds.trust_radius, "dt_min": bounds.dt_min,
            "dt_max": bounds.dt_max,
            "max_xy_deviation": bounds.max_xy_deviation,
            "max_z_deviation": bounds.max_z_deviation,
            "pin_endpoints": bounds.pin_endpoints,
            "collision": {
                "d_safe": rcfg.collision_d_safe,
                "d_warn": rcfg.collision_d_warn,
                "top_k": rcfg.collision_top_k,
            },
        },
        solve_stats=s,
        q_arm_final=result.q_arm,
        dt_final=result.dt,
        achieved_pos=result.achieved_pos,
        achieved_quat=result.achieved_quat_wxyz,
        cart_dev=result.cart_dev,
        cart_dev_axes=result.cart_dev_axes,
        self_coll=result.self_coll,
        q_full_final=result.q_full,
    )

    # Optional densification: linear-interp the optimized (q_arm, dt) onto a
    # uniform `output.densify_dt` grid, re-evaluate FK / collision per dense
    # step, and emit a sibling `<stem>_densified.json`. Mirrors the standalone
    # `optimize/densify_cli.py` pipeline so a single `plan.py` run produces
    # both the coarse optimizer JSON and the control-rate trajectory.
    densify_dt_raw = (cfg.get("output") or {}).get("densify_dt")
    dense_q_arm_for_viz: np.ndarray | None = None
    dense_dt_for_viz: float | None = None
    if densify_dt_raw is not None:
        try:
            from optimize.densify import densify_joint_space, densify_targets

            target_dt = float(densify_dt_raw)
            if target_dt <= 0:
                raise ValueError(
                    f"output.densify_dt must be > 0; got {target_dt}"
                )

            dense = densify_joint_space(result.q_arm, result.dt, target_dt)
            dense_q_arm_for_viz = dense.q_arm
            dense_dt_for_viz = target_dt
            dense_targets_pos, dense_targets_quat = densify_targets(
                write_warm.tcp_target_pos, write_warm.tcp_target_quat_wxyz,
                result.dt, target_dt,
            )
            n_dense = dense.q_arm.shape[0]
            print(
                f"[optimize] densify: N={result.q_arm.shape[0]} -> "
                f"N_dense={n_dense} at dt={target_dt:.4f}s "
                f"({1.0 / target_dt:.2f} Hz)"
            )

            dense_pos = np.empty((n_dense, 3), dtype=np.float64)
            dense_quat = np.empty((n_dense, 4), dtype=np.float64)
            dense_coll = np.zeros(n_dense, dtype=bool)
            dense_q_full = np.empty((n_dense, 16), dtype=np.float64)
            for i in range(n_dense):
                p, qw = model.fk_pose(dense.q_arm[i])
                dense_pos[i] = p
                dense_quat[i] = qw
                dense_q_full[i] = model.assemble_q_full(dense.q_arm[i])
                dense_coll[i] = model.self_collides(dense.q_arm[i])

            cart_dev_dense = np.linalg.norm(
                dense_pos - dense_targets_pos, axis=1
            )
            vel_ratios_dense = np.abs(np.diff(dense.q_arm, axis=0)) / (
                dense.dt[:, None] * model.v_max[None, :] + 1e-12
            )
            densify_stats = {
                "target_dt": float(target_dt),
                "target_hz": float(1.0 / target_dt),
                "n_dense": int(n_dense),
                "total_time": float(dense.t_global[-1]),
                "max_cart_dev": float(cart_dev_dense.max()),
                "max_joint_vel_ratio": float(vel_ratios_dense.max()),
                "n_collisions": int(dense_coll.sum()),
                "collision_evaluated": True,
            }
            print(f"[optimize] densify stats: {densify_stats}")

            tool_offset_meta = (
                rcfg.tool_offset_4x4.tolist()
                if rcfg.tool_offset_4x4 is not None
                else np.eye(4).tolist()
            )
            dense_payload = {
                "metadata": {
                    "source_trajopt_json": str(out_path.name),
                    "config_path": str(config_path),
                    "generated_at_iso": _dt.datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "arm": rcfg.arm,
                    "tool_offset_4x4": tool_offset_meta,
                    "densify_stats": densify_stats,
                },
                "q_trajectory": [
                    {
                        "t_global": float(dense.t_global[i]),
                        "q_arm": [float(x) for x in dense.q_arm[i]],
                        "q_full": [float(x) for x in dense_q_full[i]],
                        "target_position": [float(x) for x in dense_targets_pos[i]],
                        "target_quat_wxyz": [float(x) for x in dense_targets_quat[i]],
                        "achieved_position": [float(x) for x in dense_pos[i]],
                        "achieved_quat_wxyz": [float(x) for x in dense_quat[i]],
                        "cartesian_deviation": float(cart_dev_dense[i]),
                        "self_collision": bool(dense_coll[i]),
                        "parent_segment": int(dense.parent_segment[i]),
                    }
                    for i in range(n_dense)
                ],
            }
            dense_path = out_path.with_name(out_path.stem + "_densified.json")
            dense_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dense_path, "w") as f:
                json.dump(dense_payload, f, indent=2, allow_nan=True)
            print(f"[optimize] wrote {n_dense} dense steps -> {dense_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[optimize] densify skipped: {e}")

    # Visualization PNGs. Each renderer is isolated in its own try/except so
    # a missing solve_stats field in one (e.g. iter_history for `local_dp`,
    # which doesn't populate it) doesn't suppress the others.
    try:
        from optimize.visualize import (
            render_cart_deviation_png,
            render_iter_history_png,
            render_min_distance_png,
            render_q_traj_png,
            render_vel_accel_png,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[optimize] visualize import failed, skipping all PNGs: {e}")
    else:
        png_base = out_path.with_suffix("")

        try:
            render_iter_history_png(str(out_path), str(png_base) + "_iter.png")
        except Exception as e:  # noqa: BLE001
            print(f"[optimize] iter PNG skipped: {e}")

        try:
            render_q_traj_png(
                str(out_path),
                warm.q_arm,
                warm.t_global,
                str(png_base) + "_qtraj.png",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[optimize] qtraj PNG skipped: {e}")

        try:
            render_cart_deviation_png(
                str(out_path),
                warm.q_arm,
                warm.t_global,
                warm.tcp_target_pos,
                str(png_base) + "_cart.png",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[optimize] cart PNG skipped: {e}")

        try:
            render_min_distance_png(str(out_path), str(png_base) + "_dist.png")
        except Exception as e:  # noqa: BLE001
            print(f"[optimize] dist PNG skipped: {e}")

        # Joint vel/accel PNG. Prefer densified q_arm (uniform dt, smoother
        # curves); fall back to the sparse SCO output when densification is
        # off or failed.
        if dense_q_arm_for_viz is not None and dense_dt_for_viz is not None:
            q_opt_for_viz = dense_q_arm_for_viz
            dt_opt_for_viz: np.ndarray | float = dense_dt_for_viz
        else:
            q_opt_for_viz = result.q_arm
            dt_opt_for_viz = result.dt
        try:
            render_vel_accel_png(
                q_warm=warm.q_arm,
                dt_warm=warm.dt,
                q_opt=q_opt_for_viz,
                dt_opt=dt_opt_for_viz,
                v_max=model.v_max,
                out_png=str(png_base) + "_vel_accel.png",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[optimize] vel_accel PNG skipped: {e}")

    return bool(s.get("ok"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Boustrophedon surface-cover planner.")
    parser.add_argument("--config", required=True, help="path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = config_path.parent

    np.set_printoptions(suppress=True, precision=4)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    input_cfg = cfg.get("input") or {}
    optimize_cfg = cfg.get("optimize") or cfg.get("trajopt") or {}
    warmstart_json = input_cfg.get("warmstart_json")
    decimate_stride = int(input_cfg.get("decimate_stride", 1))

    # ---- Optimizer-only mode: skip planner, load WarmStart from JSON ----
    if warmstart_json:
        from optimize import WarmStart, load_warmstart

        ws_path = _resolve_path(warmstart_json, cfg_dir)
        print(f"[plan] optimizer-only: loading warmstart from {ws_path} "
              f"(stride={decimate_stride})")
        warm = load_warmstart(ws_path, decimate_stride=decimate_stride)
        print(f"[plan] warmstart: arm={warm.arm} N={warm.q_arm.shape[0]} "
              f"total_time={warm.dt.sum():.3f}s")

        out_raw = (cfg.get("output") or {}).get("json_path", "outputs/optimized.json")
        out_stub = Path(_resolve_path(out_raw, cfg_dir))
        ok = _run_optimizer(
            warm=warm,
            cfg=cfg,
            config_path=config_path,
            cfg_dir=cfg_dir,
            output_json_stub=out_stub,
            timestamp=timestamp,
        )
        sys.exit(0 if ok else 2)

    # ---- Planner mode (with optional end-to-end optimizer) ----
    cfg["mesh"]["mesh_path"] = _resolve_path(cfg["mesh"]["mesh_path"], cfg_dir)
    resolved_json = Path(_resolve_path(cfg["output"]["json_path"], cfg_dir))
    cfg["output"]["json_path"] = str(
        resolved_json.with_name(f"{resolved_json.stem}_{timestamp}.json")
    )

    print(f"mesh (visualization only): {cfg['mesh']['mesh_path']}")
    print(f"polygon ({len(cfg['region']['polygon'])} pts), "
          f"step_over={cfg['pattern']['step_over']} m, "
          f"point_spacing={cfg['pattern']['point_spacing']} m")

    result = plan_boustrophedon(cfg)

    print(f"generated {len(result.waypoints)} waypoints across "
          f"{result.waypoints[-1].pass_index + 1 if result.waypoints else 0} passes")

    check_result = None
    robot_cfg = cfg.get("robot", {}) or {}
    if robot_cfg.get("enabled", False):
        from robot_check import RobotCheckConfig, check_trajectory

        rc = RobotCheckConfig(
            enabled=True,
            arm=robot_cfg.get("arm", "right_arm"),
            hz=float(robot_cfg.get("hz", 30.0)),
            ee_speed=float(robot_cfg.get("ee_speed", 0.05)),
            min_waypoints=int(robot_cfg.get("min_waypoints", 2)),
            nb_redundant_search=int(robot_cfg.get("nb_redundant_search", 512)),
            max_cartesian_deviation=float(robot_cfg.get("max_cartesian_deviation", 0.002)),
            ignore_rotation=bool(robot_cfg.get("ignore_rotation", False)),
            hbmp_arm_left=str(robot_cfg.get("hbmp_arm_left", "hb11_left")),
            hbmp_arm_right=str(robot_cfg.get("hbmp_arm_right", "hb11_right")),
            hbmp_torso=str(robot_cfg.get("hbmp_torso", "hb11_torso")),
            wall=robot_cfg.get("wall", {}) or {},
            attached_tool=_parse_attached_tool(robot_cfg.get("attached_tool"), "/home/user"),
            world_obstacles=_parse_world_obstacles(
                robot_cfg.get("world_obstacles") or [], "/home/user"
            ),
            initial_q=robot_cfg.get("initial_q"),
            joint_vel_limits=robot_cfg.get("joint_vel_limits"),
            yaw_search_offsets_deg=robot_cfg.get("yaw_search_offsets_deg"),
            pitch_search_offsets_deg=robot_cfg.get("pitch_search_offsets_deg"),
            roll_search_offsets_deg=robot_cfg.get("roll_search_offsets_deg"),
            search_method=str(
                (robot_cfg.get("search") or {}).get("search_method", "astar")
            ),
            max_iterations=int(
                (robot_cfg.get("search") or {}).get("max_iterations", 20)
            ),
            heuristic_weight=float(
                (robot_cfg.get("search") or {}).get("heuristic_weight", 5.0)
            ),
            heuristic_base_cost=float(
                (robot_cfg.get("search") or {}).get("heuristic_base_cost", 0.1)
            ),
            jump_threshold=float(
                (robot_cfg.get("search") or {}).get("jump_threshold", 1.5)
            ),
            accel_weight=float(
                (robot_cfg.get("search") or {}).get("accel_weight", 0.0)
            ),
            keep_init=bool(
                (robot_cfg.get("search") or {}).get("keep_init", False)
            ),
            joint_preference_filters=_parse_joint_preference_filters(
                (robot_cfg.get("search") or {}).get("joint_preference_filters")
            ),
            tool_offset=(
                None
                if np.allclose(result.tool_offset_4x4, np.eye(4))
                else result.tool_offset_4x4
            ),
            save_all_ik_results=bool(robot_cfg.get("save_all_ik_results", False)),
            resample_enabled=bool((robot_cfg.get("resample") or {}).get("enabled", False)),
            resample_check_collision=bool(
                (robot_cfg.get("resample") or {}).get("check_collision", False)
            ),
            resample_max_expansion=int(
                (robot_cfg.get("resample") or {}).get("max_expansion", 1000)
            ),
            resample_joint_vel_limits=_parse_resample_limits(
                (robot_cfg.get("resample") or {}).get("joint_vel_limits")
            ),
            refine_enabled=bool((robot_cfg.get("refine") or {}).get("enabled", False)),
            refine_jump_threshold=float(
                (robot_cfg.get("refine") or {}).get("jump_threshold", 0.1)
            ),
            refine_region_buffer=int(
                (robot_cfg.get("refine") or {}).get("region_buffer", 2)
            ),
            refine_yaw_offsets_deg=(robot_cfg.get("refine") or {}).get("yaw_offsets_deg"),
            refine_pitch_offsets_deg=(robot_cfg.get("refine") or {}).get("pitch_offsets_deg"),
            refine_xy_offsets_m=(robot_cfg.get("refine") or {}).get("xy_offsets_m"),
            refine_nb_redundant_search=int(
                (robot_cfg.get("refine") or {}).get("nb_redundant_search", 8)
            ),
            plane_u_axis=(
                result.plane.u_axis
                if (robot_cfg.get("refine") or {}).get("enabled", False)
                else None
            ),
            plane_v_axis=(
                result.plane.v_axis
                if (robot_cfg.get("refine") or {}).get("enabled", False)
                else None
            ),
            densify_method=str(
                (robot_cfg.get("densify") or {}).get("method", "topp_smooth")
            ),
            densify_joint_accel_limits=_parse_accel_limits(
                (robot_cfg.get("densify") or {}).get("joint_accel_limits")
            ),
            densify_vel_headroom=float(
                (robot_cfg.get("densify") or {}).get("vel_headroom", 0.05)
            ),
            densify_accel_headroom=float(
                (robot_cfg.get("densify") or {}).get("accel_headroom", 0.05)
            ),
            densify_dt_min=float(
                (robot_cfg.get("densify") or {}).get("dt_min", 0.005)
            ),
        )
        print(f"\n[robot_check] arm={rc.arm} hz={rc.hz} ee_speed={rc.ee_speed} m/s")
        check_result = check_trajectory(result.waypoints, rc)
        print(
            f"[robot_check] ok={check_result.ok} "
            f"n_q_waypoints={check_result.n_q_waypoints} "
            f"total_dist={check_result.total_dist:.4f}m "
            f"worst_joint_speed_ratio={check_result.worst_joint_speed_ratio:.3f} "
            f"(dof {check_result.worst_dof})"
        )
        _print_failures(check_result.failures)

    json_path = cfg["output"]["json_path"]
    save_full_pools = bool(robot_cfg.get("save_all_ik_results", False))
    write_trajectory_json(
        result=result,
        obj_path=cfg["mesh"]["mesh_path"],
        config_path=str(config_path),
        out_path=json_path,
        check_result=check_result,
        save_full_pools=save_full_pools,
    )

    if _HAS_VISUALIZE:
        png_path = str(Path(json_path).with_suffix(".png"))
        pool_sizes = None
        if check_result is not None and check_result.ik_pools is not None:
            pool_sizes = np.asarray(
                [len(p) for p in check_result.ik_pools], dtype=np.int32
            )
        try:
            render_png(
                mesh_path=cfg["mesh"]["mesh_path"],
                trajectory_path=json_path,
                out_png=png_path,
                pool_sizes=pool_sizes,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[visualize] render_png skipped: {e}")

        if check_result is not None:
            try:
                from visualize import render_joint_jump_png

                jump_png_path = str(
                    Path(json_path).with_name(Path(json_path).stem + "_joint_jump.png")
                )
                render_joint_jump_png(
                    trajectory_path=json_path,
                    out_png=jump_png_path,
                    jump_threshold=float(
                        (robot_cfg.get("search") or {}).get("jump_threshold", 1.5)
                    ),
                )
            except Exception as e:  # noqa: BLE001
                print(f"[visualize] render_joint_jump_png skipped: {e}")

            if check_result.ik_pool_statuses is not None:
                try:
                    from visualize import render_ik_grid_png

                    grid_png_path = str(
                        Path(json_path).with_name(Path(json_path).stem + "_ik_grid.png")
                    )
                    render_ik_grid_png(
                        trajectory_path=json_path,
                        out_png=grid_png_path,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[visualize] render_ik_grid_png skipped: {e}")

            if check_result.ik_pool_collision_checked is not None:
                cf_counts = np.asarray(
                    [
                        sum(1 for c in row if c == 1)
                        for row in check_result.ik_pool_collision_checked
                    ],
                    dtype=np.int32,
                )
                # Red-X cross flag: every dedup'd IK candidate at this
                # waypoint was collision-tested AND every test came back
                # collided — Phase 1 has no fallback here.
                cross_mask = np.zeros(
                    len(check_result.ik_pool_collision_checked), dtype=bool,
                )
                if check_result.ik_pools is not None:
                    for i, row in enumerate(check_result.ik_pool_collision_checked):
                        n_clean = sum(1 for c in row if c == 1)
                        n_collided = sum(1 for c in row if c == 2)
                        n_dedup = len(check_result.ik_pools[i])
                        if n_dedup > 0 and n_clean == 0 and n_collided == n_dedup:
                            cross_mask[i] = True
                try:
                    render_png(
                        mesh_path=cfg["mesh"]["mesh_path"],
                        trajectory_path=json_path,
                        out_png=str(
                            Path(json_path).with_name(
                                Path(json_path).stem + "_collision_free.png"
                            )
                        ),
                        pool_sizes=cf_counts,
                        colorbar_label=(
                            "# collision-free IK candidates "
                            "(lazy lower bound, per waypoint)"
                        ),
                        cross_mask=cross_mask,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[visualize] render_png (collision_free) skipped: {e}")
    else:
        print("[visualize] trimesh/matplotlib unavailable — skipping PNG render")

    # ---- Optional end-to-end optimizer phase ----
    optimizer_ok = True
    if optimize_cfg.get("enabled", False):
        if check_result is None:
            print(
                "[optimize] skipped: optimize.enabled=true requires "
                "robot.enabled=true (need a q_trajectory to refine)."
            )
        else:
            from optimize import WarmStart

            warm = WarmStart.from_planner_result(
                check_result,
                arm=robot_cfg.get("arm", "left_arm"),
                tool_offset_4x4=result.tool_offset_4x4,
                joint_vel_limits=robot_cfg.get("joint_vel_limits"),
                source_path=json_path,
                plane_u_axis=result.plane.u_axis,
                plane_v_axis=result.plane.v_axis,
            )
            optimizer_ok = _run_optimizer(
                warm=warm,
                cfg=cfg,
                config_path=config_path,
                cfg_dir=cfg_dir,
                output_json_stub=Path(json_path),
                timestamp=timestamp,
            )

    if check_result is not None and not check_result.ok:
        sys.exit(2)
    if not optimizer_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
